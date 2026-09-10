"""Day/time scan for the JFK-LHR supersonic segment. For every candidate
departure, march across the supersonic legs picking the best-wind flight
level at each one (limits.best_level), and rank candidates by total
supersonic-segment time. Reads only the .npz written by era5.reduce_to_legs;
never touches the netCDF.

march_legs is the shared per-leg march: run_search calls it for ~31,000
candidates at once and collapses the result to means, report.run_report
calls it for a single candidate and keeps every per-leg quantity.
"""
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from concopt import limits
from concopt.atmos import KT_TO_MS, fl_to_pressure, isa, speed_of_sound
from concopt.era5 import ARCHIVE_START, load_legs_npz
from concopt.route import build_legs, parse_pln, supersonic_segment

NY_TZ = ZoneInfo("America/New_York")
DEPARTURE_LOCAL_HOURS = range(8, 15)  # 08:00..14:00 local, inclusive

NM_TO_M = 1852.0

# Linear burn along the supersonic segment, keyed on cum_nm (not time) --
# drives ceiling_ft, which is what actually keeps the optimiser off levels
# the aircraft can't hold. Does not change max_tas above FL430.
WEIGHT_ACCEL_T = 165.0
WEIGHT_DECEL_T = 135.0

# best_level picks from this 16-level grid (1000 ft / FL10 steps) rather than
# the 4 raw ERA5 pressure levels -- interpolated per leg below, not stored in
# the npz (16 levels there would be ~350 MB vs ~32 MB for 4). FL450-FL600
# sits inside the ERA5 mandatory-level span (150-70 hPa = FL446-FL605), so
# every target is bracketed -- see the assert in march_legs.
TARGET_FL = np.arange(450.0, 601.0, 10.0)

# Time from brakes release to the accel point (start of the first supersonic
# leg -- ~120 nm / ~20 min into the flight at LINND). accumulated_s starts
# here rather than at 0 so leg weather is sampled at the correct clock time
# and the total elapsed time is usable by Phase 4. Configurable from the CLI.
DEPARTURE_TO_ACCEL_S = 20.0 * 60.0


def candidate_departures():
    """Every date from era5.ARCHIVE_START to today, at 08:00-14:00
    America/New_York (7 candidates/day) -- about 4,419 x 7 = 30,933 rows.
    Built tz-aware with zoneinfo and converted to UTC: the offset changes
    with DST (EDT/EST), so naive hour arithmetic would silently shift
    every winter candidate by an hour. Returns a DataFrame with one row
    per candidate: local_date, local_hour, departure_utc (naive, UTC)."""
    rows = []
    d = ARCHIVE_START
    today = dt.date.today()
    while d <= today:
        for h in DEPARTURE_LOCAL_HOURS:
            local = dt.datetime(d.year, d.month, d.day, h, tzinfo=NY_TZ)
            departure_utc = local.astimezone(dt.timezone.utc).replace(tzinfo=None)
            rows.append((d, h, np.datetime64(departure_utc)))
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows, columns=["local_date", "local_hour", "departure_utc"])


def local_to_departure_utc(local_date, local_hour):
    """One local_date/local_hour (America/New_York) -> naive UTC datetime,
    the same tz-aware conversion candidate_departures uses per row. Lets
    report.py build a single candidate the same way search builds ~31,000."""
    local = dt.datetime(local_date.year, local_date.month, local_date.day,
                         local_hour, tzinfo=NY_TZ)
    return local.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _format_hmm(seconds):
    """Seconds -> 'h:mm' string, for display only."""
    total_min = int(round(seconds / 60.0))
    h, m = divmod(total_min, 60)
    return f"{h}:{m:02d}"


def march_legs(ss_legs, ss_idx, data, dep_i8, departure_to_accel_s=DEPARTURE_TO_ACCEL_S):
    """The whole per-leg march, vectorised across candidates -- dep_i8 (int64
    ns UTC timestamps) may be length 1 (report.py, one candidate) or ~31,000
    (run_search), every operation inside the per-leg loop treats it the
    same way. ss_legs/ss_idx are the supersonic-segment Leg objects and
    their positions in data's leg axis (from route.supersonic_segment).

    At each leg, u/v/t are interpolated from the 4 raw ERA5 pressure levels
    onto the 16-level TARGET_FL grid (linear in log(pressure) -- the
    standard treatment for u/v/t between mandatory levels) before
    limits.best_level picks a level. data["level"] (hPa) runs *decreasing*
    along the level axis (it's stored high-FL-last -> low pressure last),
    so log(pressure) decreases along that axis; np.interp needs increasing
    x, hence the reversal to ascending-pressure order below. FL450-FL600
    sits inside the ERA5 mandatory-level span (150-70 hPa = FL446-FL605),
    so every target is bracketed -- asserted, not extrapolated.

    Returns (legs, weight_per_leg): legs is a dict of (n_cand, n_legs)
    arrays -- chosen_fl, best_idx, mach, tas_kt, gs_kt, wind_kt, temp_c,
    isa_dev_k, leg_time_s, elapsed_s (cumulative seconds since departure, at
    the end of each leg), and binding (str array: which of
    Mmo/CAS/total_temp/ceiling constrains the chosen level -- "ceiling"
    when the chosen level is the highest one ceiling_ft allows, i.e.
    altitude-capped rather than speed-capped) -- plus accumulated_s
    (n_cand,), the final total elapsed time. weight_per_leg is (n_legs,)."""
    n_legs = len(ss_legs)
    n_cand = len(dep_i8)

    times_i8 = data["time"].astype("datetime64[ns]").astype("int64")
    isa_t_per_level, _ = isa(TARGET_FL * 100.0 * 0.3048)  # ISA temp on the 16-level grid, fixed per level

    target_p = fl_to_pressure(TARGET_FL)          # (16,) Pa, decreasing as FL increases
    target_log_p = np.log(target_p)
    src_p = data["level"].astype(float)[::-1] * 100.0  # hPa -> Pa, reversed to ascending
    src_log_p = np.log(src_p)
    assert np.all(np.diff(src_log_p) > 0), \
        "ERA5 levels must be strictly monotonic in pressure"
    assert target_log_p.min() >= src_log_p.min() and target_log_p.max() <= src_log_p.max(), \
        "TARGET_FL must be bracketed by the ERA5 levels -- no extrapolation"

    v_idx0 = np.searchsorted(src_log_p, target_log_p, side="right") - 1
    v_idx0 = np.clip(v_idx0, 0, len(src_log_p) - 2)
    v_idx1 = v_idx0 + 1
    v_frac = (target_log_p - src_log_p[v_idx0]) / (src_log_p[v_idx1] - src_log_p[v_idx0])

    def _interp_vertical(values):
        """(n_cand, n_raw_level) at data["level"] order -> (n_cand, 16) on
        TARGET_FL, linear in log(pressure)."""
        values_asc = values[:, ::-1]  # match src_p's ascending-pressure order
        lo = values_asc[:, v_idx0]
        hi = values_asc[:, v_idx1]
        return lo + v_frac[None, :] * (hi - lo)

    u_ss = data["u"][:, :, ss_idx]      # (n_time, n_level, n_legs)
    v_ss = data["v"][:, :, ss_idx]
    temp_ss = data["t"][:, :, ss_idx]

    # Weight at each leg's midpoint, linear in cum_nm from the accel point
    # (start of the first supersonic leg) to the decel point (end of the
    # last).
    accel_cum_nm = ss_legs[0].cum_nm - ss_legs[0].dist_nm
    decel_cum_nm = ss_legs[-1].cum_nm
    mid_cum_nm = np.array([leg.cum_nm - leg.dist_nm / 2.0 for leg in ss_legs])
    weight_per_leg = np.interp(mid_cum_nm, [accel_cum_nm, decel_cum_nm],
                                [WEIGHT_ACCEL_T, WEIGHT_DECEL_T])

    accumulated_s = np.full(n_cand, float(departure_to_accel_s))
    chosen_fl = np.empty((n_cand, n_legs))
    best_idx_out = np.empty((n_cand, n_legs), dtype=int)
    mach = np.empty((n_cand, n_legs))
    tas_kt = np.empty((n_cand, n_legs))
    gs_kt = np.empty((n_cand, n_legs))
    wind_kt = np.empty((n_cand, n_legs))
    temp_c = np.empty((n_cand, n_legs))
    isa_dev_k = np.empty((n_cand, n_legs))
    leg_time_s = np.empty((n_cand, n_legs))
    elapsed_s = np.empty((n_cand, n_legs))
    binding = np.empty((n_cand, n_legs), dtype=object)

    for i, leg in enumerate(ss_legs):
        leg_time_i8 = dep_i8 + (accumulated_s * 1e9).astype("int64")

        idx1 = np.searchsorted(times_i8, leg_time_i8, side="right") - 1
        idx1 = np.clip(idx1, 0, len(times_i8) - 2)
        idx2 = idx1 + 1
        t0, t1 = times_i8[idx1], times_i8[idx2]
        frac = np.clip((leg_time_i8 - t0) / (t1 - t0), 0.0, 1.0)[:, None]

        u_leg_raw = u_ss[idx1, :, i] + frac * (u_ss[idx2, :, i] - u_ss[idx1, :, i])
        v_leg_raw = v_ss[idx1, :, i] + frac * (v_ss[idx2, :, i] - v_ss[idx1, :, i])
        temp_leg_raw = temp_ss[idx1, :, i] + frac * (temp_ss[idx2, :, i] - temp_ss[idx1, :, i])

        # (n_cand, 4) raw ERA5 levels -> (n_cand, 16) on TARGET_FL.
        u_leg = _interp_vertical(u_leg_raw)
        v_leg = _interp_vertical(v_leg_raw)
        temp_leg = _interp_vertical(temp_leg_raw)

        best_fl, best_gs_ms, best_idx, _ = limits.best_level(
            TARGET_FL[None, :], temp_leg, u_leg, v_leg, leg.track_deg, weight_per_leg[i]
        )

        along_per_level = (u_leg * np.sin(np.radians(leg.track_deg))
                            + v_leg * np.cos(np.radians(leg.track_deg)))
        isa_dev_per_level = temp_leg - isa_t_per_level[None, :]

        wind_at_best = np.take_along_axis(along_per_level, best_idx[:, None], axis=-1).squeeze(-1)
        isa_dev_at_best = np.take_along_axis(isa_dev_per_level, best_idx[:, None], axis=-1).squeeze(-1)
        temp_k_at_best = np.take_along_axis(temp_leg, best_idx[:, None], axis=-1).squeeze(-1)

        # Achieved Mach/TAS at the chosen level, and which of Mmo/CAS/
        # total-temp constrains it there. "ceiling" overrides that when the
        # chosen level is the highest one ceiling_ft(weight) allows for this
        # leg -- altitude-capped, not speed-capped (ceiling never appears in
        # mach_components/max_mach, which only cap speed at a given level).
        mach_at_best = limits.max_mach(best_fl, temp_k_at_best, weight_per_leg[i])
        tas_ms_at_best = mach_at_best * speed_of_sound(temp_k_at_best)
        mach_limit_label = limits.binding_mach_limit(best_fl, temp_k_at_best, weight_per_leg[i])

        ceiling = limits.ceiling_ft(weight_per_leg[i])
        above_ceiling_grid = TARGET_FL * 100.0 > ceiling
        top_available_idx = np.max(np.flatnonzero(~above_ceiling_grid))
        ceiling_bound = best_idx == top_available_idx
        binding_at_best = np.where(ceiling_bound, "ceiling", mach_limit_label)

        leg_s = (leg.dist_nm * NM_TO_M) / best_gs_ms
        accumulated_s = accumulated_s + leg_s

        chosen_fl[:, i] = best_fl
        best_idx_out[:, i] = best_idx
        mach[:, i] = mach_at_best
        tas_kt[:, i] = tas_ms_at_best / KT_TO_MS
        gs_kt[:, i] = best_gs_ms / KT_TO_MS
        wind_kt[:, i] = wind_at_best / KT_TO_MS
        temp_c[:, i] = temp_k_at_best - 273.15
        isa_dev_k[:, i] = isa_dev_at_best
        leg_time_s[:, i] = leg_s
        elapsed_s[:, i] = accumulated_s
        binding[:, i] = binding_at_best

    legs_out = dict(
        chosen_fl=chosen_fl, best_idx=best_idx_out, mach=mach, tas_kt=tas_kt,
        gs_kt=gs_kt, wind_kt=wind_kt, temp_c=temp_c, isa_dev_k=isa_dev_k,
        leg_time_s=leg_time_s, elapsed_s=elapsed_s, binding=binding,
        accumulated_s=accumulated_s,
    )
    return legs_out, weight_per_leg


def run_search(pln_path, npz_path, accel_id="LINND", decel_id="BARIX",
                top=50, out_path="results.csv",
                departure_to_accel_s=DEPARTURE_TO_ACCEL_S):
    """Builds legs from pln_path (same max_leg_nm default as
    `route`/reduce_to_legs, so the leg axis lines up with npz_path's), takes
    the supersonic ones, then calls march_legs across every candidate
    departure and collapses the per-leg arrays to means. Prints the top 10
    rows and four sanity checks (computed on the same filtered/sorted rows
    as the output table), writes the top `top` rows to out_path, and
    returns the full ranked DataFrame (no NaN rows)."""
    plan = parse_pln(pln_path)
    legs = build_legs(plan["waypoints"])
    mask = supersonic_segment(legs, accel_id=accel_id, decel_id=decel_id)
    ss_idx = np.flatnonzero(mask)
    ss_legs = [legs[i] for i in ss_idx]

    data = load_legs_npz(npz_path)
    candidates = candidate_departures()
    n_cand = len(candidates)
    dep_i8 = candidates["departure_utc"].values.astype("datetime64[ns]").astype("int64")

    legs_out, _weight_per_leg = march_legs(ss_legs, ss_idx, data, dep_i8, departure_to_accel_s)
    chosen_fl = legs_out["chosen_fl"]
    wind_kt = legs_out["wind_kt"]
    isa_dev_k = legs_out["isa_dev_k"]
    gs_kt = legs_out["gs_kt"]

    candidates["supersonic_time_s"] = legs_out["accumulated_s"]
    candidates["mean_fl"] = chosen_fl.mean(axis=1)
    candidates["mean_wind_kt"] = wind_kt.mean(axis=1)
    candidates["mean_isa_dev_k"] = isa_dev_k.mean(axis=1)

    # Same filter the output table gets, captured here so the sanity checks
    # below are computed on exactly those rows, not the unfiltered arrays.
    valid_mask = candidates[["supersonic_time_s", "mean_fl"]].notna().all(axis=1).to_numpy()
    chosen_fl_valid = chosen_fl[valid_mask]
    gs_kt_valid = gs_kt[valid_mask]

    candidates = candidates.dropna(subset=["supersonic_time_s", "mean_fl"])
    candidates = candidates.sort_values("supersonic_time_s", ascending=True).reset_index(drop=True)

    display = pd.DataFrame({
        "date": candidates["local_date"],
        "local_departure": candidates["local_hour"].map(lambda h: f"{h:02d}:00"),
        "supersonic_time": candidates["supersonic_time_s"].map(_format_hmm),
        "mean_fl": candidates["mean_fl"].round(0).astype(int),
        "mean_wind_kt": candidates["mean_wind_kt"].round(1),
        "mean_isa_dev_k": candidates["mean_isa_dev_k"].round(1),
    })

    print(display.head(10).to_string(index=False))

    ss_total_nm = sum(leg.dist_nm for leg in ss_legs)
    best_row = candidates.iloc[0]
    worst_row = candidates.iloc[-1]
    spread_min = (worst_row["supersonic_time_s"] - best_row["supersonic_time_s"]) / 60.0
    top_fl = TARGET_FL.max()
    frac_at_top = float(np.mean(chosen_fl_valid == top_fl))

    is_winter = best_row["local_date"].month in (11, 12, 1, 2)
    print("\nSanity checks:")
    print(f"1. Best day: {best_row['local_date']} "
          f"({'winter' if is_winter else 'NOT WINTER -- check wind sign'})")
    print(f"2. Spread best-worst: {spread_min:.1f} min over {ss_total_nm:.0f} nm "
          f"(expect ~25-40 min)")
    print(f"3. Mean chosen FL overall: {chosen_fl_valid.mean():.0f} "
          f"(top available FL {top_fl:.0f}); "
          f"at top level {frac_at_top * 100:.0f}% of leg-instances")
    print(f"4. Chosen ground speed range: {gs_kt_valid.min():.0f}-{gs_kt_valid.max():.0f} kt "
          f"(expect 800-1400 kt); NaN rows dropped: {n_cand - len(candidates)}")

    out_path = Path(out_path)
    display.head(top).to_csv(out_path, index=False)
    print(f"\nWrote top {min(top, len(display))} rows to {out_path}")

    return candidates
