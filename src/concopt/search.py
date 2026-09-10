"""Day/time scan for the JFK-LHR supersonic segment. For every candidate
departure, march across the supersonic legs picking the best-wind flight
level at each one (limits.best_level), and rank candidates by total
supersonic-segment time. Reads only the .npz written by era5.reduce_to_legs;
never touches the netCDF.
"""
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from concopt import limits
from concopt.atmos import KT_TO_MS, isa, pressure_to_fl
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


def _format_hmm(seconds):
    """Seconds -> 'h:mm' string, for display only."""
    total_min = int(round(seconds / 60.0))
    h, m = divmod(total_min, 60)
    return f"{h}:{m:02d}"


def run_search(pln_path, npz_path, accel_id="LINND", decel_id="BARIX",
                top=50, out_path="results.csv"):
    """The whole algorithm. Builds legs from pln_path (same max_leg_nm
    default as `route`/reduce_to_legs, so the leg axis lines up with
    npz_path's), takes the supersonic ones, then marches candidate
    departures across them one leg at a time -- a Python loop over ~32
    legs, every operation inside it vectorised across all ~31,000
    candidates at once. Prints the top 10 rows and four sanity checks,
    writes the top `top` rows to out_path, and returns the full ranked
    DataFrame (no NaN rows)."""
    plan = parse_pln(pln_path)
    legs = build_legs(plan["waypoints"])
    mask = supersonic_segment(legs, accel_id=accel_id, decel_id=decel_id)
    ss_idx = np.flatnonzero(mask)
    ss_legs = [legs[i] for i in ss_idx]
    n_legs = len(ss_legs)

    data = load_legs_npz(npz_path)
    times_i8 = data["time"].astype("datetime64[ns]").astype("int64")
    fl_levels = pressure_to_fl(data["level"].astype(float) * 100.0)  # hPa -> Pa -> FL, (n_level,)
    isa_t_per_level, _ = isa(fl_levels * 100.0 * 0.3048)  # ISA temp, fixed per level

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

    candidates = candidate_departures()
    n_cand = len(candidates)
    dep_i8 = candidates["departure_utc"].values.astype("datetime64[ns]").astype("int64")

    accumulated_s = np.zeros(n_cand)
    chosen_fl = np.empty((n_cand, n_legs))
    wind_kt = np.empty((n_cand, n_legs))
    isa_dev_k = np.empty((n_cand, n_legs))
    gs_kt = np.empty((n_cand, n_legs))

    for i, leg in enumerate(ss_legs):
        leg_time_i8 = dep_i8 + (accumulated_s * 1e9).astype("int64")

        idx1 = np.searchsorted(times_i8, leg_time_i8, side="right") - 1
        idx1 = np.clip(idx1, 0, len(times_i8) - 2)
        idx2 = idx1 + 1
        t0, t1 = times_i8[idx1], times_i8[idx2]
        frac = np.clip((leg_time_i8 - t0) / (t1 - t0), 0.0, 1.0)[:, None]

        u_leg = u_ss[idx1, :, i] + frac * (u_ss[idx2, :, i] - u_ss[idx1, :, i])
        v_leg = v_ss[idx1, :, i] + frac * (v_ss[idx2, :, i] - v_ss[idx1, :, i])
        temp_leg = temp_ss[idx1, :, i] + frac * (temp_ss[idx2, :, i] - temp_ss[idx1, :, i])

        best_fl, best_gs_ms, gs_per_level = limits.best_level(
            fl_levels[None, :], temp_leg, u_leg, v_leg, leg.track_deg, weight_per_leg[i]
        )
        best_idx = np.argmax(gs_per_level == best_gs_ms[:, None], axis=-1)

        along_per_level = (u_leg * np.sin(np.radians(leg.track_deg))
                            + v_leg * np.cos(np.radians(leg.track_deg)))
        isa_dev_per_level = temp_leg - isa_t_per_level[None, :]

        wind_at_best = np.take_along_axis(along_per_level, best_idx[:, None], axis=-1).squeeze(-1)
        isa_dev_at_best = np.take_along_axis(isa_dev_per_level, best_idx[:, None], axis=-1).squeeze(-1)

        chosen_fl[:, i] = best_fl
        wind_kt[:, i] = wind_at_best / KT_TO_MS
        isa_dev_k[:, i] = isa_dev_at_best
        gs_kt[:, i] = best_gs_ms / KT_TO_MS

        accumulated_s += (leg.dist_nm * NM_TO_M) / best_gs_ms

    candidates["supersonic_time_s"] = accumulated_s
    candidates["mean_fl"] = chosen_fl.mean(axis=1)
    candidates["mean_wind_kt"] = wind_kt.mean(axis=1)
    candidates["mean_isa_dev_k"] = isa_dev_k.mean(axis=1)
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
    top_fl = fl_levels.max()
    frac_at_top = float(np.mean(chosen_fl == top_fl))

    is_winter = best_row["local_date"].month in (11, 12, 1, 2)
    print("\nSanity checks:")
    print(f"1. Best day: {best_row['local_date']} "
          f"({'winter' if is_winter else 'NOT WINTER -- check wind sign'})")
    print(f"2. Spread best-worst: {spread_min:.1f} min over {ss_total_nm:.0f} nm "
          f"(expect ~25-40 min)")
    print(f"3. Mean chosen FL overall: {chosen_fl.mean():.0f} "
          f"(top available FL {top_fl:.0f}); "
          f"at top level {frac_at_top * 100:.0f}% of leg-instances")
    print(f"4. Chosen ground speed range: {gs_kt.min():.0f}-{gs_kt.max():.0f} kt "
          f"(expect 800-1400 kt); NaN rows dropped: {n_cand - len(candidates)}")

    out_path = Path(out_path)
    display.head(top).to_csv(out_path, index=False)
    print(f"\nWrote top {min(top, len(display))} rows to {out_path}")

    return candidates
