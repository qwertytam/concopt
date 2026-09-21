"""Phase 5: verify a shortlisted day against Active Sky's own atmosphere.

Workflow: the user loads a historical date/time in Active Sky by hand (a
static snapshot of ActiveSky's global weather model for that moment -- the
API takes an explicit lat/lon/altitude, not "wherever the aircraft is", so
one load covers every point queried below), then runs `concopt verify`
(--zfw, and --tow if the search used one, should match the day's `concopt
search` run). It marches the
same TOW-based climb search.py uses, then takes --points (default 12)
evenly spaced cruise legs past top of climb (including the first and last;
climb-altitude legs are excluded -- Active Sky's FL450-FL600 grid doesn't
apply there), queries Active Sky live at each one's midpoint for the same
FL450-FL600 TARGET_FL grid the search uses, and pulls the ERA5 values
search.march_legs would have used at that same point and clock time
(reusing march_legs -- never reimplementing its vertical/time
interpolation). Both sources pick their best level with the SAME weight
(the ERA5 march's weight_per_leg) and cruise_mach, so a level disagreement
reflects a genuine wind/temp difference between the sources, not a weight
mismatch.

SNAPSHOT GUARD: every point is queried before anything is printed, and the
whole run's wind+temp is fingerprinted (_atmosphere_fingerprint) and
checked against a small cache under data/ (_guard_snapshot) -- if the
fingerprint exactly matches a previous run for a DIFFERENT date/hour,
Active Sky almost certainly wasn't reloaded (confirmed live, 2026-09: a
run against 2016-01-06 came back bit-identical, at all 6 points then in
use, to the 2016-02-12 run just before it), and run_verify refuses to
proceed rather than silently comparing one date's Active Sky weather
against another date's ERA5. This is the single most valuable check here --
that failure mode is silent, plausible, and produces numbers that look
fine.

The recomputed "AS total time" extends the n_points ground speeds across the
whole segment by nearest-point assignment (see _group_gs_ms) -- an eyeball
approximation, not a full AS march (a full march would need Active Sky
queried at every sub-leg, defeating the point of sampling only --points of
them).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from concopt import fuel, limits
from concopt.asky import get_atmosphere_np
from concopt.atmos import KT_TO_MS
from concopt.era5 import load_legs_npz
from concopt.route import build_legs, climb_cruise_segment, parse_pln
from concopt.params import (ACTIVE_SKY_HOST, ACTIVE_SKY_PORT, C_TO_K,  # noqa: F401
                            NM_TO_M, SNAPSHOT_CACHE_PATH, VERIFY_TOTAL_TOLERANCE_MIN)
from concopt.params import VERIFY_DEFAULT_N_POINTS as DEFAULT_N_POINTS  # noqa: F401
from concopt.search import (TARGET_FL, _format_hmm,
                             local_to_departure_utc, march_legs,
                             resolve_tow_and_arrival)


def _select_points(n_legs, n_points):
    """n_points indices into a length-n_legs leg array, evenly spaced and
    including both ends (np.linspace, rounded); deduplicated and sorted, so
    a --points larger than n_legs just uses every leg."""
    n_points = min(n_points, n_legs)
    idx = np.round(np.linspace(0, n_legs - 1, n_points)).astype(int)
    return np.unique(idx)


def _group_gs_ms(point_idx, gs_ms_at_points, n_legs):
    """(n_legs,) ground speed array assigning every leg to its nearest
    sampled point_idx (ties -> the earlier point) -- extends the n_points
    Active Sky ground speeds across the whole segment for the total-time
    recompute."""
    if len(point_idx) == 1:
        return np.full(n_legs, gs_ms_at_points[0])
    midpoints = (point_idx[:-1] + point_idx[1:]) / 2.0
    group = np.searchsorted(midpoints, np.arange(n_legs))
    return gs_ms_at_points[group]


def _as_atmosphere(lat, lon, host, port):
    """Query Active Sky at (lat, lon) for the FL450-FL600 TARGET_FL grid
    (1000 ft steps) and return (temp_k, u_ms, v_ms), each (len(TARGET_FL),),
    reindexed onto TARGET_FL by altitude so the response order doesn't
    matter. Active Sky's WindDirection is the meteorological FROM bearing
    (true), so u/v (eastward/northward, ERA5 convention) are the standard
    from-bearing inversion, not the along/cross shortcut runways.py uses
    against a fixed runway heading."""
    target_ft = TARGET_FL * 100.0
    alt_ft, wind_dir_deg, wind_speed_kt, _pressure_hpa, temp_c = get_atmosphere_np(
        lat, lon, target_ft, host_addr=host, port=port
    )
    order = np.argsort(alt_ft)
    alt_ft = alt_ft[order]
    if not np.allclose(alt_ft, target_ft, atol=1.0):
        raise RuntimeError(
            f"Active Sky returned altitudes {alt_ft.tolist()} for requested "
            f"{target_ft.tolist()} -- can't align to the TARGET_FL grid"
        )
    wind_dir_deg = wind_dir_deg[order]
    wind_speed_kt = wind_speed_kt[order]
    temp_c = temp_c[order]

    speed_ms = wind_speed_kt * KT_TO_MS
    dir_rad = np.radians(wind_dir_deg)
    u_ms = -speed_ms * np.sin(dir_rad)
    v_ms = -speed_ms * np.cos(dir_rad)
    temp_k = temp_c + C_TO_K
    return temp_k, u_ms, v_ms


def _atmosphere_fingerprint(as_atmospheres):
    """SHA-256 hex digest of every (temp_k, u_ms, v_ms) triple queried this
    run, concatenated in query order. Active Sky's live weather model
    should never return bit-identical wind/temp for two different
    historical loads -- an exact match across runs is the signature of a
    load that didn't happen, not a coincidence (see _guard_snapshot)."""
    arr = np.concatenate([np.concatenate([t, u, v]) for t, u, v in as_atmospheres])
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _guard_snapshot(fingerprint, local_date, local_hour, cache_path):
    """Raise RuntimeError if fingerprint already belongs to a different
    date/hour's entry in cache_path, then record fingerprint under this
    run's own key either way. A repeat of the SAME date/hour is expected
    and not an error (re-verifying without reloading is legitimate); a
    match under a DIFFERENT key means Active Sky handed back a previous
    load's snapshot, not this run's -- confirmed live, 2026-09: a run
    against 2016-01-06 came back bit-identical to the 2016-02-12 run just
    before it, because the historical date hadn't actually been reloaded
    in Active Sky."""
    key = f"{local_date.isoformat()} {local_hour:02d}:00"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    for other_key, other_fingerprint in cache.items():
        if other_fingerprint == fingerprint and other_key != key:
            raise RuntimeError(
                f"Active Sky returned the same atmosphere as the {other_key} "
                "run -- it looks like the historical date wasn't reloaded. "
                "Reload it in Active Sky and retry."
            )

    cache[key] = fingerprint
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


def _mismatch_tas_cost_kt(as_idx, era5_idx, temp_k_as_grid, weight_t, cruise_mach):
    """TAS (kt) lost by flying era5_idx's level instead of as_idx's, both
    evaluated under Active Sky's OWN temperature at each level (so the cost
    reflects the real atmosphere, not a mix of two sources' pictures of
    it) -- 0 when the two sources picked the same level. Isolates the
    performance cost of a level disagreement from the wind/ground-speed
    deltas already reported separately: a mismatch entirely above the Mmo
    knee, where cruise_mach alone binds and TAS is flat with altitude,
    costs ~0 kt; one that crosses into the CAS-limited lower part of the
    envelope (e.g. FL470 vs FL490) costs real speed. A raw mismatch COUNT
    conflates these -- 5 mismatches that are all free reads the same as 5
    that cost 40+ kt each."""
    if as_idx == era5_idx:
        return 0.0
    as_fl = TARGET_FL[as_idx]
    era5_fl = TARGET_FL[era5_idx]
    tas_as_fl = limits.max_tas(as_fl, temp_k_as_grid[as_idx], weight_t, cruise_mach)
    tas_era5_fl = limits.max_tas(era5_fl, temp_k_as_grid[era5_idx], weight_t, cruise_mach)
    return float(abs(tas_as_fl - tas_era5_fl) / KT_TO_MS)


def run_verify(pln_path, npz_path, local_date, local_hour,
               decel_id, n_points=DEFAULT_N_POINTS, host=ACTIVE_SKY_HOST, port=ACTIVE_SKY_PORT,
               tow_t=None, zfw_t=None,
               min_landing_fuel_t=fuel.MIN_LANDING_FUEL_T,
               subsonic_npz_path=None,
               cruise_mach=limits.CRUISE_MACH,
               csv_path=None,
               snapshot_cache_path=SNAPSHOT_CACHE_PATH,
               arrival_upper_npz_path=None):
    """Compare Active Sky's live atmosphere against the ERA5 values the
    search used, at n_points evenly spaced cruise legs (climb-consumed legs
    excluded -- see march_legs' eff_dist_nm) for local_date/local_hour
    (America/New_York).

    zfw_t (tonnes, required) runs the SAME fixed point `concopt search`/
    `concopt report` use (search.resolve_tow_and_arrival) rather than a bare
    TOW guess, so the day is verified at the weight it was actually found
    under -- an Active Sky check flown at the wrong weight undercuts the
    whole comparison. subsonic_npz_path AND arrival_upper_npz_path are both
    required then: the fixed point's arrival fuel needs the same stitched
    post-decel wind source search used (B6), or the two TOWs won't agree.
    tow_t is an optional override (match the --tow the search was run with):
    it skips the fixed point entirely, so no arrival/subsonic data is needed
    at all -- zfw_t is then unused here beyond being required.

    Every point is queried before anything is printed; the whole run is
    then fingerprinted and checked against snapshot_cache_path
    (_guard_snapshot) -- a match against a different date/hour raises
    rather than proceeding to compare against stale Active Sky weather.

    Prints the per-point comparison (each row's wind/temp delta, and the
    TAS cost of a level mismatch rather than just flagging it) and the
    recomputed segment time under each source. With csv_path given, also
    appends one summary row (mean/std wind and temp delta, wind sign
    count, ERA5/AS total minutes) to it, so repeated runs accumulate into
    something rankable. Returns a DataFrame of the per-point rows."""
    plan = parse_pln(pln_path)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs, decel_id=decel_id)
    cc_idx = np.flatnonzero(mask)
    cc_legs = [legs[i] for i in cc_idx]

    data = load_legs_npz(npz_path)
    departure_utc = local_to_departure_utc(local_date, local_hour)
    departure_utc_ts = pd.Timestamp(departure_utc)
    dep_i8 = np.array([departure_utc_ts.value], dtype="int64")

    if zfw_t is None:
        raise ValueError("zfw_t (--zfw) is required; tow_t (--tow) is only an override on top of it")

    if tow_t is None:
        # zfw_t given -- reproduce search's own fixed point exactly (arrival
        # fuel included), so this TOW matches the one that day was found
        # under. A bare march_legs call here would drift from search's TOW
        # by the arrival fuel search now folds into its own fixed point.
        if subsonic_npz_path is None or arrival_upper_npz_path is None:
            raise ValueError(
                "subsonic_npz_path (--subsonic-npz) AND arrival_upper_npz_path "
                "(--arrival-upper-npz) are both required with zfw_t (--zfw), so the "
                "fixed point's arrival fuel matches the search run being verified"
            )
        subsonic_data = load_legs_npz(subsonic_npz_path)
        arrival_upper_data = load_legs_npz(arrival_upper_npz_path)
        arrival_idx = np.flatnonzero(~mask)
        arrival_legs = [legs[i] for i in arrival_idx]
        arrival_nm = legs[-1].cum_nm - cc_legs[-1].cum_nm

        tow_arr, _n_iterations, _fuel_flags, legs_out, weight_per_leg, climb, _arrival_out = (
            resolve_tow_and_arrival(
                cc_legs, cc_idx, arrival_legs, arrival_nm, data, subsonic_data, dep_i8,
                zfw_t=zfw_t, min_landing_fuel_t=min_landing_fuel_t, cruise_mach=cruise_mach,
                arrival_upper_data=arrival_upper_data,
            )
        )
        tow_t = float(tow_arr[0])
    else:
        legs_out, weight_per_leg, climb = march_legs(cc_legs, cc_idx, data, dep_i8,
                                                       tow_t, cruise_mach)

    # Restrict to legs entirely past top of climb -- verify.py compares
    # against Active Sky's FL450-FL600 TARGET_FL grid, which doesn't apply
    # to the climb-altitude portion of the flight.
    climb_ground_nm = float(climb["ground_dist_nm"][0])
    cruise_local_idx = [i for i, leg in enumerate(cc_legs)
                         if leg.cum_nm - leg.dist_nm >= climb_ground_nm - 1e-6]
    ss_legs = [cc_legs[i] for i in cruise_local_idx]
    n_legs = len(ss_legs)

    era5 = {k: v[0][cruise_local_idx] for k, v in legs_out.items()
            if k not in ("accumulated_s", "weight_at_barix")}
    era5_total_s = float(legs_out["accumulated_s"][0]) - float(climb["time_min"][0]) * 60.0
    weight_per_leg = weight_per_leg[0][cruise_local_idx]  # (n_legs,), the ERA5 march's weight state

    point_idx = _select_points(n_legs, n_points)

    # Query every point before printing anything -- the snapshot guard
    # needs the whole run's Active Sky response fingerprinted and checked
    # before any per-point output, so a stale (not-reloaded) load is
    # refused rather than printed as if it were real.
    as_atmospheres = []
    for i in point_idx:
        leg = ss_legs[i]
        try:
            as_atmospheres.append(_as_atmosphere(leg.lat_mid, leg.lon_mid, host, port))
        except RuntimeError as e:
            raise RuntimeError(
                f"Active Sky query failed for leg {i} ({leg.lat_mid:.3f}, {leg.lon_mid:.3f}): {e}"
            ) from e

    fingerprint = _atmosphere_fingerprint(as_atmospheres)
    _guard_snapshot(fingerprint, local_date, local_hour, snapshot_cache_path)

    rows = []
    print(f"{len(point_idx)} verification points (of {n_legs} supersonic sub-legs), "
          f"{local_date} {local_hour:02d}:00 local:\n")
    header = (f"{'leg':>4} {'lat':>8} {'lon':>9} {'AS FL':>6} {'ERA5 FL':>8}  "
              f"{'AS wind':>8} {'ERA5 wind':>10} {'dWind':>7}  "
              f"{'AS temp':>8} {'ERA5 temp':>10} {'dTemp':>7}")
    print(header)

    for i, (temp_k_as, u_ms_as, v_ms_as) in zip(point_idx, as_atmospheres):
        leg = ss_legs[i]
        weight_t = float(weight_per_leg[i])
        weight_arr = np.array([weight_t])
        best_fl_as, best_gs_ms_as, best_idx_as, _ = limits.best_level(
            TARGET_FL[None, :], temp_k_as[None, :], u_ms_as[None, :], v_ms_as[None, :],
            leg.track_deg, weight_arr, cruise_mach
        )
        best_fl_as = float(best_fl_as[0])
        best_gs_ms_as = float(best_gs_ms_as[0])
        best_idx_as = int(best_idx_as[0])

        track_rad = np.radians(leg.track_deg)
        along_ms_as = u_ms_as * np.sin(track_rad) + v_ms_as * np.cos(track_rad)
        wind_kt_as = float(along_ms_as[best_idx_as] / KT_TO_MS)
        temp_c_as = float(temp_k_as[best_idx_as] - C_TO_K)

        era5_fl = float(era5["chosen_fl"][i])
        era5_idx = int(era5["best_idx"][i])
        era5_wind_kt = float(era5["wind_kt"][i])
        era5_temp_c = float(era5["temp_c"][i])

        tas_cost_kt = _mismatch_tas_cost_kt(best_idx_as, era5_idx, temp_k_as, weight_t, cruise_mach)

        rows.append(dict(
            leg_idx=int(i), lat=leg.lat_mid, lon=leg.lon_mid,
            as_fl=best_fl_as, era5_fl=era5_fl,
            as_wind_kt=wind_kt_as, era5_wind_kt=era5_wind_kt,
            as_temp_c=temp_c_as, era5_temp_c=era5_temp_c,
            gs_ms_as=best_gs_ms_as, tas_cost_kt=tas_cost_kt,
        ))

        level_flag = f"  <- MISMATCH ({tas_cost_kt:.0f} kt TAS)" if best_idx_as != era5_idx else ""
        print(f"{i:>4} {leg.lat_mid:>8.3f} {leg.lon_mid:>9.3f} "
              f"FL{best_fl_as:<4.0f} FL{era5_fl:<6.0f}  "
              f"{wind_kt_as:>7.1f} {era5_wind_kt:>9.1f} {wind_kt_as - era5_wind_kt:>+7.1f}  "
              f"{temp_c_as:>7.1f} {era5_temp_c:>9.1f} {temp_c_as - era5_temp_c:>+7.1f}"
              f"{level_flag}")

    result = pd.DataFrame(rows)
    n = len(result)

    wind_delta = (result["as_wind_kt"] - result["era5_wind_kt"]).to_numpy()
    temp_delta = (result["as_temp_c"] - result["era5_temp_c"]).to_numpy()

    # Sign count, not just mean/std -- a consistent sign across points is
    # the signal that separates a fixable bias from unfixable scatter; the
    # mean alone can look small while every point still points the same
    # way (or vice versa).
    neg_n = int((wind_delta < 0).sum())
    pos_n = int((wind_delta > 0).sum())
    dominant_n, dominant_label = (neg_n, "negative") if neg_n >= pos_n else (pos_n, "positive")

    print(f"\nWind delta (AS-ERA5): mean {wind_delta.mean():+.1f} kt, "
          f"std {wind_delta.std():.1f} kt, {dominant_n} of {n} points {dominant_label} "
          f"(std well below the mean magnitude points at a systematic bias; "
          f"std comparable to or above it points at scatter)")
    print(f"Temp delta (AS-ERA5): mean {temp_delta.mean():+.1f} C, "
          f"std {temp_delta.std():.1f} C over {n} points")

    mismatch_mask = (result["as_fl"] != result["era5_fl"]).to_numpy()
    n_mismatch = int(mismatch_mask.sum())
    if n_mismatch:
        costs = result.loc[mismatch_mask, "tas_cost_kt"]
        print(f"Level mismatches: {n_mismatch}/{n} points, TAS cost "
              f"{costs.mean():.0f} kt mean / {costs.max():.0f} kt max "
              f"(a mismatch above the Mmo knee costs ~0 kt -- only one that "
              f"crosses into the CAS-limited part of the envelope costs real speed)")
    else:
        print(f"Level mismatches: 0/{n} points")

    gs_ms_by_leg = _group_gs_ms(point_idx, result["gs_ms_as"].to_numpy(), n_legs)
    as_total_s = float(sum(leg.dist_nm * NM_TO_M / gs
                           for leg, gs in zip(ss_legs, gs_ms_by_leg)))
    diff_min = (as_total_s - era5_total_s) / 60.0

    print(f"\nSupersonic segment time: ERA5 {_format_hmm(era5_total_s)}  "
          f"AS (recomputed) {_format_hmm(as_total_s)}  diff {diff_min:+.1f} min "
          f"({'OK' if abs(diff_min) <= VERIFY_TOTAL_TOLERANCE_MIN else 'CHECK -- large divergence'})")

    if csv_path is not None:
        csv_path = Path(csv_path)
        summary_row = pd.DataFrame([dict(
            date=local_date.isoformat(), hour=local_hour, n_points=n,
            wind_delta_mean_kt=wind_delta.mean(), wind_delta_std_kt=wind_delta.std(),
            wind_sign_dominant=dominant_label, wind_sign_n=dominant_n,
            temp_delta_mean_c=temp_delta.mean(), temp_delta_std_c=temp_delta.std(),
            n_mismatch=n_mismatch,
            tas_cost_mean_kt=(costs.mean() if n_mismatch else 0.0),
            tas_cost_max_kt=(costs.max() if n_mismatch else 0.0),
            era5_total_min=era5_total_s / 60.0, as_total_min=as_total_s / 60.0,
            diff_min=diff_min,
        )])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        print(f"\nAppended one row to {csv_path}")

    return result
