"""Phase 5: verify a shortlisted day against Active Sky's own atmosphere.

Workflow: the user loads a historical date/time in Active Sky by hand (a
static snapshot of ActiveSky's global weather model for that moment -- the
API takes an explicit lat/lon/altitude, not "wherever the aircraft is", so
one load covers every point queried below), then runs `concopt verify`. It
takes --points evenly spaced supersonic legs (including the first and last),
queries Active Sky live at each one's midpoint for the same FL450-FL600
TARGET_FL grid the search uses, and pulls the ERA5 values search.march_legs
would have used at that same point and clock time (reusing march_legs --
never reimplementing its vertical/time interpolation). Both sources pick
their best level with the SAME weight (the ERA5 march's weight_per_leg) and
cruise_mach, so a level disagreement reflects a genuine wind/temp difference
between the sources, not a weight mismatch.

The recomputed "AS total time" extends the n_points ground speeds across the
whole segment by nearest-point assignment (see _group_gs_ms) -- an eyeball
approximation, not a full AS march (a full march would need Active Sky
queried at every sub-leg, defeating the point of sampling only --points of
them).
"""
import numpy as np
import pandas as pd

from concopt import limits
from concopt.asky import get_atmosphere_np
from concopt.atmos import KT_TO_MS
from concopt.era5 import load_legs_npz
from concopt.route import build_legs, parse_pln, supersonic_segment
from concopt.search import (DEPARTURE_TO_ACCEL_S, NM_TO_M, TARGET_FL,
                             _format_hmm, local_to_departure_utc, march_legs)


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
    temp_k = temp_c + 273.15
    return temp_k, u_ms, v_ms


def run_verify(pln_path, npz_path, local_date, local_hour, accel_id="LINND",
               decel_id="BARIX", n_points=6, host="localhost", port=19285,
               departure_to_accel_s=DEPARTURE_TO_ACCEL_S,
               cruise_mach=limits.CRUISE_MACH):
    """Compare Active Sky's live atmosphere against the ERA5 values the
    search used, at n_points evenly spaced supersonic legs for local_date/
    local_hour (America/New_York). Prints the per-point comparison and the
    recomputed segment time under each source, and returns a DataFrame of
    the per-point rows."""
    plan = parse_pln(pln_path)
    legs = build_legs(plan["waypoints"])
    mask = supersonic_segment(legs, accel_id=accel_id, decel_id=decel_id)
    ss_idx = np.flatnonzero(mask)
    ss_legs = [legs[i] for i in ss_idx]
    n_legs = len(ss_legs)

    data = load_legs_npz(npz_path)
    departure_utc = local_to_departure_utc(local_date, local_hour)
    departure_utc_ts = pd.Timestamp(departure_utc)
    dep_i8 = np.array([departure_utc_ts.value], dtype="int64")

    legs_out, weight_per_leg = march_legs(ss_legs, ss_idx, data, dep_i8,
                                            departure_to_accel_s, cruise_mach)
    era5 = {k: v[0] for k, v in legs_out.items()
            if k not in ("accumulated_s", "weight_at_barix")}
    era5_total_s = float(legs_out["accumulated_s"][0]) - departure_to_accel_s
    weight_per_leg = weight_per_leg[0]  # (n_legs,), the ERA5 march's weight state

    point_idx = _select_points(n_legs, n_points)

    rows = []
    print(f"{len(point_idx)} verification points (of {n_legs} supersonic sub-legs), "
          f"{local_date} {local_hour:02d}:00 local:\n")
    header = (f"{'leg':>4} {'lat':>8} {'lon':>9} {'AS FL':>6} {'ERA5 FL':>8}  "
              f"{'AS wind':>8} {'ERA5 wind':>10} {'dWind':>7}  "
              f"{'AS temp':>8} {'ERA5 temp':>10} {'dTemp':>7}")
    print(header)

    for i in point_idx:
        leg = ss_legs[i]
        try:
            temp_k_as, u_ms_as, v_ms_as = _as_atmosphere(leg.lat_mid, leg.lon_mid, host, port)
        except RuntimeError as e:
            raise RuntimeError(
                f"Active Sky query failed for leg {i} ({leg.lat_mid:.3f}, {leg.lon_mid:.3f}): {e}"
            ) from e

        weight_t = np.array([float(weight_per_leg[i])])
        best_fl_as, best_gs_ms_as, best_idx_as, _ = limits.best_level(
            TARGET_FL[None, :], temp_k_as[None, :], u_ms_as[None, :], v_ms_as[None, :],
            leg.track_deg, weight_t, cruise_mach
        )
        best_fl_as = float(best_fl_as[0])
        best_gs_ms_as = float(best_gs_ms_as[0])
        best_idx_as = int(best_idx_as[0])

        track_rad = np.radians(leg.track_deg)
        along_ms_as = u_ms_as * np.sin(track_rad) + v_ms_as * np.cos(track_rad)
        wind_kt_as = float(along_ms_as[best_idx_as] / KT_TO_MS)
        temp_c_as = float(temp_k_as[best_idx_as] - 273.15)

        era5_fl = float(era5["chosen_fl"][i])
        era5_wind_kt = float(era5["wind_kt"][i])
        era5_temp_c = float(era5["temp_c"][i])

        rows.append(dict(
            leg_idx=int(i), lat=leg.lat_mid, lon=leg.lon_mid,
            as_fl=best_fl_as, era5_fl=era5_fl,
            as_wind_kt=wind_kt_as, era5_wind_kt=era5_wind_kt,
            as_temp_c=temp_c_as, era5_temp_c=era5_temp_c,
            gs_ms_as=best_gs_ms_as,
        ))

        level_flag = "  <- LEVEL MISMATCH" if abs(best_fl_as - era5_fl) > 1e-6 else ""
        print(f"{i:>4} {leg.lat_mid:>8.3f} {leg.lon_mid:>9.3f} "
              f"FL{best_fl_as:<4.0f} FL{era5_fl:<6.0f}  "
              f"{wind_kt_as:>7.1f} {era5_wind_kt:>9.1f} {wind_kt_as - era5_wind_kt:>+7.1f}  "
              f"{temp_c_as:>7.1f} {era5_temp_c:>9.1f} {temp_c_as - era5_temp_c:>+7.1f}"
              f"{level_flag}")

    result = pd.DataFrame(rows)

    wind_delta = (result["as_wind_kt"] - result["era5_wind_kt"]).to_numpy()
    temp_delta = (result["as_temp_c"] - result["era5_temp_c"]).to_numpy()
    n_mismatch = int((result["as_fl"] != result["era5_fl"]).sum())
    print(f"\nWind delta (AS-ERA5): mean {wind_delta.mean():+.1f} kt, "
          f"std {wind_delta.std():.1f} kt over {len(result)} points")
    print(f"Temp delta (AS-ERA5): mean {temp_delta.mean():+.1f} C, "
          f"std {temp_delta.std():.1f} C over {len(result)} points")
    print(f"Level mismatches: {n_mismatch}/{len(result)} points "
          f"(a std well below the mean magnitude points at a systematic bias; "
          f"a std comparable to or above it points at scatter)")

    gs_ms_by_leg = _group_gs_ms(point_idx, result["gs_ms_as"].to_numpy(), n_legs)
    as_total_s = float(sum(leg.dist_nm * NM_TO_M / gs
                           for leg, gs in zip(ss_legs, gs_ms_by_leg)))
    diff_min = (as_total_s - era5_total_s) / 60.0

    print(f"\nSupersonic segment time: ERA5 {_format_hmm(era5_total_s)}  "
          f"AS (recomputed) {_format_hmm(as_total_s)}  diff {diff_min:+.1f} min "
          f"({'OK' if abs(diff_min) <= 5.0 else 'CHECK -- large divergence'})")

    return result
