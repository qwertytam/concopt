"""Phase C3: turn a flight -- real or synthetic -- into the two callables
run_inflight's seam needs (state_source, weather_source), so run_inflight
can be exercised without a live SimConnect + Active Sky.

Two ways to get a flight-shaped DataFrame to replay:

  * a real recording: `pd.read_csv(a --record CSV)` -- the C2 schema
    (inflight.RECORD_COLUMNS) already has everything replay_sources needs.
  * build_synthetic_flight: no recording exists yet, so generate one from
    the model itself -- the same march/arrival computation `concopt report`
    runs (search.resolve_tow_and_arrival), sampled and lightly perturbed
    into a flight-shaped profile with the SAME columns a real recording
    has. This does not validate the physics (it came from the physics --
    it cannot), but it drives run_inflight through every phase (ground,
    climb, cruise, decel, descent, approach, touchdown) end to end, which
    nothing else in this project has ever done -- run_inflight's own
    docstring used to say as much.

Both feed the SAME replay_sources, so a real recording and a synthetic one
are interchangeable as far as run_inflight is concerned.
"""
import time

import numpy as np
import pandas as pd

from concopt import arrival, fuel, limits
from concopt.atmos import fl_to_pressure, isa
from concopt.route import build_legs, climb_cruise_segment, destination_point, parse_pln
from concopt.search import TOP_OF_CLIMB_FL, resolve_tow_and_arrival

# On-ground ground speed the take-off roll starts at -- mirrors
# inflight.BRAKE_RELEASE_GS_KT (not imported, to avoid a replay -> inflight
# -> replay import cycle; the two are asserted equal in test_replay.py).
_BRAKE_RELEASE_GS_KT = 40.0

# Profile columns replay_sources reads. A strict subset of
# inflight.RECORD_COLUMNS (every name here IS one of those, so a real
# --record CSV is a valid profile as-is) -- listed here separately rather
# than importing RECORD_COLUMNS, so this module never needs the recorder's
# advisory/prediction columns a profile has no way to know in advance
# (rec_fl, predicted_total_s, ...); replay_sources only reads what it
# actually uses to answer state_source/weather_source.
_STATE_COLUMNS = (
    "lat_deg", "lon_deg", "alt_ft", "mach", "tas_kt", "gs_kt", "weight_t",
    "on_ground", "zulu_s",
)
_OPTIONAL_STATE_COLUMNS = (
    "fuel_kg", "sim_wind_kt", "sim_wind_dir_deg", "sim_temp_c",
    "sim_pressure_hpa", "agl_ft", "vs_fpm", "cas_kt", "track_deg",
)


def _nan_to_none(value):
    """pandas' NaN for a missing optional field -> None, the same
    "unavailable" sentinel inflight._read_state's live optional block uses
    (see inflight._get_optional) -- keeps every None-aware helper
    downstream (inflight._record_row, _wind_along_track_kt, ...) working
    the same for a replayed row as for a live one."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _row_to_state(row):
    """One profile row (a pandas Series -- a DataFrame row or a recording
    row) -> the dict inflight._read_state(aq) returns."""
    state = {col: row[col] for col in _STATE_COLUMNS}
    state["on_ground"] = bool(state["on_ground"])
    for col in _OPTIONAL_STATE_COLUMNS:
        state[col] = _nan_to_none(row[col]) if col in row else None
    return state


def _row_weather(row, alt_ft):
    """One profile row's as_* columns, echoed back for whatever alt_ft was
    queried -- get_atmosphere_np's own (alt_ft, wind_dir_deg, wind_speed_kt,
    pressure_hpa, temp_c) shape. A recording (or a synthetic profile) only
    ever sampled Active Sky AT THE AIRCRAFT's own position (see
    inflight._as_point_atmosphere) -- there is no recorded grid to answer a
    query at a DIFFERENT altitude (the lookahead point, the arrival wind
    profile) with, so this answers every altitude in the request with that
    SAME single point's reading. That is a genuine approximation, not a
    replay of what Active Sky would actually have said at those other
    altitudes/positions -- good enough to exercise the advisor's branches
    (a real number in, a real number out, at every level), not to validate
    its numbers. Missing as_* columns for this row still return NaN, not a
    fabricated calm -- exactly what an unavailable live Active Sky query
    would also leave the caller holding."""
    alt_ft = np.atleast_1d(np.asarray(alt_ft, dtype=float))
    n = len(alt_ft)

    def _col(name):
        value = _nan_to_none(row[name]) if name in row else None
        return np.nan if value is None else float(value)

    return (
        alt_ft,
        np.full(n, _col("as_wind_dir_deg")),
        np.full(n, _col("as_wind_kt")),
        np.full(n, _col("as_pressure_hpa")),
        np.full(n, _col("as_temp_c")),
    )


def replay_sources(profile_df, replay_speed=1.0, clock=time.monotonic):
    """(state_source, weather_source) that replay profile_df -- a real
    recording (pd.read_csv of a --record CSV) or a synthetic one from
    build_synthetic_flight below -- in order.

    Keyed by WALL-CLOCK time since the first call (scaled by replay_speed),
    not by call count: run_inflight calls state_source/weather_source more
    often near the ground (inflight.LOW_ALT_INTERVAL_S) and at whatever
    cadence --interval gives it otherwise, and this answers "what does the
    flight look like at this point in elapsed time" correctly either way,
    rather than one profile row per call regardless of how much or little
    wall-clock time has actually passed. Floor lookup (the last row at or
    before the target time), not interpolation -- a recording's rows ARE
    the discrete samples that actually happened; there is nothing truer to
    invent between them.

    Past the last row (the flight already ended): keeps returning the last
    row rather than raising. run_inflight's own on_ground state machine is
    what ends the loop (touchdown detected, or the caller breaks on
    StopIteration-shaped input some other way) -- this never manufactures
    that decision on its own, so a caller that fails to detect touchdown
    correctly is left spinning on a stuck-on-the-ground reading forever,
    which is a louder, more diagnosable failure than a crash mid-test."""
    profile_df = profile_df.sort_values("elapsed_s").reset_index(drop=True)
    elapsed = profile_df["elapsed_s"].to_numpy(dtype=float)
    n_rows = len(profile_df)
    if n_rows == 0:
        raise ValueError("replay_sources: profile_df has no rows to replay")
    t0 = {}

    def _current_row():
        if "value" not in t0:
            t0["value"] = clock()
        target_elapsed_s = (clock() - t0["value"]) * replay_speed
        idx = int(np.searchsorted(elapsed, target_elapsed_s, side="right") - 1)
        idx = int(np.clip(idx, 0, n_rows - 1))
        return profile_df.iloc[idx]

    def state_source():
        return _row_to_state(_current_row())

    def weather_source(lat, lon, alt_ft):
        return _row_weather(_current_row(), alt_ft)

    return state_source, weather_source


def _lat_lon_at_cum_nm(legs, target_cum_nm):
    """The (lat, lon, track_deg) at along-route distance target_cum_nm --
    build_synthetic_flight's counterpart to route.current_progress_nm (which
    goes the other way, point -> cum_nm). Same technique as
    route.project_along_route: reconstruct a leg's start point from its own
    midpoint/track/dist_nm (Leg keeps only the midpoint), then walk forward
    along its track."""
    target_cum_nm = float(np.clip(target_cum_nm, 0.0, legs[-1].cum_nm))
    for leg in legs:
        leg_start_cum_nm = leg.cum_nm - leg.dist_nm
        if target_cum_nm <= leg.cum_nm or leg is legs[-1]:
            along_nm = np.clip(target_cum_nm - leg_start_cum_nm, 0.0, leg.dist_nm)
            start_lat, start_lon = destination_point(
                leg.lat_mid, leg.lon_mid, leg.track_deg + 180.0, leg.dist_nm / 2.0)
            lat, lon = destination_point(start_lat, start_lon, leg.track_deg, along_nm)
            return float(lat), float(lon), float(leg.track_deg)
    raise AssertionError("unreachable -- target_cum_nm clipped into [0, legs[-1].cum_nm]")


def build_synthetic_flight(pln_path, data, dep_i8, subsonic_data=None, arrival_upper_data=None,
                            tow_t=None, zfw_t=None, min_landing_fuel_t=fuel.MIN_LANDING_FUEL_T,
                            decel_descent_min=None, cruise_mach=limits.CRUISE_MACH,
                            decel_id="BARIX", sample_interval_s=15.0, seed=0,
                            as_wind_bias_kt=3.0):
    """A synthetic flight-shaped DataFrame (same columns a --record CSV has,
    a valid replay_sources profile) for JFK->LHR on pln_path, built from
    the SAME model concopt report runs (search.resolve_tow_and_arrival:
    the TOW-based climb, the per-leg cruise march, the real arrival.arrival()
    decel/level/descent/approach split) rather than hand-authored numbers --
    "flying the model's own predicted profile".

    data/subsonic_data/arrival_upper_data are era5.load_legs_npz-shaped
    dicts (real ones, or synthetic still-air ones -- see
    tests/test_report.py's _still_air_npz for the construction; this
    function is deliberately data-agnostic so a fast in-memory still-air
    fixture works here exactly like a real ERA5 .npz would).

    The climb and the four arrival segments have no sub-leg granularity of
    their own (march_legs only marches the climb+cruise span at sub-leg
    resolution -- see its own docstring), so each contributes ONE control
    point at its start (climb) or its own end (each arrival segment); every
    cruise sub-leg with eff_dist_nm > 0 (the real march, not climb-altitude
    noise -- same filter report.py's own step-climb schedule uses)
    contributes its own. Control points are then linearly interpolated onto
    a sample_interval_s grid and given a small seeded random perturbation
    (mach/gs/lat/lon) so replaying this is not a tautology against the
    control points themselves -- this does not validate the model (it IS
    the model), it only exercises run_inflight's loop against a flight that
    actually reaches every phase.

    as_wind_bias_kt offsets the synthetic Active-Sky-at-aircraft wind
    (as_wind_kt) away from the synthetic sim/actual wind by a constant --
    otherwise the two would always agree exactly and the very columns
    Phase C2's recorder schema exists to compare (Q1: "how wrong is the
    weather chain?") would be a no-op on this fixture."""
    plan = parse_pln(pln_path)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs, decel_id=decel_id)
    cc_idx = np.flatnonzero(mask)
    cc_legs = [legs[i] for i in cc_idx]
    arrival_idx = np.flatnonzero(~mask)
    arrival_legs = [legs[i] for i in arrival_idx]
    arrival_nm = legs[-1].cum_nm - cc_legs[-1].cum_nm

    tow_arr, _n_iter, _flags, legs_out, weight_per_leg, climb, arrival_out = (
        resolve_tow_and_arrival(
            cc_legs, cc_idx, arrival_legs, arrival_nm, data, subsonic_data, dep_i8,
            tow_t=tow_t, zfw_t=zfw_t, min_landing_fuel_t=min_landing_fuel_t,
            decel_descent_min=decel_descent_min, cruise_mach=cruise_mach,
            arrival_upper_data=arrival_upper_data,
        )
    )
    tow_t_val = float(tow_arr[0])
    leg = {k: v[0] for k, v in legs_out.items() if k not in ("accumulated_s", "weight_at_barix")}
    weight_per_leg = weight_per_leg[0]
    total_elapsed_s = float(legs_out["accumulated_s"][0])
    weight_at_barix_t = float(legs_out["weight_at_barix"][0])
    climb_time_s = float(climb["time_min"][0]) * 60.0
    climb_ground_nm = float(climb["ground_dist_nm"][0])
    climb_mass_t = float(climb["mass_t"][0])
    decel_cum_nm = float(cc_legs[-1].cum_nm)
    touchdown_cum_nm = float(legs[-1].cum_nm)

    # --- control points, brake release -> touchdown -------------------------
    points = [
        dict(elapsed_s=0.0, cum_nm=0.0, alt_ft=0.0, mach=0.0, tas_kt=0.0, gs_kt=0.0,
             weight_t=tow_t_val, on_ground=True),
        dict(elapsed_s=20.0, cum_nm=0.05, alt_ft=0.0, mach=0.02, tas_kt=15.0, gs_kt=15.0,
             weight_t=tow_t_val, on_ground=True),
        # Crosses _BRAKE_RELEASE_GS_KT (40 kt) between here and the point above.
        dict(elapsed_s=40.0, cum_nm=0.2, alt_ft=0.0, mach=0.09, tas_kt=60.0, gs_kt=60.0,
             weight_t=tow_t_val, on_ground=True),
    ]
    liftoff_elapsed_s = 55.0
    points.append(dict(elapsed_s=liftoff_elapsed_s, cum_nm=0.4, alt_ft=200.0, mach=0.18,
                        tas_kt=110.0, gs_kt=110.0, weight_t=tow_t_val, on_ground=False))
    # Top of climb -- march_legs' own accumulated_s starts counting from
    # here (climb["time_min"]*60), so this point's elapsed_s lines up
    # exactly with leg["elapsed_s"]'s own convention, no offset needed.
    points.append(dict(elapsed_s=climb_time_s, cum_nm=climb_ground_nm,
                        alt_ft=TOP_OF_CLIMB_FL * 100.0, mach=0.95, tas_kt=550.0, gs_kt=550.0,
                        weight_t=climb_mass_t, on_ground=False))

    # Real cruise legs only (eff_dist_nm > 0) -- a leg wholly inside the
    # climb has climb-altitude-noise chosen_fl/mach/etc (best_level ran on
    # it regardless of whether it was actually flown at cruise -- see
    # march_legs' own docstring), same filter report.py's step-climb
    # schedule and mean-FL summary both use.
    n_legs = len(cc_legs)
    for i in np.flatnonzero(leg["eff_dist_nm"] > 0.0):
        weight_after_t = (float(weight_per_leg[i + 1]) if i + 1 < n_legs
                           else weight_at_barix_t)
        points.append(dict(
            elapsed_s=float(leg["elapsed_s"][i]), cum_nm=float(cc_legs[i].cum_nm),
            alt_ft=float(leg["chosen_fl"][i]) * 100.0, mach=float(leg["mach"][i]),
            tas_kt=float(leg["tas_kt"][i]), gs_kt=float(leg["gs_kt"][i]),
            weight_t=weight_after_t, on_ground=False,
        ))

    # Arrival: decel -> level -> descent -> approach, each segment's own
    # (nm, time, fuel) from arrival.arrival()'s real per-day breakdown
    # (search.resolve_tow_and_arrival's own arrival_out) -- one control
    # point at each segment's END (arrival.py has no sub-segment
    # granularity to sample further than this).
    a = {k: float(v[0]) for k, v in arrival_out.items()
         if k not in ("by_schedule", "flags") and np.ndim(v) == 1}
    cursor_elapsed_s, cursor_cum_nm, cursor_weight_t = (
        total_elapsed_s, decel_cum_nm, weight_at_barix_t)

    cursor_elapsed_s += a["decel_time_min"] * 60.0
    cursor_cum_nm += a["decel_nm"]
    cursor_weight_t -= a["decel_fuel_t"]
    points.append(dict(elapsed_s=cursor_elapsed_s, cum_nm=cursor_cum_nm, alt_ft=a["level_fl"] * 100.0,
                        mach=1.0, tas_kt=float(a["schedule_kt"]),
                        gs_kt=float(a["schedule_kt"]) + a["level_wind_kt"],
                        weight_t=cursor_weight_t, on_ground=False))

    cursor_elapsed_s += a["level_time_min"] * 60.0
    cursor_cum_nm += a["level_nm"]
    cursor_weight_t -= a["level_fuel_t"]
    points.append(dict(elapsed_s=cursor_elapsed_s, cum_nm=cursor_cum_nm, alt_ft=a["level_fl"] * 100.0,
                        mach=arrival.LEVEL_MACH, tas_kt=float(a["schedule_kt"]) * 0.95,
                        gs_kt=float(a["schedule_kt"]) * 0.95 + a["level_wind_kt"],
                        weight_t=cursor_weight_t, on_ground=False))

    cursor_elapsed_s += a["descent_time_min"] * 60.0
    cursor_cum_nm += a["descent_nm"]
    cursor_weight_t -= a["descent_fuel_t"]
    points.append(dict(elapsed_s=cursor_elapsed_s, cum_nm=cursor_cum_nm, alt_ft=1500.0,
                        mach=0.3, tas_kt=220.0, gs_kt=220.0,
                        weight_t=cursor_weight_t, on_ground=False))

    touchdown_elapsed_s = cursor_elapsed_s + arrival.APPROACH_MIN * 60.0
    cursor_weight_t -= arrival.APPROACH_FUEL_T
    points.append(dict(elapsed_s=touchdown_elapsed_s, cum_nm=touchdown_cum_nm, alt_ft=0.0,
                        mach=0.15, tas_kt=140.0, gs_kt=140.0,
                        weight_t=cursor_weight_t, on_ground=True))

    # --- densify onto a regular sample_interval_s grid -----------------------
    elapsed_pts = np.array([p["elapsed_s"] for p in points])
    assert np.all(np.diff(elapsed_pts) >= 0), "control points must be time-ordered"
    assert np.all(np.diff([p["cum_nm"] for p in points]) >= 0), \
        "control points must be route-position-ordered"

    grid = np.arange(0.0, elapsed_pts[-1], sample_interval_s)
    grid = np.append(grid, elapsed_pts[-1])  # exact touchdown sample, not just "close to"
    n = len(grid)

    def _interp(key):
        return np.interp(grid, elapsed_pts, [p[key] for p in points])

    cum_nm = _interp("cum_nm")
    alt_ft = _interp("alt_ft")
    mach = _interp("mach")
    tas_kt = _interp("tas_kt")
    gs_kt = _interp("gs_kt")
    weight_t = _interp("weight_t")
    on_ground = (grid < liftoff_elapsed_s) | (grid >= touchdown_elapsed_s)

    rng = np.random.default_rng(seed)
    # "a little noise so it is not a tautology" -- small enough that mach/gs
    # stay physically sane (a 0.3% mach jitter, a couple of kt of gs jitter)
    # and the position jitter (~0.1 nm) stays well inside build_legs'
    # nearest-leg search tolerance, but large enough that replaying this
    # profile is not just reading the control points back verbatim.
    mach = np.clip(mach * (1.0 + rng.normal(0.0, 0.003, n)), 0.0, None)
    gs_kt = np.clip(gs_kt + rng.normal(0.0, 2.0, n), 0.0, None)
    cum_nm_noisy = np.clip(cum_nm + rng.normal(0.0, 0.1, n), 0.0, touchdown_cum_nm)

    lat_deg = np.empty(n)
    lon_deg = np.empty(n)
    track_deg = np.empty(n)
    for i in range(n):
        lat_deg[i], lon_deg[i], track_deg[i] = _lat_lon_at_cum_nm(legs, cum_nm_noisy[i])

    vs_fpm = np.gradient(alt_ft, grid, edge_order=1) * 60.0

    # Synthetic weather, so the recorder's own three-way weather-chain
    # comparison (Q1) has real, non-degenerate columns to compare. The
    # along-track wind implied by gs - tas at each sample stands in for
    # "what the sim actually reports" (sim_*); Active Sky's reading at the
    # aircraft (as_*) is the SAME wind offset by as_wind_bias_kt, a stand-in
    # for the systematic hand-off error this schema exists to measure, not
    # a claim about any specific real value.
    along_wind_kt = gs_kt - tas_kt
    sim_wind_dir_deg = np.where(along_wind_kt >= 0.0, (track_deg + 180.0) % 360.0, track_deg)
    sim_wind_kt = np.abs(along_wind_kt)
    isa_t_k, _ = isa(alt_ft * 0.3048)
    sim_temp_c = isa_t_k - 273.15
    sim_pressure_hpa = fl_to_pressure(alt_ft / 100.0) / 100.0

    as_wind_kt = sim_wind_kt + as_wind_bias_kt
    as_wind_dir_deg = sim_wind_dir_deg
    as_temp_c = sim_temp_c
    as_pressure_hpa = sim_pressure_hpa

    fuel_kg = ((weight_t - zfw_t) * 1000.0) if zfw_t is not None else np.full(n, np.nan)

    return pd.DataFrame({
        "elapsed_s": grid,
        "zulu_s": grid % 86400.0,
        "lat_deg": lat_deg, "lon_deg": lon_deg,
        "cum_nm": cum_nm, "track_deg": track_deg,
        "alt_ft": alt_ft, "agl_ft": alt_ft,  # JFK/EGLL are both near sea level
        "vs_fpm": vs_fpm,
        "mach": mach, "cas_kt": np.nan, "tas_kt": tas_kt, "gs_kt": gs_kt,
        "weight_t": weight_t, "fuel_kg": fuel_kg, "on_ground": on_ground.astype(int),
        "as_wind_dir_deg": as_wind_dir_deg, "as_wind_kt": as_wind_kt,
        "as_temp_c": as_temp_c, "as_pressure_hpa": as_pressure_hpa,
        "sim_wind_dir_deg": sim_wind_dir_deg, "sim_wind_kt": sim_wind_kt,
        "sim_temp_c": sim_temp_c, "sim_pressure_hpa": sim_pressure_hpa,
    })
