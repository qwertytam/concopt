"""Phase 6: live in-flight altitude advisor plus flight recorder, for
`concopt inflight`. Reads the sim over SimConnect (python-SimConnect,
localhost -- P3D and Python are the same machine), projects --lookahead-nm
ahead along the loaded route, queries Active Sky there exactly the way
verify.py does (reuses verify._as_atmosphere, never reimplements it), and
feeds the result to limits.best_level UNCHANGED -- same ceiling, same
limits, same code search.march_legs uses. Real weight (TOTAL_WEIGHT) is
used directly; there is no burn-off schedule to integrate here.

SimConnect caveat, proven live against this project's P3D v5 install
(2026-09): python-SimConnect's own bundled SimConnect.dll does NOT speak
P3D v5's protocol version. Symptom: SimConnect() call from a script.py hangs
forever rather than raising, because SimConnect.connect() only breaks its
`while self.ok is False: pass` spin-wait on an OPEN event, and a version
mismatch gets a SIMCONNECT_EXCEPTION back instead of OPEN -- silently, no
exception, no timeout. Fix: point --simconnect-dll at a SimConnect.dll that
already works with the running sim -- any P3D add-on that talks SimConnect
ships one (this project's dev machine used FSLabs's
Libraries\\SimConnect_P3D_v5.dll; Little Navmap's install directory has one
too). Confirmed live: PLANE_LATITUDE read back correctly with that dll,
hung indefinitely with the bundled one.

RECORDER: elapsed-since-brake-release is measured off time.monotonic(), not
ZULU_TIME -- ZULU_TIME (seconds since midnight GMT) wraps at 86400 and a
flight can span that wrap; wall-clock elapsed time doesn't have that
problem and is what the comparison actually needs. ZULU_TIME is still
logged verbatim per row as the absolute reference.

RECORDER SCHEMA (RECORD_COLUMNS / RECORD_SCHEMA_VERSION, written as a
schema_version column on every row): one row per tick, flat, no nesting --
pd.read_csv and nothing else. There is no second chance at a given day's
weather, so the schema records what the three Phase C questions need:

  1. how wrong is the weather chain? THREE sources per sample, side by
     side: Active Sky at the aircraft (as_*, a point query at the
     aircraft's own altitude -- not the FL450-600 lookahead grid the
     advisor uses), what the sim says the aircraft is actually in (sim_*,
     AMBIENT_*), and what its own speeds imply (gs_minus_tas_kt). The
     ERA5-to-Active-Sky hand-off is already known to be systematic (mean
     -18.9 kt over seven verify runs); Active-Sky-to-actual has never been
     measured, and these columns are that measurement.
  2. what did the advice cost? rec_fl/rec_action/rec_gain_kt/rec_gain_s/
     binding, from _advisory -- the SAME numbers the panel line formats, so
     "recommended then followed" and "recommended then ignored" are both
     visible against level_fl on later rows.
  3. are the assumed constants right? agl_ft and phase cut the approach
     segment exactly, and the tick tightens below LOW_ALT_FT so a 1.5 min
     allowance gets ~18 samples rather than 1.

An unavailable value writes an EMPTY field, never 0 or a sentinel -- a zero
wind and a missing wind must not look the same (see _get_optional and
_round_or_none). Every optional column comes from a stock SimConnect
variable that an add-on as deep as FSLabs Concorde may simply not wire.

DISPLAY: by default (--live, the default) the advisor redraws one screen in
place with rich.live.Live rather than printing a fresh scrolling block every
tick -- see _render_screen -- since this runs on the sim PC while flying and
has to stay readable at a glance without scrolling. --no-live falls back to
the original scrolling prints (_print_advisor et al.), for piping to a log.
Presentation only: _level_table/_recommendation_line (the advisory logic),
_read_state (the SimConnect reads) and the recorder state machine in
run_inflight are the same either way.
"""
import csv
import time
from itertools import groupby

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text
from SimConnect import AircraftRequests, SimConnect

from concopt import arrival, limits
from concopt.asky import get_atmosphere_np
from concopt.atmos import (KT_TO_MS, fl_to_pressure, isa, pressure_to_fl,
                            speed_of_sound)
from concopt.route import (build_legs, current_progress_nm,
                            great_circle_nm, parse_pln, project_along_route,
                            supersonic_segment)
from concopt.params import (ACCEL_WAYPOINT_ID, ACTIVE_SKY_HOST, ACTIVE_SKY_PORT,  # noqa: F401
                            APPROACH_AGL_FT, ARRIVAL_REFRESH_S, ARRIVAL_WIND_SAMPLE_FL,
                            BRAKE_RELEASE_GS_KT, C_TO_K, DECEL_WAYPOINT_ID,
                            DEFAULT_GAIN_THRESHOLD_KT, DEFAULT_INTERVAL_S,
                            DEFAULT_LOOKAHEAD_NM, FASTEST_SCHEDULE_KT, FT_TO_M,
                            INHG_TO_HPA, LB_TO_KG, LOW_ALT_FT, LOW_ALT_INTERVAL_S,
                            NM_TO_M, SIMCONNECT_REQUEST_TIME_MS)
from concopt.search import TARGET_FL, _format_hmm

# --- Flight recorder schema --------------------------------------------------
# Bumped whenever RECORD_COLUMNS changes, and written as a column on every
# row so a notebook can tell which vintage it is reading without guessing
# from the header. 1 = the original narrow schema (zulu_s .. last_waypoint).
RECORD_SCHEMA_VERSION = 2

# One row per interval, flat, no nesting: pd.read_csv and nothing else. A
# value that is not available writes EMPTY, never 0 or a sentinel -- a zero
# wind and a missing wind must not look the same -- so every optional read
# below carries None through to csv.writer, which writes it as an empty
# field that pandas reads back as NaN. See _record_row for what each column
# is and why it is here.
RECORD_COLUMNS = [
    "schema_version",
    # time
    "zulu_s", "elapsed_s",
    # where, along which track
    "lat_deg", "lon_deg", "cum_nm", "last_waypoint", "leg_track_deg", "track_deg",
    # vertical
    "alt_ft", "level_fl", "agl_ft", "vs_fpm",
    # speeds
    "mach", "cas_kt", "tas_kt", "gs_kt",
    # mass and phase
    "weight_t", "fuel_kg", "on_ground", "phase",
    # weather chain, source 1: what Active Sky says is here
    "as_wind_dir_deg", "as_wind_kt", "as_wind_along_kt", "as_temp_c", "as_pressure_hpa",
    # weather chain, source 2: what the sim says the aircraft is actually in
    "sim_wind_dir_deg", "sim_wind_kt", "sim_wind_along_kt", "sim_temp_c", "sim_pressure_hpa",
    # weather chain, source 3: what the aircraft's own speeds imply
    "gs_minus_tas_kt",
    # what the advisor said, and why
    "rec_fl", "rec_action", "rec_gain_kt", "rec_gain_s", "binding",
    # what it predicted, so drift is visible over the flight, not just at the end
    "predicted_remaining_s", "predicted_total_s", "arrival_time_min", "arrival_source",
]



def _connect(dll_path=None):
    """Open a SimConnect session. See the module docstring's SimConnect
    caveat -- the printed warning is the whole mitigation; there's no way to
    time out connect()'s spin-wait from the outside without patching the
    library, so this just tells the user what a hang here means."""
    print("Connecting to SimConnect"
          + (f" via {dll_path}" if dll_path else " (bundled dll)")
          + " -- if this hangs, you're on the wrong SimConnect.dll version, "
            "Ctrl+C and pass --simconnect-dll (see inflight.py's docstring)")
    kwargs = {"library_path": dll_path} if dll_path else {}
    sm = SimConnect(**kwargs)
    aq = AircraftRequests(sm, _time=SIMCONNECT_REQUEST_TIME_MS)
    print("Connected.")
    return sm, aq


def _get_optional(aq, name, scale=1.0):
    """aq.get for a sim variable the recorder wants but can live without --
    python-SimConnect returns None both for a name it doesn't know and for
    data it hasn't received, and an add-on as deep as FSLabs Concorde does
    not necessarily wire every stock variable. Returns None (which
    csv.writer writes as an EMPTY field, read back as NaN) rather than a
    sentinel: a zero wind and a missing wind must not look the same.

    Contrast the required reads in _read_state, which deliberately crash on
    None -- if PLANE_LATITUDE stops arriving the run is over, but if
    AMBIENT_WIND_VELOCITY does, the rest of the recording is still worth
    having."""
    value = aq.get(name)
    if value is None:
        return None
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return None


def _read_state(aq):
    """One poll of the sim variables the advisor/recorder need. No
    reconnect/None-handling for the REQUIRED ones -- if Prepar3D stops
    responding mid-flight that should surface as a plain crash, not be
    swallowed. The optional block below (everything via _get_optional) is
    recorder-only telemetry that degrades to an empty CSV cell instead."""
    return dict(
        lat_deg=aq.get("PLANE_LATITUDE"),
        lon_deg=aq.get("PLANE_LONGITUDE"),
        alt_ft=aq.get("PLANE_ALTITUDE"),
        mach=aq.get("AIRSPEED_MACH"),
        tas_kt=aq.get("AIRSPEED_TRUE"),
        gs_kt=aq.get("GPS_GROUND_SPEED") / KT_TO_MS,
        weight_t=aq.get("TOTAL_WEIGHT") * LB_TO_KG / 1000.0,
        on_ground=bool(aq.get("SIM_ON_GROUND")),
        zulu_s=aq.get("ZULU_TIME"),
        # --- optional, recorder only ---
        # The whole model is a fuel calculation and nothing recorded fuel.
        fuel_kg=_get_optional(aq, "FUEL_TOTAL_QUANTITY_WEIGHT", LB_TO_KG),
        # What the aircraft is ACTUALLY flying in, for the Active-Sky-to-sim
        # hand-off nobody has measured. AMBIENT WIND DIRECTION is the
        # meteorological FROM bearing in degrees true -- the same convention
        # Active Sky reports, so both decompose with _wind_along_track_kt.
        sim_wind_kt=_get_optional(aq, "AMBIENT_WIND_VELOCITY"),
        sim_wind_dir_deg=_get_optional(aq, "AMBIENT_WIND_DIRECTION"),
        sim_temp_c=_get_optional(aq, "AMBIENT_TEMPERATURE"),
        sim_pressure_hpa=_get_optional(aq, "AMBIENT_PRESSURE", INHG_TO_HPA),
        # Height above the surface, for cutting the approach segment at
        # arrival.py's own 1,500 ft descent-end -- alt_ft can't do that at
        # an airport whose elevation isn't zero.
        agl_ft=_get_optional(aq, "PLANE_ALT_ABOVE_GROUND"),
        vs_fpm=_get_optional(aq, "VERTICAL_SPEED"),
        # CAS is one of the three binding limits (530 kt above FL430); with
        # it the limit model can be checked against what was actually held.
        cas_kt=_get_optional(aq, "AIRSPEED_INDICATED"),
        # Actual track made good, in RADIANS from SimConnect. Recorded next
        # to the route leg's own track so drift is visible.
        track_deg=_get_optional(aq, "GPS_GROUND_TRUE_TRACK", 180.0 / np.pi),
    )


def _live_weather_source(host, port):
    """THE SEAM's live weather_source: get_atmosphere_np bound to host/port,
    with get_atmosphere_np's own (lat, lon, alt_ft) -> (alt_ft, wind_dir_deg,
    wind_speed_kt, pressure_hpa, temp_c) shape unchanged. Every Active Sky
    touch in run_inflight goes through a weather_source built this way (or
    a replay one, same shape -- concopt.replay.replay_sources) rather than
    calling get_atmosphere_np/host/port directly, so a real flight is
    unaffected by the seam and a replay never has to know host/port at
    all."""
    def weather_source(lat, lon, alt_ft):
        return get_atmosphere_np(lat, lon, alt_ft, host_addr=host, port=port)
    return weather_source


def _lookahead_atmosphere(weather_source, lat, lon):
    """weather_source's counterpart to verify._as_atmosphere -- reimplemented
    here (rather than importing that function) so weather_source is the ONE
    place in run_inflight that talks to Active Sky, live or replayed. Same
    body as verify._as_atmosphere, with weather_source(lat, lon, alt_ft) in
    place of get_atmosphere_np(lat, lon, alt_ft, host_addr=host, port=port).
    Returns (temp_k, u_ms, v_ms), each (len(TARGET_FL),)."""
    target_ft = TARGET_FL * 100.0
    alt_ft, wind_dir_deg, wind_speed_kt, _pressure_hpa, temp_c = weather_source(
        lat, lon, target_ft)
    order = np.argsort(alt_ft)
    alt_ft = alt_ft[order]
    if not np.allclose(alt_ft, target_ft, atol=1.0):
        raise RuntimeError(
            f"weather_source returned altitudes {alt_ft.tolist()} for requested "
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


def _level_table(temp_k, u_ms, v_ms, track_deg, weight_t, cruise_mach, current_fl):
    """Everything the advisor needs for one lookahead point, across the
    FL450-FL600 TARGET_FL grid. limits.best_level does the actual level
    selection UNCHANGED (same ceiling, same limits, same code
    search.march_legs uses); the rest is the same small amount of
    surrounding per-level arithmetic march_legs/verify.py already carry
    (along-track wind, ISA deviation, binding limit), just for a single
    live point instead of a leg from the ERA5 archive.

    Returns (table, best_idx, current_idx, binding_at_best). table is a
    DataFrame indexed like TARGET_FL: fl, temp_c, isa_dev_c, wind_kt,
    max_mach, max_tas_kt, gs_kt, above_ceiling."""
    weight_arr = np.array([weight_t])
    _best_fl, _best_gs_ms, best_idx, gs_per_level = limits.best_level(
        TARGET_FL[None, :], temp_k[None, :], u_ms[None, :], v_ms[None, :],
        track_deg, weight_arr, cruise_mach,
    )
    best_idx = int(best_idx[0])
    gs_per_level = gs_per_level[0]

    track_rad = np.radians(track_deg)
    wind_per_level = u_ms * np.sin(track_rad) + v_ms * np.cos(track_rad)
    isa_t_per_level, _ = isa(TARGET_FL * 100.0 * FT_TO_M)
    isa_dev_per_level = temp_k - isa_t_per_level
    mach_per_level = limits.max_mach(TARGET_FL, temp_k, weight_t, cruise_mach)
    tas_ms_per_level = mach_per_level * speed_of_sound(temp_k)

    ceiling = limits.ceiling_ft(weight_t, isa_dev_per_level)
    above_ceiling = TARGET_FL * 100.0 > ceiling
    not_above = ~above_ceiling
    top_available_idx = int(np.max(np.flatnonzero(not_above))) if not_above.any() else -1
    mach_limit_label = limits.binding_mach_limit(TARGET_FL, temp_k, weight_t, cruise_mach)
    binding_at_best = "ceiling" if best_idx == top_available_idx else mach_limit_label[best_idx]

    current_idx = int(np.argmin(np.abs(TARGET_FL - current_fl)))

    table = pd.DataFrame(dict(
        fl=TARGET_FL, temp_c=temp_k - C_TO_K, isa_dev_c=isa_dev_per_level,
        wind_kt=wind_per_level / KT_TO_MS, max_mach=mach_per_level,
        max_tas_kt=tas_ms_per_level / KT_TO_MS, gs_kt=gs_per_level / KT_TO_MS,
        above_ceiling=above_ceiling,
    ))
    return table, best_idx, current_idx, binding_at_best


def _wind_along_track_kt(wind_kt, wind_dir_deg, track_deg):
    """Along-track wind component in kt, positive for a tailwind, from a
    wind given as a meteorological FROM bearing (degrees true) -- the
    convention BOTH Active Sky and SimConnect's AMBIENT WIND DIRECTION
    report in, so the two sources decompose identically and their
    difference means something. Same u/v inversion as verify._as_atmosphere
    followed by the same along-track projection limits.ground_speed uses.

    None in -> None out, so a missing wind stays an empty cell instead of
    becoming a confident zero."""
    if wind_kt is None or wind_dir_deg is None or track_deg is None:
        return None
    dir_rad = np.radians(wind_dir_deg)
    u_kt = -wind_kt * np.sin(dir_rad)
    v_kt = -wind_kt * np.cos(dir_rad)
    track_rad = np.radians(track_deg)
    return float(u_kt * np.sin(track_rad) + v_kt * np.cos(track_rad))


def _flight_phase(cum_nm, accel_cum_nm, decel_cum_nm, mach, agl_ft, on_ground):
    """climb / cruise / decel / descent / approach / ground for one sample,
    so a notebook can cut segments without re-inferring them from the
    altitude trace.

    Cut from the ROUTE position where possible (the accel and decel
    waypoints are exactly where search.py's own segment boundaries are, so
    the recording's segments line up with the model's), and from Mach only
    inside the arrival, where "decel" (M2.0 -> M1.0, still high) and
    "descent" (down to 1,500 ft) genuinely differ by speed rather than by
    position. approach needs agl_ft; where that read was unavailable the
    sample stays "descent" rather than guessing -- the empty agl_ft column
    is what says so."""
    if on_ground:
        return "ground"
    if cum_nm >= decel_cum_nm:
        if agl_ft is not None and agl_ft < APPROACH_AGL_FT:
            return "approach"
        return "decel" if (mach is not None and mach >= 1.0) else "descent"
    if cum_nm >= accel_cum_nm:
        return "cruise"
    return "climb"


def _as_point_atmosphere(weather_source, lat, lon, alt_ft):
    """Active Sky at the aircraft's OWN position and altitude -- one point,
    unlike _lookahead_atmosphere's FL450-FL600 grid at the LOOKAHEAD point.
    The grid query answers "where should I fly"; this one answers "what does
    Active Sky think the aircraft is in right now", which is the only thing
    comparable to what the sim reports it is actually in.

    Returns dict(wind_dir_deg, wind_kt, temp_c, pressure_hpa) or None if
    weather_source doesn't answer -- the recorder writes empty cells rather
    than dropping the row or inventing a calm."""
    try:
        _alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c = weather_source(
            lat, lon, [alt_ft])
    except (RuntimeError, TypeError, KeyError, ValueError, IndexError):
        return None
    if len(wind_dir_deg) < 1:
        return None
    return dict(wind_dir_deg=float(wind_dir_deg[0]), wind_kt=float(wind_speed_kt[0]),
                temp_c=float(temp_c[0]), pressure_hpa=float(pressure_hpa[0]))


def _build_live_arrival_wind_fn(weather_source, lat, lon, track_deg):
    """weather_source's counterpart to search._build_arrival_wind_fn: query
    weather_source at ARRIVAL_WIND_SAMPLE_FL (lat, lon) near the arrival
    region and return a wind_at_fl callable for arrival.arrival() -- linear
    in log(pressure) against the query's OWN returned pressure at each
    sampled altitude, the same convention search.py's stitched ERA5 profile
    uses. Wind direction is the meteorological FROM bearing (true), same
    u/v inversion as _lookahead_atmosphere.

    Returns None if weather_source doesn't answer for this point -- a live
    query this far ahead of the aircraft can fall outside Active Sky's
    loaded scenario/date, and the caller falls back to the pre-flight
    report's arrival figure rather than silently reverting to a flat
    constant. ALSO returns None if the response comes back so degenerate
    (see the duplicate-pressure paragraph below) that fewer than two
    distinct levels survive -- there is then no profile left to interpolate,
    and the same pre-flight fallback is the right answer, not a fabricated
    single-point number.

    DUPLICATE PRESSURES (D2, closing a gap left open at C3 -- see
    tests/test_replay.py's own weather approximation,
    concopt.replay._row_weather, for how a synthetic/degenerate response can
    trigger this): wind_at_fl's frac would divide by
    (src_log_p[idx1] - src_log_p[idx0]), which is silently 0 if TWO of the
    ARRIVAL_WIND_SAMPLE_FL queries come back at the SAME pressure -- giving
    NaN wind_kt/temp_k rather than raising, the same "silent, plausible,
    and produces numbers that look fine" failure shape verify.py's SNAPSHOT
    GUARD comment warns about elsewhere in this project. A real Active Sky
    response should never do this (pressure strictly decreases with
    altitude), but nothing here asserted it -- so below, once sorted by
    pressure, adjacent duplicate levels are collapsed (the first of each
    equal run kept) BEFORE building the interpolation source arrays, not
    papered over by clamping the divide -- clamping would turn a bad
    reading into a plausible-looking number, which is exactly what this is
    meant to avoid. Ten samples with one duplicate pair still leaves plenty
    to interpolate across; only the extreme case (every sample the same
    pressure, fewer than two distinct levels survive) falls back to None."""
    try:
        alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c = weather_source(
            lat, lon, ARRIVAL_WIND_SAMPLE_FL * 100.0)
    except (RuntimeError, TypeError, KeyError, ValueError):
        return None
    if (len(alt_ft) != len(ARRIVAL_WIND_SAMPLE_FL)
            or not np.all(np.isfinite(pressure_hpa)) or np.any(pressure_hpa <= 0.0)):
        return None

    order = np.argsort(pressure_hpa)  # ascending pressure -> descending FL
    pressure_sorted_hpa = pressure_hpa[order]
    src_log_p = np.log(pressure_sorted_hpa * 100.0)

    speed_ms = wind_speed_kt * KT_TO_MS
    dir_rad = np.radians(wind_dir_deg)
    u_ms = -speed_ms * np.sin(dir_rad)
    v_ms = -speed_ms * np.cos(dir_rad)
    track_rad = np.radians(track_deg)
    along_ms = (u_ms * np.sin(track_rad) + v_ms * np.cos(track_rad))[order]
    temp_k = (temp_c + C_TO_K)[order]

    # Collapse adjacent duplicate pressures (src_log_p is sorted, so any
    # duplicates are guaranteed adjacent) -- see the docstring above.
    keep = np.concatenate(([True], np.diff(src_log_p) > 0))
    if np.count_nonzero(keep) < 2:
        return None
    src_log_p = src_log_p[keep]
    along_ms = along_ms[keep]
    temp_k = temp_k[keep]
    pressure_sorted_hpa = pressure_sorted_hpa[keep]

    fl_min = float(pressure_to_fl(pressure_sorted_hpa[-1] * 100.0))
    fl_max = float(pressure_to_fl(pressure_sorted_hpa[0] * 100.0))

    def wind_at_fl(level_fl):
        level_fl = np.asarray(level_fl, dtype=float)
        fl_clamped = (level_fl < fl_min) | (level_fl > fl_max)
        level_fl = np.clip(level_fl, fl_min, fl_max)
        target_log_p = np.log(fl_to_pressure(level_fl))

        idx0 = np.clip(np.searchsorted(src_log_p, target_log_p, side="right") - 1,
                        0, len(src_log_p) - 2)
        idx1 = idx0 + 1
        frac = (target_log_p - src_log_p[idx0]) / (src_log_p[idx1] - src_log_p[idx0])

        wind_kt = (along_ms[idx0] + frac * (along_ms[idx1] - along_ms[idx0])) / KT_TO_MS
        temp_k_at = temp_k[idx0] + frac * (temp_k[idx1] - temp_k[idx0])
        return {"wind_kt": wind_kt, "temp_k": temp_k_at, "fl_clamped": fl_clamped}

    return wind_at_fl


def _live_arrival(cruise_fl, arrival_nm, isa_dev_c, mass_at_barix_t, wind_at_fl):
    """arrival.arrival() at the CURRENT state -- n_cand is 1 throughout, so
    every input is a scalar and the (1,)-shaped output arrays are squeezed
    back to floats. speed=380 matches arrival()'s own default (the call
    signature this was speced against); real pre-flight callers
    (fuel._arrival_from_march) use speed="auto" instead, to dodge 380 kt's
    NaN-fuel gap at low decel_end_fl mass -- but that gap is in
    level_fuel_t, not level_time_min, and only time feeds the live panel's
    predicted-remaining clock, so it doesn't matter here."""
    out = arrival.arrival(cruise_fl, arrival_nm, wind_at_fl, isa_dev_c,
                           mass_at_barix_t, speed=FASTEST_SCHEDULE_KT)
    return dict(
        time_min=float(out["time_min"][0]),
        decel_time_min=float(out["decel_time_min"][0]),
        level_time_min=float(out["level_time_min"][0]),
        descent_time_min=float(out["descent_time_min"][0]),
        level_fl=float(out["level_fl"][0]),
        level_wind_kt=float(out["level_wind_kt"][0]),
        flags=str(out["flags"][0]),
        source="live",
    )


def _build_arrival_text(arrival_info):
    """The Arrival (BARIX -> touchdown) breakdown for the live panel -- at
    most 4 lines (decel / level / descent+approach), per the layout spec:
    this panel is read at a glance while flying. arrival_info is None
    before anything is computable yet (no --compare report given AND Active
    Sky hasn't answered a live query yet); "fallback" means Active Sky
    didn't answer THIS tick's query and the pre-flight report's total is
    shown instead -- with no live decel/level/descent split, since the
    report CSV only carries the total (arrival_s), not report.py's own
    per-segment breakdown.

    D2: both the live and fallback lines carry an explicit [LIVE]/
    [PRE-FLIGHT] tag, not just a difference in wording -- read at a glance
    mid-flight, "no live decel/level/descent split" is easy to miss, and
    this is exactly the distinction that must not be missed (a stale
    pre-flight figure silently read as a live measurement)."""
    if arrival_info is None:
        return Text("Arrival (BARIX -> touchdown): not yet available")
    if arrival_info.get("source") == "fallback":
        return Text(
            "Arrival (BARIX -> touchdown) [PRE-FLIGHT]: Active Sky unavailable "
            f"this far ahead -- pre-flight figure {arrival_info['time_min']:.1f} min total"
        )
    return Text(
        f"Arrival (BARIX -> touchdown) [LIVE], {arrival_info['time_min']:.1f} min total:\n"
        f"  decel {arrival_info['decel_time_min']:.1f} min\n"
        f"  level {arrival_info['level_time_min']:.1f} min  FL{arrival_info['level_fl']:.0f}, "
        f"{arrival_info['level_wind_kt']:+.0f} kt\n"
        f"  descent {arrival_info['descent_time_min']:.1f} min  /  approach "
        f"{arrival.APPROACH_MIN:.1f} min"
    )


def _advisory(table, best_idx, current_idx, remaining_nm, gain_threshold_kt):
    """The advisory decision as NUMBERS: which level is recommended, what it
    is worth, and whether that clears gain_threshold_kt. Factored out of
    _recommendation_line (which now formats this) so the live panel and the
    recorder cannot drift apart -- without it the recorder would have to
    either re-derive the same arithmetic or parse the printed line back.

    The decision itself is unchanged: HOLD when already at the best level or
    when the gain is under the threshold, so the panel doesn't nag over
    noise. gain_kt/gain_s are reported even on a HOLD -- a near-miss
    recommendation is exactly what a post-flight read wants to see, and a
    HOLD at +2.9 kt is a different event from a HOLD at +0.0 kt.

    Returns dict(action, current_fl, rec_fl, gain_kt, gain_s); gain_s is
    None (-> an empty CSV cell) where a ground speed is non-positive rather
    than a divide-by-zero."""
    current_fl = float(table["fl"].iloc[current_idx])
    rec_fl = float(table["fl"].iloc[best_idx])
    gs_current_kt = float(table["gs_kt"].iloc[current_idx])
    gs_best_kt = float(table["gs_kt"].iloc[best_idx])
    gain_kt = gs_best_kt - gs_current_kt

    if gs_current_kt > 0.0 and gs_best_kt > 0.0:
        gain_s = remaining_nm * NM_TO_M * (
            1.0 / (gs_current_kt * KT_TO_MS) - 1.0 / (gs_best_kt * KT_TO_MS))
    else:
        gain_s = None

    if best_idx == current_idx or gain_kt < gain_threshold_kt:
        action = "HOLD"
    else:
        action = "CLIMB" if rec_fl > current_fl else "DESCEND"

    return dict(action=action, current_fl=current_fl, rec_fl=rec_fl,
                gain_kt=gain_kt, gain_s=gain_s)


def _recommendation_line(table, best_idx, current_idx, remaining_nm,
                          gain_threshold_kt, binding_at_best):
    """"CLIMB to FL550 (+14 kt, ~48 s over the remaining 1,240 nm) --
    binding: CAS" or "HOLD FL530" -- suppressed (HOLD) whenever the gain is
    under gain_threshold_kt, so this doesn't nag every tick over noise.
    Pure formatting of _advisory's numbers; the decision lives there."""
    adv = _advisory(table, best_idx, current_idx, remaining_nm, gain_threshold_kt)

    if adv["action"] == "HOLD":
        return f"HOLD FL{adv['current_fl']:.0f}"

    return (f"{adv['action']} to FL{adv['rec_fl']:.0f} ({adv['gain_kt']:+.0f} kt, "
            f"~{adv['gain_s']:.0f} s "
            f"over the remaining {remaining_nm:,.0f} nm) -- binding: {binding_at_best}")


def _print_advisor(state, la_lat, la_lon, track_deg, table, best_idx, current_idx,
                    binding_at_best, remaining_nm, gain_threshold_kt):
    note = []
    for i in range(len(table)):
        tags = []
        if i == current_idx:
            tags.append("current")
        if i == best_idx:
            tags.append("recommended")
        if table["above_ceiling"].iloc[i]:
            tags.append("above ceiling")
        note.append(", ".join(tags))

    display = pd.DataFrame({
        "FL": table["fl"].map(lambda f: f"FL{f:.0f}"),
        "temp_c": table["temp_c"].round(1),
        "isa_dev_c": table["isa_dev_c"].round(1),
        "wind_kt": table["wind_kt"].round(1),
        "max_mach": table["max_mach"].round(2),
        "max_tas_kt": table["max_tas_kt"].round(1),
        "gs_kt": table["gs_kt"].round(1),
        "note": note,
    })

    print(f"\nlat {state['lat_deg']:.3f} lon {state['lon_deg']:.3f} "
          f"alt {state['alt_ft']:.0f} ft  M{state['mach']:.2f}  "
          f"weight {state['weight_t']:.1f} t")
    print(f"lookahead point: ({la_lat:.3f}, {la_lon:.3f}), track {track_deg:.0f}")
    print(display.to_string(index=False))
    print(_recommendation_line(table, best_idx, current_idx, remaining_nm,
                                gain_threshold_kt, binding_at_best))


def _format_hmm_or_na(seconds):
    """_format_hmm, but None OR NaN -> 'n/a' -- the live panel shows this
    whenever a timing isn't knowable yet (elapsed before the recorder has
    seen brake release, or a pre-flight total with no --compare report
    given). NaN, not just None, because predicted_remaining_s/
    predicted_total_s are ARITHMETIC on arrival_info['time_min']
    (run_inflight: time_to_decel_s + arrival_time_s): a None guard on the
    inputs doesn't catch a NaN *value* flowing through that arithmetic
    (`x is not None` is True for float('nan')) -- found via the C3 replay
    harness, where a degenerate Active Sky reading (see
    _build_live_arrival_wind_fn's docstring) produced exactly that NaN and
    _format_hmm's plain int(round(seconds / 60.0)) crashed on it with
    ValueError, taking the whole live session down on the next redraw. Any
    unexpected NaN reaching a display function should render as "not
    available", not end the flight's advisor."""
    return "n/a" if seconds is None or np.isnan(seconds) else _format_hmm(seconds)


def _recommendation_text(line):
    """The plain _recommendation_line string, styled for the live panel --
    a climb/descend call stands out, a hold is muted, per the layout spec.
    Reads only the line's own leading "HOLD"/"CLIMB"/"DESCEND" word, so this
    can't drift out of step with _recommendation_line's actual wording or
    thresholds -- that function is untouched and still the only place the
    advisory decision is made."""
    style = "dim" if line.startswith("HOLD") else "bold white on dark_green"
    return Text(line, style=style, justify="center")


def _build_level_table(table, best_idx, current_idx):
    """The FL450-600 grid as a rich Table for the live panel -- the same
    values _print_advisor's plain DataFrame shows for --no-live, current/
    recommended marked in their own column instead of free-text "note" (no
    room for prose in a glance-readable table), above-ceiling rows greyed
    via row style rather than dropped, per the layout spec."""
    rich_table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
    for name, justify in [("FL", "right"), ("Wind kt", "right"), ("Temp C", "right"),
                           ("ISA dev", "right"), ("Mmax", "right"), ("TASmax kt", "right"),
                           ("GS kt", "right"), ("", "left")]:
        rich_table.add_column(name, justify=justify)

    for i in range(len(table)):
        row = table.iloc[i]
        tags = []
        if i == current_idx:
            tags.append("CURRENT")
        if i == best_idx:
            tags.append("REC")

        if row["above_ceiling"]:
            style = "grey50"
        elif i == best_idx:
            style = "bold green"
        elif i == current_idx:
            style = "bold cyan"
        else:
            style = None

        rich_table.add_row(
            f"FL{row['fl']:.0f}", f"{row['wind_kt']:+.0f}", f"{row['temp_c']:.1f}",
            f"{row['isa_dev_c']:+.1f}", f"{row['max_mach']:.2f}",
            f"{row['max_tas_kt']:.0f}", f"{row['gs_kt']:.0f}", " ".join(tags),
            style=style,
        )
    return rich_table


def _render_screen(state, table, best_idx, current_idx, binding_at_best, remaining_nm,
                    gain_threshold_kt, next_wp_id, dist_to_next_nm, distance_run_nm,
                    elapsed_s, predicted_remaining_s, predicted_total_s,
                    preflight_predicted_total_s, countdown_s, arrival_info=None,
                    recorder_note=None):
    """Everything --live (the default) redraws in place each tick, as one
    rich renderable -- pure and testable via rich.console.Console(record=
    True), no terminal or Live instance needed. Layout, top to bottom, is
    the priority order from the spec: the recommendation first (the only
    line that matters at a glance), then current state, the level table,
    the arrival breakdown (arrival_info -- see _live_arrival/
    _build_arrival_text), progress (predicted total shown live alongside
    the pre-flight figure so a divergence is visible in the air), and a
    countdown so a static screen still looks alive rather than hung between
    ticks."""
    line = _recommendation_line(table, best_idx, current_idx, remaining_nm,
                                 gain_threshold_kt, binding_at_best)
    recommendation = _recommendation_text(line)

    state_text = Text(
        f"FL{state['alt_ft'] / 100.0:.0f}   M{state['mach']:.2f}   "
        f"TAS {state['tas_kt']:.0f} kt   GS {state['gs_kt']:.0f} kt   "
        f"Weight {state['weight_t']:.1f} t\n"
        f"{state['lat_deg']:.3f}, {state['lon_deg']:.3f}   "
        f"next: {next_wp_id} ({dist_to_next_nm:.0f} nm)"
    )

    level_table = _build_level_table(table, best_idx, current_idx)

    arrival_text = _build_arrival_text(arrival_info)

    progress_text = Text(
        f"Distance run {distance_run_nm:,.0f} nm / to run {remaining_nm:,.0f} nm   "
        f"Elapsed {_format_hmm_or_na(elapsed_s)}   "
        f"Remaining {_format_hmm_or_na(predicted_remaining_s)}\n"
        f"Predicted total: live {_format_hmm_or_na(predicted_total_s)}   "
        f"vs pre-flight {_format_hmm_or_na(preflight_predicted_total_s)}"
    )

    countdown_text = Text(f"next update in {countdown_s:.0f}s", style="dim", justify="right")

    parts = [recommendation, Text(""), state_text, Text(""), level_table, Text(""),
             arrival_text, Text(""), progress_text]
    if recorder_note:
        parts.append(Text(recorder_note, style="yellow"))
    parts.append(countdown_text)
    return Group(*parts)


def _round_or_none(value, ndigits):
    """round() that passes None through, so an unavailable reading stays
    None (an empty CSV cell) instead of raising on the way out."""
    return None if value is None else round(float(value), ndigits)


def _record_row(elapsed_s, state, cum_nm, last_waypoint, leg_track_deg, flight_phase,
                 as_point, advisory, binding_at_best, arrival_info,
                 predicted_remaining_s, predicted_total_s):
    """One recorder row as a {column: value} dict, in RECORD_COLUMNS order.
    Pure -- no sim, no Active Sky, no file -- so the schema is testable
    without a flight.

    The three weather sources sit side by side deliberately (question 1 of
    the three this recording exists to answer): Active Sky's own reading AT
    THE AIRCRAFT (as_*), what the sim says the aircraft is actually flying
    in (sim_*), and what the aircraft's own speeds imply (gs_minus_tas_kt).
    The ERA5-to-Active-Sky hand-off is already known to be systematic
    (mean -18.9 kt over seven verify runs); the Active-Sky-to-actual one
    has never been measured, and these columns are the measurement.

    as_wind_along_kt and sim_wind_along_kt are both projected on
    leg_track_deg -- the ROUTE leg's track at the aircraft, the same track
    search.march_legs and limits.ground_speed decompose on -- so the two
    are directly comparable to each other AND to the ERA5/Active Sky
    numbers everywhere else in this project. The aircraft's own track made
    good is recorded separately as track_deg: drift makes it differ, and
    gs_minus_tas_kt (which is inherently in the aircraft's frame, not the
    route's) is the column that difference shows up in. Raw direction and
    speed are kept for both sources so any other decomposition can be
    redone offline from the file alone."""
    as_point = as_point or {}
    as_wind_kt = as_point.get("wind_kt")
    as_wind_dir_deg = as_point.get("wind_dir_deg")

    tas_kt, gs_kt = state["tas_kt"], state["gs_kt"]
    gs_minus_tas_kt = (None if tas_kt is None or gs_kt is None else float(gs_kt) - float(tas_kt))

    arrival_info = arrival_info or {}

    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "zulu_s": state["zulu_s"],
        "elapsed_s": _round_or_none(elapsed_s, 1),
        "lat_deg": _round_or_none(state["lat_deg"], 6),
        "lon_deg": _round_or_none(state["lon_deg"], 6),
        "cum_nm": _round_or_none(cum_nm, 2),
        "last_waypoint": last_waypoint,
        "leg_track_deg": _round_or_none(leg_track_deg, 2),
        "track_deg": _round_or_none(state.get("track_deg"), 2),
        "alt_ft": _round_or_none(state["alt_ft"], 1),
        # The flown flight level, so a step climb and "was the advice
        # taken" are one column apart rather than an arithmetic away.
        "level_fl": _round_or_none(state["alt_ft"] / 100.0, 1),
        "agl_ft": _round_or_none(state.get("agl_ft"), 1),
        "vs_fpm": _round_or_none(state.get("vs_fpm"), 1),
        "mach": _round_or_none(state["mach"], 4),
        "cas_kt": _round_or_none(state.get("cas_kt"), 2),
        "tas_kt": _round_or_none(tas_kt, 2),
        "gs_kt": _round_or_none(gs_kt, 2),
        "weight_t": _round_or_none(state["weight_t"], 3),
        "fuel_kg": _round_or_none(state.get("fuel_kg"), 1),
        "on_ground": int(bool(state["on_ground"])),
        "phase": flight_phase,
        "as_wind_dir_deg": _round_or_none(as_wind_dir_deg, 2),
        "as_wind_kt": _round_or_none(as_wind_kt, 2),
        "as_wind_along_kt": _round_or_none(
            _wind_along_track_kt(as_wind_kt, as_wind_dir_deg, leg_track_deg), 2),
        "as_temp_c": _round_or_none(as_point.get("temp_c"), 2),
        "as_pressure_hpa": _round_or_none(as_point.get("pressure_hpa"), 2),
        "sim_wind_dir_deg": _round_or_none(state.get("sim_wind_dir_deg"), 2),
        "sim_wind_kt": _round_or_none(state.get("sim_wind_kt"), 2),
        "sim_wind_along_kt": _round_or_none(
            _wind_along_track_kt(state.get("sim_wind_kt"), state.get("sim_wind_dir_deg"),
                                  leg_track_deg), 2),
        "sim_temp_c": _round_or_none(state.get("sim_temp_c"), 2),
        "sim_pressure_hpa": _round_or_none(state.get("sim_pressure_hpa"), 2),
        "gs_minus_tas_kt": _round_or_none(gs_minus_tas_kt, 2),
        "rec_fl": _round_or_none(advisory["rec_fl"], 1),
        "rec_action": advisory["action"],
        "rec_gain_kt": _round_or_none(advisory["gain_kt"], 2),
        "rec_gain_s": _round_or_none(advisory["gain_s"], 1),
        "binding": binding_at_best,
        "predicted_remaining_s": _round_or_none(predicted_remaining_s, 1),
        "predicted_total_s": _round_or_none(predicted_total_s, 1),
        "arrival_time_min": _round_or_none(arrival_info.get("time_min"), 2),
        # "live" vs "fallback" -- without it a pre-flight fallback figure
        # reads as if it were a live measurement of that moment.
        "arrival_source": arrival_info.get("source"),
    }


def _row_values(row):
    """A _record_row dict flattened to RECORD_COLUMNS order for csv.writer.
    None stays None: csv.writer writes it as an empty field, which
    pd.read_csv reads back as NaN -- never as 0."""
    return [row[column] for column in RECORD_COLUMNS]


def _waypoint_cum_nm(legs):
    """{waypoint_id: cum_nm}, in route order. Sub-legs of a subdivided
    parent leg all share that parent's from_id/to_id (see route.build_legs),
    so only the last sub-leg of each run carries the true cumulative
    distance to that waypoint -- same groupby report._waypoint_table uses."""
    table = {legs[0].from_id: legs[0].cum_nm - legs[0].dist_nm}
    for (_from_id, to_id), members in groupby(
            range(len(legs)), key=lambda i: (legs[i].from_id, legs[i].to_id)):
        last = list(members)[-1]
        table[to_id] = legs[last].cum_nm
    return table


def _last_waypoint_passed(waypoint_cum_nm, current_cum_nm):
    """The id of the last waypoint (in route order) at or before
    current_cum_nm."""
    passed = [wid for wid, cum in waypoint_cum_nm.items() if cum <= current_cum_nm]
    return passed[-1] if passed else next(iter(waypoint_cum_nm))


def _parse_hmm_seconds(s):
    """Inverse of search._format_hmm: 'h:mm' -> seconds."""
    h, m = str(s).split(":")
    return (int(h) * 60 + int(m)) * 60.0


def _report_arrival_s(report_df):
    """report_df's own arrival_s column (report.py, on the decel waypoint's
    row) -- the report's OWN prediction of BARIX -> touchdown time, from the
    same real per-day arrival.arrival() model the report was built with.
    Raises rather than silently falling back to the flat 35 min
    placeholder this was written to replace: a report CSV predating the
    column means `concopt report` needs rerunning, not a quiet regression to
    the old flat number."""
    if "arrival_s" not in report_df.columns:
        raise ValueError(
            "report CSV has no 'arrival_s' column -- it was written by an "
            "older concopt report; rerun concopt report to regenerate it "
            "with the real per-day arrival model"
        )
    return float(report_df["arrival_s"].dropna().iloc[-1])


def _preflight_predicted_total_s(report_df):
    """report_df's own last elapsed (brake release -> decel waypoint) plus
    its own arrival_s -- NOT the flat 35 min constant: the report CSV was itself
    generated with the real arrival model, so adding the flat 35 min
    constant on top double-counts against a plan that already accounts for
    its own (usually ~30 min, but day-dependent) arrival."""
    return _parse_hmm_seconds(report_df["elapsed"].iloc[-1]) + _report_arrival_s(report_df)


def compare_to_report(report_df, cpa, touchdown_elapsed_s, accel_id=ACCEL_WAYPOINT_ID, decel_id=DECEL_WAYPOINT_ID):
    """Predicted (report_df, a concopt report --out CSV) vs actual (cpa, the
    recorder's per-waypoint closest-point-of-approach snapshots) at every
    waypoint report_df covers, plus the three numbers the recorder exists to
    measure. Pure and testable -- no printing, no SimConnect, no file I/O
    (run_inflight does all three); mirrors verify.py's split between
    computation (_as_atmosphere) and the orchestration that prints
    (run_verify).

    Returns (table, constants): table is a DataFrame of predicted/actual/
    delta per waypoint; constants is a dict of the three feedback numbers
    (measured_*_s, plus the report's own predicted brake-to-accel/decel-to-
    touchdown/supersonic times, for the caller to print against).
    predicted_brake_to_accel_s comes from report_df's own accel_id row
    (report.py's climb model makes this vary by day/TOW, so it's read back
    from the report rather than a fixed constant)."""
    rows = []
    for _, r in report_df.iterrows():
        wid = r["waypoint"]
        measured = cpa.get(wid)
        rows.append(dict(
            waypoint=wid,
            pred_elapsed_s=_parse_hmm_seconds(r["elapsed"]),
            actual_elapsed_s=measured["elapsed_s"] if measured else np.nan,
            pred_fl=float(r["chosen_fl"]),
            actual_fl=measured["fl"] if measured else np.nan,
            pred_gs_kt=float(r["gs_kt"]) if not pd.isna(r["gs_kt"]) else np.nan,
            actual_gs_kt=measured["gs_kt"] if measured else np.nan,
        ))
    table = pd.DataFrame(rows)
    table["delta_elapsed_s"] = table["actual_elapsed_s"] - table["pred_elapsed_s"]
    table["delta_fl"] = table["actual_fl"] - table["pred_fl"]
    table["delta_gs_kt"] = table["actual_gs_kt"] - table["pred_gs_kt"]

    accel_rows = table[table["waypoint"] == accel_id]
    decel_rows = table[table["waypoint"] == decel_id]
    have_both = not accel_rows.empty and not decel_rows.empty
    accel_actual_s = float(accel_rows["actual_elapsed_s"].iloc[0]) if not accel_rows.empty else np.nan
    decel_actual_s = float(decel_rows["actual_elapsed_s"].iloc[0]) if not decel_rows.empty else np.nan

    constants = dict(
        measured_brake_to_accel_s=accel_actual_s,
        predicted_brake_to_accel_s=(float(accel_rows["pred_elapsed_s"].iloc[0])
                                     if not accel_rows.empty else np.nan),
        measured_decel_to_touchdown_s=(touchdown_elapsed_s - decel_actual_s
                                        if not decel_rows.empty else np.nan),
        # The report's own real per-day arrival total (report.py's
        # arrival_s column), not the old flat 35 min constant the plan
        # was actually predicted with -- comparing against the constant
        # would flag every day's real vs-real difference as a model error.
        predicted_decel_to_touchdown_s=_report_arrival_s(report_df),
        measured_supersonic_s=(decel_actual_s - accel_actual_s) if have_both else np.nan,
        predicted_supersonic_s=(float(decel_rows["pred_elapsed_s"].iloc[0])
                                 - float(accel_rows["pred_elapsed_s"].iloc[0])
                                 if have_both else np.nan),
    )
    return table, constants


def _print_comparison(table, constants):
    display = pd.DataFrame({
        "waypoint": table["waypoint"],
        "pred_elapsed": table["pred_elapsed_s"].map(_format_hmm),
        "actual_elapsed": table["actual_elapsed_s"].map(
            lambda s: "" if pd.isna(s) else _format_hmm(s)),
        "delta_min": (table["delta_elapsed_s"] / 60.0).round(1),
        "pred_fl": table["pred_fl"].round(0),
        "actual_fl": table["actual_fl"].round(0),
        "pred_gs_kt": table["pred_gs_kt"].round(1),
        "actual_gs_kt": table["actual_gs_kt"].round(1),
    })
    print("\nPredicted vs actual, by waypoint:")
    print(display.to_string(index=False))

    def _line(label, measured_s, predicted_s):
        if np.isnan(measured_s):
            print(f"  {label}: not measured (waypoint missing from the recording)")
            return
        diff_min = (measured_s - predicted_s) / 60.0
        print(f"  {label}: measured {_format_hmm(measured_s)} vs {_format_hmm(predicted_s)} "
              f"(update these constants: {diff_min:+.1f} min)")

    print("\nFeedback into the model:")
    _line("brake-release -> accel point", constants["measured_brake_to_accel_s"],
          constants["predicted_brake_to_accel_s"])
    _line("decel point -> touchdown", constants["measured_decel_to_touchdown_s"],
          constants["predicted_decel_to_touchdown_s"])
    _line("supersonic segment", constants["measured_supersonic_s"],
          constants["predicted_supersonic_s"])


def run_inflight(pln_path, interval_s=DEFAULT_INTERVAL_S, lookahead_nm=DEFAULT_LOOKAHEAD_NM,
                  record_path=None, compare_path=None, accel_id=ACCEL_WAYPOINT_ID, decel_id=DECEL_WAYPOINT_ID,
                  host=ACTIVE_SKY_HOST, port=ACTIVE_SKY_PORT, cruise_mach=limits.CRUISE_MACH,
                  gain_threshold_kt=DEFAULT_GAIN_THRESHOLD_KT, simconnect_dll=None, live=True,
                  state_source=None, weather_source=None, replay_speed=1.0):
    """The live advisor loop, every interval_s: read the sim, project
    lookahead_nm ahead along the route, query Active Sky there, show the
    level table and recommendation. If record_path is given, also runs the
    flight recorder as a small state machine alongside it (brake release ->
    recording -> touchdown), writing one row per interval once recording
    has started, and exits once touchdown is detected. If compare_path is
    also given, runs compare_to_report against the finished recording at
    that point and prints the result.

    live (default True) redraws one screen in place with rich.live.Live --
    see _render_screen. --no-live sets live=False, falling back to the
    original scrolling _print_advisor/plain-print behaviour, for piping to
    a log. This is a presentation choice only: the advisory logic
    (_level_table/_recommendation_line), the SimConnect reads (_read_state)
    and the recorder state machine below are identical either way.

    THE SEAM (Phase C3): every sim read goes through state_source() (the
    same shape _read_state(aq) returns) and every Active Sky touch goes
    through weather_source(lat, lon, alt_ft) (the same shape get_atmosphere_np
    returns -- see _live_weather_source/_lookahead_atmosphere/
    _as_point_atmosphere/_build_live_arrival_wind_fn). Both default to None,
    meaning "connect for real": a SimConnect session is opened (below) and
    state_source becomes _read_state bound to it, and weather_source becomes
    _live_weather_source(host, port) -- a real flight is completely
    unaffected by this seam existing. Given explicitly (concopt.replay's
    replay_sources, built from a recorded flight or a synthetic one), NO
    SimConnect connection is attempted at all -- run_inflight never learns
    the difference between a real flight and a replayed one beyond that.

    replay_speed (default 1.0, a real flight) scales every wall-clock use in
    the loop below by the same factor: sleeps are divided by it (so a
    replay can run faster than real time) and the two elapsed-time
    measurements below (elapsed_s, touchdown_elapsed_s -- both otherwise
    time.monotonic() since brake release, per the module docstring's
    RECORDER note) are multiplied back up by it, so a compressed replay's
    recording still reports true flight-equivalent seconds rather than the
    compressed wall-clock duration it actually took to replay. This is a
    plain scalar, not a third injectable source -- the two callables above
    are the whole seam.

    Covered by tests/test_replay.py via the seam above (a live SimConnect +
    Active Sky are never touched there); see compare_to_report and
    _level_table/_recommendation_line/_render_screen for the other pure
    pieces."""
    plan = parse_pln(pln_path)
    legs = build_legs(plan["waypoints"])
    mask = supersonic_segment(legs, accel_id=accel_id, decel_id=decel_id)
    ss_legs = [leg for leg, m in zip(legs, mask) if m]
    route_end_cum_nm = ss_legs[-1].cum_nm  # decel waypoint's own cum_nm
    # The accel waypoint's own cum_nm (start of the first supersonic leg) --
    # the recorder's climb/cruise boundary, so its phase column cuts at the
    # same places search.py's own segments do.
    accel_cum_nm = ss_legs[0].cum_nm - ss_legs[0].dist_nm

    # The arrival segment itself -- decel waypoint through touchdown, the
    # same post-decel span route.climb_cruise_segment's complement covers
    # in search.py/report.py -- for the live arrival.arrival() call below.
    decel_cum_nm = route_end_cum_nm
    touchdown_cum_nm = legs[-1].cum_nm
    arrival_nm = touchdown_cum_nm - decel_cum_nm

    # A representative arrival-leg position -- search._build_arrival_wind_fn's
    # same "sample near the midpoint of the post-decel span" convention --
    # for querying Active Sky's live arrival wind profile each tick. None if
    # decel_id is the route's last waypoint (no post-decel legs at all).
    decel_leg_idx = int(np.flatnonzero(mask)[-1])
    arrival_legs = legs[decel_leg_idx + 1:]
    if arrival_legs:
        arr_leg = arrival_legs[len(arrival_legs) // 2]
        arr_lat, arr_lon, arr_track_deg = arr_leg.lat_mid, arr_leg.lon_mid, arr_leg.track_deg
    else:
        arr_lat = arr_lon = arr_track_deg = None

    # Read the compare report up front (not just at touchdown) so its total
    # predicted time can sit in the live panel's progress line throughout
    # the flight, not just in the post-touchdown comparison. Its own
    # arrival_s also backs the live arrival estimate's Active-Sky-unavailable
    # fallback below.
    report_df = None
    preflight_predicted_total_s = None
    preflight_arrival_s = None
    if compare_path is not None:
        report_df = pd.read_csv(compare_path)
        preflight_arrival_s = _report_arrival_s(report_df)
        preflight_predicted_total_s = _preflight_predicted_total_s(report_df)

    # THE SEAM: state_source/weather_source given (replay) -> no SimConnect
    # connection is attempted at all, live or otherwise. Neither given (a
    # real flight) -> connect for real and build the live defaults, exactly
    # the calls this replaced.
    if state_source is None:
        sm, aq = _connect(simconnect_dll)
        state_source = lambda: _read_state(aq)  # noqa: E731
    if weather_source is None:
        weather_source = _live_weather_source(host, port)

    recording = record_path is not None
    csv_file = None
    cpa = {}
    wp_positions = {}
    waypoint_cum_nm = {}
    if recording:
        wp_positions = {wid: (lat, lon) for wid, lat, lon in plan["waypoints"]}
        cpa = {wid: None for wid in wp_positions}
        waypoint_cum_nm = _waypoint_cum_nm(legs)
        csv_file = open(record_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(RECORD_COLUMNS)

    phase = "ground"
    prev_on_ground = None
    prev_gs_kt = None
    t0_monotonic = None
    touchdown_elapsed_s = None
    recorder_note = None  # live panel only -- --no-live prints these as before
    # The live arrival estimate is refreshed at most every ARRIVAL_REFRESH_S
    # (see that constant) and reused in between, so the low-altitude tick
    # doesn't multiply its 10-level Active Sky query.
    arrival_info = None
    arrival_refresh_at = None

    live_display = Live(console=Console()) if live else None
    if live_display is not None:
        live_display.start()

    interrupted = False
    try:
        while True:
            state = state_source()
            lat, lon = state["lat_deg"], state["lon_deg"]

            la_lat, la_lon, track_deg = project_along_route(legs, lat, lon, lookahead_nm)
            current_cum_nm, leg_idx, along_nm = current_progress_nm(legs, lat, lon)
            remaining_nm = max(route_end_cum_nm - current_cum_nm, 0.0)  # to the decel waypoint
            total_remaining_nm = max(touchdown_cum_nm - current_cum_nm, 0.0)  # to touchdown
            next_wp_id = legs[leg_idx].to_id
            dist_to_next_nm = max(legs[leg_idx].dist_nm - along_nm, 0.0)

            temp_k, u_ms, v_ms = _lookahead_atmosphere(weather_source, la_lat, la_lon)
            current_fl = state["alt_ft"] / 100.0
            table, best_idx, current_idx, binding_at_best = _level_table(
                temp_k, u_ms, v_ms, track_deg, state["weight_t"], cruise_mach, current_fl)

            # Live arrival estimate, at the CURRENT state -- not the
            # pre-flight one, which is the entire point of an in-flight
            # advisor. arrival_wind_fn is None (query Active Sky failed, or
            # this route has no post-decel span) -> fall back to the
            # pre-flight report's own arrival total, flagged as a fallback
            # rather than silently reused as if it were live. Throttled to
            # ARRIVAL_REFRESH_S so the tightened low-altitude tick doesn't
            # repeat its 10-level query 12x as often; at the default
            # interval this refreshes every tick exactly as before.
            now_monotonic = time.monotonic()
            if arrival_refresh_at is None or now_monotonic >= arrival_refresh_at:
                arrival_wind_fn = (
                    _build_live_arrival_wind_fn(weather_source, arr_lat, arr_lon, arr_track_deg)
                    if arr_lat is not None else None)
                if arrival_wind_fn is not None:
                    isa_dev_at_cruise = float(table["isa_dev_c"].iloc[current_idx])
                    # Projected mass at the decel waypoint = current weight minus
                    # the cruise burn still to come; no burn-off schedule is
                    # tracked here (see module docstring), so current weight is
                    # used as-is -- a few tonnes heavy vs the real BARIX mass,
                    # which understates specific range slightly on the subsonic
                    # table lookup.
                    arrival_info = _live_arrival(current_fl, arrival_nm, isa_dev_at_cruise,
                                                  state["weight_t"], arrival_wind_fn)
                elif preflight_arrival_s is not None:
                    arrival_info = dict(time_min=preflight_arrival_s / 60.0, source="fallback")
                else:
                    arrival_info = None
                # /replay_speed: ARRIVAL_REFRESH_S is a flight-time quantity: a
                # replay compresses wall-clock time by replay_speed, so the
                # wall-clock threshold must shrink by the same factor to still
                # mean "once every ARRIVAL_REFRESH_S of FLIGHT time" (real
                # flight, replay_speed=1.0, is unaffected).
                arrival_refresh_at = now_monotonic + ARRIVAL_REFRESH_S / replay_speed

            # *replay_speed: elapsed_s/touchdown_elapsed_s are wall-clock
            # (time.monotonic()) since brake release, per the module
            # docstring's RECORDER note -- a replay compresses that wall
            # clock by replay_speed to run faster than real time, so the
            # measurement is scaled back up here to report true flight-
            # equivalent seconds rather than the compressed replay duration
            # (real flight, replay_speed=1.0, is unaffected).
            elapsed_s = ((time.monotonic() - t0_monotonic) * replay_speed
                         if phase in ("recording", "done") else None)
            current_gs_kt = float(table["gs_kt"].iloc[current_idx])
            if current_cum_nm < decel_cum_nm:
                # Still cruising: time to the decel waypoint at the current
                # ground speed, plus the arrival estimate above (a fixed
                # BARIX -> touchdown total, not a remaining-distance one --
                # arrival()'s decel/level/descent split only makes sense
                # starting from the decel waypoint itself).
                time_to_decel_s = (remaining_nm * NM_TO_M / (current_gs_kt * KT_TO_MS)
                                    if current_gs_kt > 0 else None)
                arrival_time_s = arrival_info["time_min"] * 60.0 if arrival_info is not None else None
                predicted_remaining_s = (
                    time_to_decel_s + arrival_time_s
                    if time_to_decel_s is not None and arrival_time_s is not None else None)
            else:
                # Already past the decel waypoint: arrival()'s segment split
                # no longer applies to what's left to fly, so extrapolate
                # the current ground speed over the true remaining distance
                # to touchdown instead (this is what "remaining" used to
                # silently drop to 0 for, the bug this task fixes).
                predicted_remaining_s = (
                    total_remaining_nm * NM_TO_M / (current_gs_kt * KT_TO_MS)
                    if current_gs_kt > 0 else None)
            predicted_total_s = (
                elapsed_s + predicted_remaining_s
                if elapsed_s is not None and predicted_remaining_s is not None else None)

            # Sample faster near the ground so the arrival's short segments
            # are resolvable at all -- see LOW_ALT_FT. Only while recording:
            # with no --record there is nothing to resolve, and the advisor
            # itself has no reason to redraw 12x as often.
            tick_s = interval_s
            if recording and state["alt_ft"] < LOW_ALT_FT:
                tick_s = min(interval_s, LOW_ALT_INTERVAL_S)

            def _screen(countdown_s):
                return _render_screen(
                    state, table, best_idx, current_idx, binding_at_best, remaining_nm,
                    gain_threshold_kt, next_wp_id, dist_to_next_nm, current_cum_nm,
                    elapsed_s, predicted_remaining_s, predicted_total_s,
                    preflight_predicted_total_s, countdown_s, arrival_info, recorder_note)

            if live_display is not None:
                live_display.update(_screen(tick_s))
            else:
                _print_advisor(state, la_lat, la_lon, track_deg, table, best_idx, current_idx,
                                binding_at_best, remaining_nm, gain_threshold_kt)

            if recording:
                gs_kt = state["gs_kt"]
                # prev_gs_kt is None only on the very first sample -- normally
                # that just means "nothing to compare yet, wait for a real
                # crossing". But if state_source's FIRST reading is already
                # past BRAKE_RELEASE_GS_KT (SimConnect started late after a
                # real brake release, or a replay/recording that begins
                # mid-roll), there is no crossing left to observe and the
                # OR-less version of this check never fires -- the recorder
                # would silently never start, writing nothing but the header
                # for the entire flight. Found via the C3 replay harness
                # (first synthetic-flight run: an off-by-one start position
                # produced exactly this, an all-header CSV) -- in the air
                # this is "the advisor was started a few seconds late" and
                # would have lost the whole recording with no error at all.
                if phase == "ground" and (
                        (prev_gs_kt is not None and prev_gs_kt < BRAKE_RELEASE_GS_KT <= gs_kt)
                        or (prev_gs_kt is None and gs_kt >= BRAKE_RELEASE_GS_KT)
                ):
                    phase = "recording"
                    t0_monotonic = time.monotonic()
                    recorder_note = f"Brake release detected -- recording to {record_path}"
                    if live_display is None:
                        print(f"\n{recorder_note}")
                elif phase == "recording" and prev_on_ground is False and state["on_ground"]:
                    phase = "done"
                    touchdown_elapsed_s = (time.monotonic() - t0_monotonic) * replay_speed
                    recorder_note = (f"Touchdown detected, elapsed "
                                      f"{_format_hmm(touchdown_elapsed_s)} since brake release")
                    if live_display is None:
                        print(f"\n{recorder_note}")

                if phase in ("recording", "done"):
                    elapsed_row_s = (time.monotonic() - t0_monotonic) * replay_speed
                    last_wp = _last_waypoint_passed(waypoint_cum_nm, current_cum_nm)
                    # Active Sky AT THE AIRCRAFT -- a different query from the
                    # lookahead grid above, and the only one comparable to
                    # what the sim reports the aircraft is actually flying in.
                    as_point = _as_point_atmosphere(weather_source, lat, lon, state["alt_ft"])
                    row = _record_row(
                        elapsed_row_s, state, current_cum_nm, last_wp,
                        legs[leg_idx].track_deg,
                        _flight_phase(current_cum_nm, accel_cum_nm, decel_cum_nm,
                                       state["mach"], state.get("agl_ft"), state["on_ground"]),
                        as_point,
                        _advisory(table, best_idx, current_idx, remaining_nm, gain_threshold_kt),
                        binding_at_best, arrival_info,
                        predicted_remaining_s, predicted_total_s)
                    csv_writer.writerow(_row_values(row))
                    csv_file.flush()
                    for wid, (wlat, wlon) in wp_positions.items():
                        d = great_circle_nm(lat, lon, wlat, wlon)
                        prev = cpa[wid]
                        if prev is None or d < prev["dist_nm"]:
                            cpa[wid] = dict(dist_nm=d, elapsed_s=elapsed_row_s, fl=current_fl,
                                             mach=state["mach"], tas_kt=state["tas_kt"], gs_kt=gs_kt)

                prev_on_ground = state["on_ground"]
                prev_gs_kt = gs_kt

                if phase == "done":
                    break

            if live_display is not None:
                for countdown_s in range(max(int(tick_s), 1), 0, -1):
                    live_display.update(_screen(countdown_s))
                    time.sleep(1 / replay_speed)
            else:
                time.sleep(tick_s / replay_speed)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if live_display is not None:
            live_display.stop()
        if csv_file:
            csv_file.close()

    if interrupted:
        print("\nStopped (Ctrl+C).")

    if recording and phase == "done" and compare_path:
        table, constants = compare_to_report(report_df, cpa, touchdown_elapsed_s,
                                              accel_id, decel_id)
        _print_comparison(table, constants)
