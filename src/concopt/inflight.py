"""Phase 6: live in-flight altitude advisor plus flight recorder, for
`concopt inflight`. Reads the sim over SimConnect (python-SimConnect,
localhost -- P3D and Python are the same machine), projects --lookahead-nm
ahead along the loaded route, queries Active Sky there exactly the way
verify.py does (reuses verify._as_atmosphere, never reimplements it) --
except temperature, which above the tropopause is the sim's own (see
_advisor_temp_k) -- and feeds the result to limits.best_level UNCHANGED -- same ceiling, same
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

RECORDER: elapsed-since-brake-release is SIM time, accumulated from
ZULU_TIME tick to tick (_SimClock) -- not time.monotonic(). The aircraft
moves in sim time, and P3D does not always keep up with the wall clock:
the first real recording (2026-10-21) ran at ~87% for its last 12 minutes
(730 s wall, 639 s ZULU_TIME), which wall-clock elapsed time silently
charged to the flight, as if the model had been slow. A pause stops the
clock the same way. ZULU_TIME wraps at 86400 (seconds since midnight GMT)
and a flight can span that, so each step is taken modulo a day; a tick
whose ZULU_TIME read failed falls back to wall-clock time since the last
good read. While paused, no rows are written at all (_sim_paused). ZULU_TIME
is still logged verbatim per row as the absolute reference.

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
import datetime as dt
import os
import sys
import time
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text
from SimConnect import AircraftRequests, SimConnect

from concopt import arrival, limits, runways
from concopt.asky import get_atmosphere_np
from concopt.atmos import (KT_TO_MS, fl_to_pressure, isa, pressure_to_fl,
                            speed_of_sound)
from concopt.route import (build_legs, current_progress_nm, destination_point,
                            great_circle_nm, parse_pln, project_along_route,
                            supersonic_segment)
from concopt.params import (ACTIVE_SKY_HOST, ACTIVE_SKY_PORT,  # noqa: F401
                            APPROACH_AGL_FT, ARRIVAL_REFRESH_S, ARRIVAL_WIND_SAMPLE_FL,
                            LHR_SURFACE_QUERY_FT, LHR_WIND_SAMPLE_FL,
                            RUNWAYS,
                            BRAKE_RELEASE_GS_KT, C_TO_K,
                            DEFAULT_GAIN_THRESHOLD_KT, DEFAULT_INTERVAL_S,
                            DEFAULT_LOOKAHEAD_NM, FASTEST_SCHEDULE_KT, FT_TO_M,
                            INHG_TO_HPA, ISA_H1_M, LB_TO_KG, LOW_ALT_FT, LOW_ALT_INTERVAL_S,
                            NM_TO_M, SIM_CLOCK_JUMP_S, SIMCONNECT_REQUEST_TIME_MS, S_PER_DAY)
from concopt.search import TARGET_FL, _format_hmm

# --- Flight recorder schema --------------------------------------------------
# Bumped whenever RECORD_COLUMNS changes, and written as a column on every
# row so a notebook can tell which vintage it is reading without guessing
# from the header. 1 = the original narrow schema (zulu_s .. last_waypoint).
RECORD_SCHEMA_VERSION = 3  # 3: eng1_n2..eng4_n2 (FSLabs N2 via FSUIPC)

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
    # engines: FSLabs' own N2 per engine, via FSUIPC (N1 isn't published)
    "eng1_n2", "eng2_n2", "eng3_n2", "eng4_n2",
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
    """One poll of the sim variables the advisor/recorder need. A required
    read can come back None (a dropped SimConnect read -- TOTAL_WEIGHT did,
    mid-cruise, 2026-10-23); run_inflight's _check_sim_state turns that into
    a skipped tick with a notice rather than a crash, so nothing here does
    arithmetic on a raw read. The optional block below is recorder-only
    telemetry that degrades to an empty CSV cell instead."""
    return dict(
        lat_deg=aq.get("PLANE_LATITUDE"),
        lon_deg=aq.get("PLANE_LONGITUDE"),
        alt_ft=aq.get("PLANE_ALTITUDE"),
        mach=aq.get("AIRSPEED_MACH"),
        tas_kt=aq.get("AIRSPEED_TRUE"),
        gs_kt=_get_optional(aq, "GPS_GROUND_SPEED", 1.0 / KT_TO_MS),
        weight_t=_get_optional(aq, "TOTAL_WEIGHT", LB_TO_KG / 1000.0),
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


class _SimClock:
    """Sim seconds since construction (brake release), from ZULU_TIME --
    see the module docstring's RECORDER note. elapsed_s is called once per
    tick, in order: each good ZULU_TIME read adds its step since the last
    good one (modulo a day, for the midnight wrap); a missing read
    (None/NaN -- one dropped SimConnect read in the first real recording)
    answers with wall-clock time since the last good read, scaled by
    replay_speed like every other wall-clock use in run_inflight, and
    doesn't move the reference, so the next good read covers the gap from
    ZULU_TIME alone.

    A step over SIM_CLOCK_JUMP_S is the sim's clock being set (Active Sky
    changing the time of day), not flown: it is replaced by the wall-clock
    time since the last read, like a dropped read. base_s lets a resumed
    recording continue from its last row's elapsed time."""

    def __init__(self, zulu_s, now_s, replay_speed=1.0, base_s=0.0):
        self.replay_speed = replay_speed
        self.base_s = base_s
        self.zulu_s = zulu_s if _zulu_ok(zulu_s) else None
        self.now_s = now_s

    def elapsed_s(self, zulu_s, now_s):
        if not _zulu_ok(zulu_s):
            return self.base_s + (now_s - self.now_s) * self.replay_speed
        step_s = None if self.zulu_s is None else (zulu_s - self.zulu_s) % S_PER_DAY
        if step_s is None or step_s > SIM_CLOCK_JUMP_S:
            self.base_s += (now_s - self.now_s) * self.replay_speed
        else:
            self.base_s += step_s
        self.zulu_s, self.now_s = zulu_s, now_s
        return self.base_s


def _sim_paused(last_row_zulu_s, zulu_s):
    """True when ZULU_TIME hasn't moved since the last recorded row: the
    sim is paused, and the recorder skips the row rather than writing the
    same frozen state once a tick for as long as the pause lasts (which
    would count a 5-minute pause as five more advisor ticks in the
    post-flight notebook's Q4). A missing read is not a pause -- that row
    is still written, on _SimClock's wall-clock fallback."""
    return _zulu_ok(zulu_s) and zulu_s == last_row_zulu_s


def _zulu_ok(zulu_s):
    return zulu_s is not None and not np.isnan(zulu_s)


def _parse_utc_s(text):
    """'HH:MM' or 'HH:MM:SS' (UTC, sim time) -> seconds since midnight."""
    parts = [int(p) for p in text.split(":")]
    if len(parts) not in (2, 3) or not (0 <= parts[0] < 24 and all(0 <= p < 60 for p in parts[1:])):
        raise ValueError(f"expected HH:MM or HH:MM:SS, got {text!r}")
    return parts[0] * 3600 + parts[1] * 60 + (parts[2] if len(parts) == 3 else 0)


def _late_start_elapsed_s(zulu_s, brake_release_utc_s):
    """Elapsed time to start a late-started recording at: sim time since
    brake_release_utc_s (modulo a day, for a flight across midnight), or 0
    when that isn't given or ZULU_TIME wasn't read."""
    if brake_release_utc_s is None or not _zulu_ok(zulu_s):
        return 0.0
    return float((zulu_s - brake_release_utc_s) % S_PER_DAY)


def _connect_fsuipc():
    """An FSUIPC connection for FSLabs' N2 L:vars, or None (with a note) if
    FSUIPC isn't there -- N2 is recorder/panel telemetry, not worth ending
    the run over."""
    from concopt.fsuipc_ipc import Fsuipc
    try:
        return Fsuipc()
    except (RuntimeError, OSError, AttributeError) as exc:  # AttributeError: no ctypes.WinDLL off Windows
        print(f"N2 unavailable: {exc}")
        return None


def _read_n2(fsuipc):
    """{eng1_n2..eng4_n2}: FSLabs' N2 per engine, or None each when FSUIPC
    isn't connected or a read fails (an empty CSV cell, never 0)."""
    values = (None,) * 4
    if fsuipc is not None:
        try:
            values = fsuipc.read_n2()
        except RuntimeError:
            pass
    return {f"eng{i}_n2": v for i, v in zip((1, 2, 3, 4), values)}


def _n2_text(state):
    """The panel's N2 line ("" when there's no N2), with the key to log
    observations."""
    n2 = [state.get(f"eng{i}_n2") for i in (1, 2, 3, 4)]
    if all(v is None for v in n2):
        return ""
    shown = "  ".join("--" if v is None else f"{v:.0f}" for v in n2)
    return f"\nN2 % (eng 1-4): {shown}   [press o to log observed N1/N2]"


# The observation log -- N1/N2 read off the gauges by hand (FSLabs publishes
# no N1), next to FSLabs' own N2 and the flight state, for an N1-from-N2 model.
OBS_COLUMNS = ["zulu_s", "elapsed_s", "engine", "n1_obs", "n2_obs",
               "eng1_n2", "eng2_n2", "eng3_n2", "eng4_n2",
               "alt_ft", "mach", "cas_kt", "tas_kt", "sim_temp_c", "on_ground"]


def _parse_observation(text):
    """"N1 N2" or "ENGINE N1 N2" (e.g. "53 74" / "2 53 74") -> dict(engine,
    n1_obs, n2_obs); engine None when not given. Raises ValueError."""
    parts = text.replace(",", " ").split()
    if len(parts) not in (2, 3):
        raise ValueError("enter N1 N2, or ENGINE N1 N2")
    engine = int(parts[0]) if len(parts) == 3 else None
    n1, n2 = (float(p) for p in parts[-2:])
    if engine is not None and engine not in (1, 2, 3, 4):
        raise ValueError("engine must be 1-4")
    if not (0.0 <= n1 <= 120.0 and 0.0 <= n2 <= 120.0):
        raise ValueError("N1 and N2 are percent, 0-120")
    return dict(engine=engine, n1_obs=n1, n2_obs=n2)


def _observation_row(obs, state, elapsed_s):
    """One OBS_COLUMNS row: what was typed, FSLabs' N2 at the same moment,
    and the flight state (the N1/N2 relationship may move with altitude and
    Mach)."""
    row = {**obs, "zulu_s": state.get("zulu_s"), "elapsed_s": elapsed_s,
           "on_ground": int(bool(state.get("on_ground")))}
    for col in ("eng1_n2", "eng2_n2", "eng3_n2", "eng4_n2", "alt_ft", "mach", "cas_kt",
                "tas_kt", "sim_temp_c"):
        row[col] = state.get(col)
    return row


def _append_observation(path, row):
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(OBS_COLUMNS)
        writer.writerow([row.get(c) for c in OBS_COLUMNS])


def _wait_for_key(seconds, enabled):
    """Sleep `seconds`, or return True early when "o" is pressed (Windows
    console only -- msvcrt)."""
    if not enabled:
        time.sleep(seconds)
        return False
    import msvcrt
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        while msvcrt.kbhit():
            if msvcrt.getwch().lower() == "o":
                return True
        time.sleep(0.05)
    return False


def _take_observation(path, state, elapsed_s, live_display):
    """Prompt for observed N1/N2 (pausing the live panel), append it to
    `path`, and return a note for the panel. Enter alone cancels."""
    if live_display is not None:
        live_display.stop()
    try:
        while True:
            text = input("\nObserved N1 N2 (e.g. '53 74', or '2 53 74' for engine 2; "
                         "Enter to cancel): ").strip()
            if not text:
                return "Observation cancelled"
            try:
                obs = _parse_observation(text)
                break
            except ValueError as exc:
                print(f"  {exc}")
        _append_observation(path, _observation_row(obs, state, elapsed_s))
        eng = f" (engine {obs['engine']})" if obs["engine"] else ""
        return f"Logged N1 {obs['n1_obs']:g} / N2 {obs['n2_obs']:g}{eng} to {path}"
    finally:
        if live_display is not None:
            live_display.start()


class _BadSimData(ValueError):
    """The sim answered, but with a position/state that can't be flown on
    (NaN while P3D reloads scenery or Active Sky changes the time)."""


def _check_sim_state(state):
    """Raise _BadSimData if the required reads aren't finite numbers."""
    for key in ("lat_deg", "lon_deg", "alt_ft", "mach", "tas_kt", "gs_kt", "weight_t"):
        value = state[key]
        if value is None or not np.isfinite(value):
            raise _BadSimData(f"the sim returned {value!r} for {key}")


def _update_cpa(cpa, wp_positions, lat, lon, elapsed_s, fl, mach, tas_kt, gs_kt):
    """Fold one position into each waypoint's closest point of approach."""
    for wid, (wlat, wlon) in wp_positions.items():
        d = great_circle_nm(lat, lon, wlat, wlon)
        prev = cpa[wid]
        if prev is None or d < prev["dist_nm"]:
            cpa[wid] = dict(dist_nm=d, elapsed_s=elapsed_s, fl=fl,
                             mach=mach, tas_kt=tas_kt, gs_kt=gs_kt)


def _prepare_recording(record_path, mode, input_fn=input):
    """What to do about a recording that already exists at record_path:
    append to it and resume, or start a new one. mode is "ask" (prompt),
    "append" or "new". Returns None for a fresh recording (no file, an empty
    one, or "new" -- the old file is renamed to a .bak copy first, never
    overwritten), or a dict of the existing rows to resume from."""
    path = Path(record_path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    existing = pd.read_csv(path)
    if existing.empty:
        return None

    elapsed = existing["elapsed_s"].dropna()
    last_elapsed_s = float(elapsed.iloc[-1]) if len(elapsed) else 0.0
    print(f"{path} already exists: {len(existing)} rows, last at "
          f"{_format_hmm(last_elapsed_s)} elapsed ({existing['phase'].iloc[-1]}, "
          f"after {existing['last_waypoint'].iloc[-1]}).")

    if mode == "ask":
        try:
            while mode == "ask":
                answer = input_fn("[A]ppend to it and resume, or start a [N]ew recording "
                                  "(the old file is kept as a .bak copy)? ").strip().lower()
                mode = {"a": "append", "append": "append", "n": "new", "new": "new"}.get(answer, "ask")
        except EOFError:
            raise SystemExit("A recording already exists and there is no terminal to ask "
                             "about it -- pass --record-mode append or --record-mode new.")

    if mode == "new":
        stamp = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")
        os.replace(path, backup)
        print(f"Moved the existing recording to {backup}; starting a new one.")
        return None

    if list(existing.columns) != RECORD_COLUMNS:
        raise SystemExit(f"{path} has different columns from this version's recording "
                         f"(schema {RECORD_SCHEMA_VERSION}), so it can't be appended to -- "
                         "use --record-mode new.")
    last = existing.iloc[-1]
    return dict(rows=existing, elapsed_s=last_elapsed_s,
                zulu_s=None if pd.isna(last["zulu_s"]) else float(last["zulu_s"]),
                on_ground=bool(last["on_ground"]), gs_kt=last["gs_kt"])


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


def _advisor_temp_k(as_temp_k, sim_temp_c, alt_ft):
    """The FL450-FL600 temperature the advisor's level table runs on: the
    sim's own AMBIENT_TEMPERATURE at the aircraft, held across the whole
    grid, once the aircraft is above the ISA tropopause (ISA_H1_M);
    Active Sky's grid (as_temp_k) below that, or when the sim read failed.

    Active Sky's GetAtmosphere temperature is not what the sim flies in.
    The first real recording (2026-10-21) had the two 4-5 C apart in
    cruise, with Active Sky's jumping 5-17 C from one tick to the next
    (an 11 C step between FL523 and FL528) while the sim's stayed smooth;
    the sim's own Mach and TAS agreed with its own temperature to within
    2.5 kt and missed Active Sky's by up to 17 kt. Winds and pressure
    agreed to a fraction of a knot/hPa, so only temperature is swapped.
    Holding it constant across the grid is ISA's isothermal layer
    (11-20 km) -- the same constant ISA deviation at every level; the sim
    actually cooled ~0.5-0.8 C per 1,000 ft up there, a few degrees off at
    the far end of the grid, against Active Sky's 5-17 C."""
    if sim_temp_c is None or alt_ft * FT_TO_M < ISA_H1_M:
        return as_temp_k
    return np.full_like(as_temp_k, sim_temp_c + C_TO_K)


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


def _flight_phase(cum_nm, accel_cum_nm, decel_cum_nm, mach, agl_ft, on_ground,
                  decel_start_cum_nm=None, prev_phase=None):
    """climb / cruise / decel / descent / approach / ground for one sample,
    so a notebook can cut segments without re-inferring them from the
    altitude trace.

    Cut from the ROUTE position where possible (the accel waypoint and the
    decel start are exactly where search.py's own segment boundaries are,
    so the recording's segments line up with the model's). The decel waypoint
    is where the aircraft must be subsonic BY, so "decel" runs from
    decel_start_cum_nm (the live arrival model's own decel distance before
    it; the waypoint itself when unknown) up to the waypoint -- and past it
    only while still supersonic, an overrun. approach needs agl_ft; where
    that read was unavailable the sample stays "descent" rather than
    guessing -- the empty agl_ft column is what says so.

    Once "approach", it stays "approach" until the ground (prev_phase, the
    previous row's phase): levelling off around APPROACH_AGL_FT otherwise
    flips it approach/descent row to row, as the 2026-10-23 recording did
    for 2 minutes at ~1,500 ft. A go-around stays "approach" too -- rare
    enough not to model."""
    if on_ground:
        return "ground"
    if prev_phase == "approach":
        return "approach"
    if decel_start_cum_nm is None:
        decel_start_cum_nm = decel_cum_nm
    if cum_nm >= decel_cum_nm:
        if agl_ft is not None and agl_ft < APPROACH_AGL_FT:
            return "approach"
        return "decel" if (mach is not None and mach >= 1.0) else "descent"
    if cum_nm >= decel_start_cum_nm:
        return "decel"
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


def _build_live_arrival_wind_fn(weather_source, lat, lon, track_deg,
                                levels_fl=ARRIVAL_WIND_SAMPLE_FL):
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
            lat, lon, levels_fl * 100.0)
    except (RuntimeError, TypeError, KeyError, ValueError):
        return None
    if (len(alt_ft) != len(levels_fl)
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


def _live_lhr_runway(weather_source, lat, lon):
    """EGLL's runway from Active Sky's surface wind there, through the same
    screen search/report apply to ERA5 (runways.select_runway: 09 whenever
    its tailwind <= TAILWIND_MAX_KT and crosswind passes, else 27). Active
    Sky gives no gust, so the mean wind stands in for it. None if Active Sky
    doesn't answer."""
    point = _as_point_atmosphere(weather_source, lat, lon, LHR_SURFACE_QUERY_FT)
    if point is None:
        return None
    speed_ms = point["wind_kt"] * KT_TO_MS
    dir_rad = np.radians(point["wind_dir_deg"])
    u_ms, v_ms = np.array([-speed_ms * np.sin(dir_rad)]), np.array([-speed_ms * np.cos(dir_rad)])
    screen = runways.select_runway(u_ms, v_ms, np.array([speed_ms]), "EGLL")
    return dict(runway=str(screen["runway"][0]), headwind_kt=float(screen["headwind_kt"][0]),
                xwind_kt=float(screen["xwind_gust_kt"][0]), flag=str(screen["flag"][0]),
                wind_dir_deg=point["wind_dir_deg"], wind_kt=point["wind_kt"])


def _live_arrival(cruise_fl, arrival_nm, isa_dev_c, mass_at_barix_t, wind_at_fl, gs_kt,
                  thrust_reversers=False, decel_wind_at_fl=None, descent_wind_at_fl=None):
    """arrival.arrival() at the CURRENT state -- n_cand is 1 throughout, so
    every input is a scalar and the (1,)-shaped output arrays are squeezed
    back to floats. The decel ends at the decel waypoint, so arrival()
    backs the cruise it replaces out of the time to get there; with no march
    to read that from, it's the current ground speed (gs_kt) at the current
    weight -- what the time-to-waypoint estimate it is subtracted from
    assumes too. speed=380 matches arrival()'s own default (the call
    signature this was speced against); real pre-flight callers
    (fuel._arrival_from_march) use speed="auto" instead, to dodge 380 kt's
    NaN-fuel gap at low decel_end_fl mass -- but that gap is in
    level_fuel_t, not level_time_min, and only time feeds the live panel's
    predicted-remaining clock, so it doesn't matter here."""
    def march_tail(back_nm):
        back_nm = np.asarray(back_nm, dtype=float)
        time_s = back_nm / gs_kt * 3600.0 if gs_kt > 0 else np.full_like(back_nm, np.nan)
        return time_s, np.full_like(back_nm, mass_at_barix_t), np.zeros(back_nm.shape, bool)

    out = arrival.arrival(cruise_fl, arrival_nm, wind_at_fl, isa_dev_c,
                           mass_at_barix_t, speed=FASTEST_SCHEDULE_KT, march_tail=march_tail,
                           thrust_reversers=thrust_reversers, decel_wind_at_fl=decel_wind_at_fl,
                           descent_wind_at_fl=descent_wind_at_fl)
    return dict(
        time_min=float(out["time_min"][0]),
        decel_nm=float(out["decel_nm"][0]),
        level_nm=float(out["level_nm"][0]),
        descent_nm=float(out["descent_nm"][0]),
        decel_time_min=float(out["decel_time_min"][0]),
        transition_time_min=float(out["transition_time_min"][0]),
        level_time_min=float(out["level_time_min"][0]),
        descent_time_min=float(out["descent_time_min"][0]),
        approach_time_min=float(out["approach_time_min"][0]),
        descent_top_kias=float(out["descent_top_kias"][0]),
        reverser_min=float(out["reverser_min"][0]),
        level_fl=float(out["level_fl"][0]),
        level_wind_kt=float(out["level_wind_kt"][0]),
        flags=str(out["flags"][0]),
        source="live",
    )


def _build_arrival_text(arrival_info):
    """The Arrival (to touchdown) breakdown for the live panel -- at
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
        return Text("Arrival (to touchdown): not yet available")
    if arrival_info.get("source") == "fallback":
        return Text(
            "Arrival (to touchdown) [PRE-FLIGHT]: Active Sky unavailable "
            f"this far ahead -- pre-flight figure {arrival_info['time_min']:.1f} min total"
        )
    a = arrival_info
    reversers = f", reversers {a['reverser_min']:.1f} min" if a["reverser_min"] > 0.0 else ""
    return Text(
        f"Arrival (to touchdown) [LIVE], {a['time_min']:.1f} min total:\n"
        f"  decel    {a['decel_time_min']:4.1f} min {a['decel_nm']:4.0f} nm   "
        f"to M1.0, subsonic by the decel waypoint\n"
        f"  level    {a['level_time_min']:4.1f} min {a['level_nm']:4.0f} nm   "
        f"FL{a['level_fl']:.0f} at M{arrival.LEVEL_MACH:.2f}, {a['level_wind_kt']:+.0f} kt\n"
        f"  descent  {a['descent_time_min']:4.1f} min {a['descent_nm']:4.0f} nm   "
        f"M{arrival.LEVEL_MACH:.2f}/{a['descent_top_kias']:.0f} KIAS -> "
        f"{arrival.SPEED_LIMIT_KIAS:.0f} KIAS at FL{arrival.SPEED_LIMIT_FL:.0f} -> "
        f"{arrival.GATE_KIAS:.0f} KIAS at 1,500 ft{reversers}\n"
        f"  approach {a['approach_time_min']:4.1f} min {arrival.APPROACH_NM:4.0f} nm   "
        f"at {arrival.GATE_KIAS:.0f} KIAS"
        + _lhr_runway_text(a.get("lhr"), a.get("extra_nm", 0.0))
    )


def _lhr_runway_text(lhr, extra_nm):
    """The arrival panel's runway line: which EGLL runway Active Sky's
    surface wind allows, and why -- 09 is the straight-in from the west, 27
    flies past to a final from the east (extra_nm longer than the .pln)."""
    if lhr is None:
        return "\n  LHR runway: Active Sky surface wind unavailable -- arrival as the .pln flies it"
    wind = f"{lhr['wind_dir_deg']:03.0f}/{lhr['wind_kt']:.0f} kt"
    head = (f"headwind {lhr['headwind_kt']:.0f} kt" if lhr["headwind_kt"] >= 0
            else f"tailwind {-lhr['headwind_kt']:.0f} kt")
    if lhr["runway"].startswith("09"):
        why = f"09 usable ({head}, crosswind {lhr['xwind_kt']:.0f} kt)"
    else:
        why = (f"09 not usable -- {lhr['runway']} ({head}, crosswind {lhr['xwind_kt']:.0f} kt), "
               f"final from the east, {extra_nm:+.0f} nm")
    unflyable = "  !! no runway passes the crosswind/tailwind screen" if lhr["flag"] == "unflyable" else ""
    return f"\n  LHR {wind}: {why}{unflyable}"


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
    than a divide-by-zero.

    remaining_nm <= 0 means the aircraft is past the decel start: the
    supersonic cruise the FL450-FL600 table is about is over, so there is
    no advice -- action "NONE", rec_fl/gain_kt/gain_s None. (The first
    full recording, 2026-10-23, said "CLIMB to FL530" all the way down to
    the runway.)"""
    current_fl = float(table["fl"].iloc[current_idx])
    if remaining_nm <= 0.0:
        return dict(action="NONE", current_fl=current_fl, rec_fl=None, gain_kt=None, gain_s=None)
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

    if adv["action"] == "NONE":
        return "No level advice -- past decel start"
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
    style = "dim" if line.startswith(("HOLD", "No level advice")) else "bold white on dark_green"
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
                    recorder_note=None, plan_vs_actual=None):
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
        + _n2_text(state)
    )

    level_table = _build_level_table(table, best_idx, current_idx)

    arrival_text = _build_arrival_text(arrival_info)

    progress_text = Text(
        f"Distance run {distance_run_nm:,.0f} nm / to decel start {remaining_nm:,.0f} nm   "
        f"Elapsed {_format_hmm_or_na(elapsed_s)}   "
        f"Remaining {_format_hmm_or_na(predicted_remaining_s)}\n"
        f"Predicted total: live {_format_hmm_or_na(predicted_total_s)}   "
        f"vs pre-flight {_format_hmm_or_na(preflight_predicted_total_s)}\n"
        + _plan_vs_actual_text(plan_vs_actual, distance_run_nm, elapsed_s)
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
        **{f"eng{i}_n2": _round_or_none(state.get(f"eng{i}_n2"), 1) for i in (1, 2, 3, 4)},
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
        "binding": binding_at_best if advisory["action"] != "NONE" else None,
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


def _plan_track(report_df, touchdown_cum_nm):
    """The pre-flight plan as (cum_nm, plan_s) points, brake release ->
    touchdown: each report waypoint at the running sum of its leg_time
    (m:ss -- to the second, where the elapsed column is whole minutes),
    then touchdown at the decel waypoint's time plus the report's own
    arrival_s, straight-line across the arrival (the report has no
    waypoints after the decel one)."""
    def _mmss_s(text):
        m, sec = str(text).split(":")
        return int(m) * 60.0 + int(sec)

    leg_s = report_df["leg_time"].map(lambda v: _mmss_s(v) if pd.notna(v) and str(v) else 0.0)
    cum_nm = np.append(report_df["cum_nm"].to_numpy(float), touchdown_cum_nm)
    plan_s = np.cumsum(leg_s.to_numpy(float))
    plan_s = np.append(plan_s, plan_s[-1] + _report_arrival_s(report_df))
    return cum_nm, plan_s


def _plan_vs_actual(plan_track, cum_nm, elapsed_s):
    """Actual against the plan at the current position, both ways round:
    behind_s, how much later than planned the aircraft got HERE (positive =
    behind), and behind_nm, how far short of where the plan had it by NOW
    (positive = behind). None before brake release or with no plan."""
    if plan_track is None or elapsed_s is None:
        return None
    track_nm, track_s = plan_track
    plan_s_here = float(np.interp(cum_nm, track_nm, track_s))
    plan_nm_now = float(np.interp(elapsed_s, track_s, track_nm))
    return dict(plan_s_here=plan_s_here, behind_s=elapsed_s - plan_s_here,
                plan_nm_now=plan_nm_now, behind_nm=plan_nm_now - cum_nm)


def _plan_vs_actual_text(pva, cum_nm, elapsed_s):
    """One line for the panel, e.g. "vs plan: 2:53 (min:sec) behind here (plan 1:07 at
    1,302 nm) / plan is at 1,350 nm by now, 48 nm ahead of you"."""
    if pva is None:
        return "vs plan: needs --compare and a recorded brake release"

    def _mmss(seconds):
        total = int(round(abs(seconds)))
        return f"{total // 60}:{total % 60:02d}"

    time_word = "behind" if pva["behind_s"] >= 0 else "ahead"
    nm_word = "behind" if pva["behind_nm"] >= 0 else "ahead"
    return (f"vs plan: {_mmss(pva['behind_s'])} (min:sec) {time_word} here (plan "
            f"{_format_hmm(pva['plan_s_here'])} at {cum_nm:,.0f} nm, actual "
            f"{_format_hmm(elapsed_s)})  /  {abs(pva['behind_nm']):,.0f} nm {nm_word} "
            f"(plan at {pva['plan_nm_now']:,.0f} nm by now)")


def compare_to_report(report_df, cpa, touchdown_elapsed_s, accel_id, decel_id):
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
        # The decel waypoint is where the aircraft must be subsonic BY.
        mach_at_decel_waypoint=(cpa[decel_id]["mach"]
                                if cpa.get(decel_id) is not None else np.nan),
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
    mach = constants.get("mach_at_decel_waypoint", np.nan)
    if not np.isnan(mach):
        verdict = "subsonic" if mach < 1.0 else "STILL SUPERSONIC"
        print(f"  at the decel waypoint: M{mach:.2f} -- {verdict}")


def run_inflight(pln_path, accel_id, decel_id, interval_s=DEFAULT_INTERVAL_S, lookahead_nm=DEFAULT_LOOKAHEAD_NM,
                  record_path=None, compare_path=None,
                  host=ACTIVE_SKY_HOST, port=ACTIVE_SKY_PORT, cruise_mach=limits.CRUISE_MACH,
                  gain_threshold_kt=DEFAULT_GAIN_THRESHOLD_KT, simconnect_dll=None, live=True,
                  state_source=None, weather_source=None, replay_speed=1.0,
                  thrust_reversers=False, record_mode="ask", brake_release_utc_s=None,
                  obs_log_path=None):
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
    replay can run faster than real time) and _SimClock's wall-clock
    fallback for a missing ZULU_TIME is multiplied back up by it. Elapsed
    time itself is sim time from each state's zulu_s (see the module
    docstring's RECORDER note), which a replay serves from the recorded
    rows, so it is flight-equivalent seconds without any scaling. This is
    a plain scalar, not a third injectable source -- the two callables
    above are the whole seam.

    Covered by tests/test_replay.py via the seam above (a live SimConnect +
    Active Sky are never touched there); see compare_to_report and
    _level_table/_recommendation_line/_render_screen for the other pure
    pieces.

    record_mode is what to do when record_path already holds a recording:
    "ask" (prompt), "append" (resume it -- phase, elapsed time and each
    waypoint's closest approach are rebuilt from its rows, and new rows go on
    the end) or "new" (the old file is renamed to a .bak copy). A tick the
    sim or Active Sky can't answer (NaN position, Active Sky not loaded with
    a date) shows a notice and retries rather than ending the run.

    brake_release_utc_s (seconds since midnight UTC, sim time; --brake-
    release-utc) is for a recorder started after the takeoff roll: the first
    reading is then already past BRAKE_RELEASE_GS_KT, and without it elapsed
    time would count from that first reading instead -- the 2026-10-23
    recording started 16.4 min late and its elapsed_s ran 16.4 min short
    all flight. Given, elapsed starts at the time since then; not given, a
    late start is warned about on the panel. Ignored when brake release is
    actually seen (a ground-speed crossing)."""
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
    # Where the live arrival reads its weather: the decel at the decel
    # waypoint itself (it ends there), the descent and approach at EGLL, and
    # the runway from EGLL's surface wind -- 27 flies past to a final from
    # the east, runways.arrival_nm_by_runway.
    decel_leg = ss_legs[-1]
    decel_lat, decel_lon = destination_point(decel_leg.lat_mid, decel_leg.lon_mid,
                                             decel_leg.track_deg, decel_leg.dist_nm / 2.0)
    lhr_lat, lhr_lon = destination_point(legs[-1].lat_mid, legs[-1].lon_mid,
                                         legs[-1].track_deg, legs[-1].dist_nm / 2.0)
    arrival_nm_by_runway = runways.arrival_nm_by_runway(legs, arrival_nm)
    runway_true_deg = {r["name"]: r["true_deg"] for r in RUNWAYS["EGLL"]}

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
        plan_track = _plan_track(report_df, touchdown_cum_nm)
    else:
        plan_track = None

    # Asked before connecting, so the answer isn't buried under SimConnect's
    # output (and a typo'd path doesn't cost a connection first).
    resume = _prepare_recording(record_path, record_mode) if record_path is not None else None

    # THE SEAM: state_source/weather_source given (replay) -> no SimConnect
    # connection is attempted at all, live or otherwise. Neither given (a
    # real flight) -> connect for real and build the live defaults, exactly
    # the calls this replaced.
    if state_source is None:
        sm, aq = _connect(simconnect_dll)
        fsuipc = _connect_fsuipc()
        state_source = lambda: {**_read_state(aq), **_read_n2(fsuipc)}  # noqa: E731
    if weather_source is None:
        weather_source = _live_weather_source(host, port)

    # Press "o" to log observed N1/N2 (only with a real keyboard, not under a
    # replay piped from tests): to obs_log_path, next to the recording by default.
    observe_keys = sys.stdin.isatty() and os.name == "nt"
    if obs_log_path is None:
        obs_log_path = (Path(record_path).with_name(Path(record_path).stem + "_n1n2.csv")
                        if record_path is not None else Path("n1n2_observations.csv"))

    recording = record_path is not None
    csv_file = None
    cpa = {}
    wp_positions = {}
    waypoint_cum_nm = {}
    if recording:
        wp_positions = {wid: (lat, lon) for wid, lat, lon in plan["waypoints"]}
        cpa = {wid: None for wid in wp_positions}
        waypoint_cum_nm = _waypoint_cum_nm(legs)
        csv_file = open(record_path, "a" if resume else "w", newline="")
        csv_writer = csv.writer(csv_file)
        if not resume:
            csv_writer.writerow(RECORD_COLUMNS)

    phase = "ground"
    prev_on_ground = None
    prev_gs_kt = None
    sim_clock = None
    last_row_zulu_s = None  # the ZULU_TIME of the last row written -- see _sim_paused
    last_flight_phase = None  # the last row's phase -- see _flight_phase's prev_phase
    touchdown_elapsed_s = None
    recorder_note = None  # live panel only -- --no-live prints these as before
    if resume:
        # Carry on as if never stopped: already recording, the clock picks up
        # at the last row's elapsed time (on the first tick, below), and the
        # waypoint approaches seen so far are rebuilt from the rows.
        phase = "recording"
        prev_on_ground = resume["on_ground"]
        prev_gs_kt = resume["gs_kt"]
        last_row_zulu_s = resume["zulu_s"]
        last_flight_phase = resume["rows"]["phase"].iloc[-1]
        for r in resume["rows"].itertuples():
            if pd.notna(r.lat_deg) and pd.notna(r.lon_deg):
                _update_cpa(cpa, wp_positions, r.lat_deg, r.lon_deg, r.elapsed_s,
                            r.level_fl, r.mach, r.tas_kt, r.gs_kt)
        recorder_note = (f"Resumed recording to {record_path} after {len(resume['rows'])} rows "
                         f"({_format_hmm(resume['elapsed_s'])} elapsed)")
        if not live:
            print(f"\n{recorder_note}")
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
            try:
                state = state_source()
                _check_sim_state(state)
                lat, lon = state["lat_deg"], state["lon_deg"]

                la_lat, la_lon, track_deg = project_along_route(legs, lat, lon, lookahead_nm)
                current_cum_nm, leg_idx, along_nm = current_progress_nm(legs, lat, lon)
                to_decel_wp_nm = max(route_end_cum_nm - current_cum_nm, 0.0)  # to the decel waypoint
                total_remaining_nm = max(touchdown_cum_nm - current_cum_nm, 0.0)  # to touchdown
                next_wp_id = legs[leg_idx].to_id
                dist_to_next_nm = max(legs[leg_idx].dist_nm - along_nm, 0.0)

                temp_k, u_ms, v_ms = _lookahead_atmosphere(weather_source, la_lat, la_lon)
                temp_k = _advisor_temp_k(temp_k, state.get("sim_temp_c"), state["alt_ft"])
                current_fl = state["alt_ft"] / 100.0
                table, best_idx, current_idx, binding_at_best = _level_table(
                    temp_k, u_ms, v_ms, track_deg, state["weight_t"], cruise_mach, current_fl)
            except (_BadSimData, RuntimeError) as exc:
                # Active Sky errors out when its date/time is being changed or
                # isn't loaded, the sim hands back NaN while it reloads: none
                # of that is the end of the flight. No row is written for a
                # tick like this; elapsed time is sim time, so nothing is lost.
                notice = (f"No usable data this tick ({str(exc)[:160]}). Check the sim and "
                          f"Active Sky date/time -- retrying in {interval_s:.0f} s "
                          f"(Ctrl+C to stop).")
                if live_display is not None:
                    live_display.update(Text(notice, style="bold yellow"))
                else:
                    print(f"\n{notice}")
                time.sleep(interval_s / replay_speed)
                continue

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
                lhr = _live_lhr_runway(weather_source, lhr_lat, lhr_lon)
                live_arrival_nm = arrival_nm_by_runway[lhr["runway"]] if lhr else arrival_nm
                if arrival_wind_fn is not None:
                    decel_wind_fn = _build_live_arrival_wind_fn(
                        weather_source, decel_lat, decel_lon, decel_leg.track_deg)
                    descent_wind_fn = _build_live_arrival_wind_fn(
                        weather_source, lhr_lat, lhr_lon,
                        runway_true_deg[lhr["runway"]] if lhr else legs[-1].track_deg,
                        LHR_WIND_SAMPLE_FL)
                    isa_dev_at_cruise = float(table["isa_dev_c"].iloc[current_idx])
                    # Projected mass at the decel waypoint = current weight minus
                    # the cruise burn still to come; no burn-off schedule is
                    # tracked here (see module docstring), so current weight is
                    # used as-is -- a few tonnes heavy vs the real BARIX mass,
                    # which understates specific range slightly on the subsonic
                    # table lookup.
                    arrival_info = _live_arrival(current_fl, live_arrival_nm, isa_dev_at_cruise,
                                                  state["weight_t"], arrival_wind_fn,
                                                  float(table["gs_kt"].iloc[current_idx]),
                                                  thrust_reversers, decel_wind_fn, descent_wind_fn)
                    arrival_info["lhr"] = lhr
                    arrival_info["extra_nm"] = live_arrival_nm - arrival_nm
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

            # Sim seconds since brake release, once per tick -- see
            # _SimClock and the module docstring's RECORDER note.
            # The decel ENDS at the decel waypoint, so supersonic cruise -- and
            # the advisor's "remaining" distance its level gains are worth
            # anything over -- stops decel_nm short of it: the live arrival
            # model's own decel distance from here, once it has one.
            decel_nm = (arrival_info.get("decel_nm", 0.0) if arrival_info is not None else 0.0)
            decel_start_cum_nm = decel_cum_nm - decel_nm
            remaining_nm = max(decel_start_cum_nm - current_cum_nm, 0.0)

            if sim_clock is None and resume:
                # Pick up from the last recorded row; the step since its
                # ZULU_TIME is counted unless it's a clock reset (see _SimClock).
                sim_clock = _SimClock(resume["zulu_s"], time.monotonic(), replay_speed,
                                       base_s=resume["elapsed_s"])
            elapsed_s = (sim_clock.elapsed_s(state["zulu_s"], time.monotonic())
                         if sim_clock is not None else None)
            current_gs_kt = float(table["gs_kt"].iloc[current_idx])
            if current_cum_nm < decel_cum_nm:
                # Still cruising: time to the decel waypoint at the current
                # ground speed, plus the arrival estimate above (a fixed
                # BARIX -> touchdown total, not a remaining-distance one --
                # arrival()'s decel/level/descent split only makes sense
                # starting from the decel waypoint itself).
                time_to_decel_s = (to_decel_wp_nm * NM_TO_M / (current_gs_kt * KT_TO_MS)
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
                    # + the 27 final's extra distance, which the .pln's legs don't have
                    (total_remaining_nm + (arrival_info or {}).get("extra_nm", 0.0))
                    * NM_TO_M / (current_gs_kt * KT_TO_MS)
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
                    preflight_predicted_total_s, countdown_s, arrival_info, recorder_note,
                    _plan_vs_actual(plan_track, current_cum_nm, elapsed_s))

            if live_display is not None:
                live_display.update(_screen(tick_s))
            else:
                _print_advisor(state, la_lat, la_lon, track_deg, table, best_idx, current_idx,
                                binding_at_best, remaining_nm, gain_threshold_kt)
                if _n2_text(state):
                    print(_n2_text(state).strip())
                if plan_track is not None and elapsed_s is not None:
                    print(_plan_vs_actual_text(_plan_vs_actual(plan_track, current_cum_nm, elapsed_s),
                                               current_cum_nm, elapsed_s))

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
                    started_late = prev_gs_kt is None
                    elapsed_s = (_late_start_elapsed_s(state["zulu_s"], brake_release_utc_s)
                                 if started_late else 0.0)
                    sim_clock = _SimClock(state["zulu_s"], time.monotonic(), replay_speed,
                                          base_s=elapsed_s)
                    # Already rolling but still on the ground: seconds late, not worth a warning.
                    if not started_late or (brake_release_utc_s is None and state["on_ground"]):
                        recorder_note = f"Brake release detected -- recording to {record_path}"
                    elif brake_release_utc_s is not None:
                        recorder_note = (f"Started after brake release -- elapsed counted from "
                                         f"--brake-release-utc ({_format_hmm(elapsed_s)} so far), "
                                         f"recording to {record_path}")
                    else:
                        recorder_note = ("WARNING: started after brake release -- elapsed time counts "
                                         "from now, not brake release. Restart with --record-mode "
                                         "append --brake-release-utc HH:MM to fix it. Recording to "
                                         f"{record_path}")
                    if live_display is None:
                        print(f"\n{recorder_note}")
                elif phase == "recording" and prev_on_ground is False and state["on_ground"]:
                    phase = "done"
                    touchdown_elapsed_s = elapsed_s
                    recorder_note = (f"Touchdown detected, elapsed "
                                      f"{_format_hmm(touchdown_elapsed_s)} since brake release")
                    if live_display is None:
                        print(f"\n{recorder_note}")

                if phase in ("recording", "done") and not _sim_paused(last_row_zulu_s, state["zulu_s"]):
                    last_row_zulu_s = state["zulu_s"]
                    elapsed_row_s = elapsed_s
                    last_wp = _last_waypoint_passed(waypoint_cum_nm, current_cum_nm)
                    # Active Sky AT THE AIRCRAFT -- a different query from the
                    # lookahead grid above, and the only one comparable to
                    # what the sim reports the aircraft is actually flying in.
                    as_point = _as_point_atmosphere(weather_source, lat, lon, state["alt_ft"])
                    row = _record_row(
                        elapsed_row_s, state, current_cum_nm, last_wp,
                        legs[leg_idx].track_deg,
                        _flight_phase(current_cum_nm, accel_cum_nm, decel_cum_nm,
                                       state["mach"], state.get("agl_ft"), state["on_ground"],
                                       decel_start_cum_nm, last_flight_phase),
                        as_point,
                        _advisory(table, best_idx, current_idx, remaining_nm, gain_threshold_kt),
                        binding_at_best, arrival_info,
                        predicted_remaining_s, predicted_total_s)
                    csv_writer.writerow(_row_values(row))
                    csv_file.flush()
                    last_flight_phase = row["phase"]
                    _update_cpa(cpa, wp_positions, lat, lon, elapsed_row_s, current_fl,
                                state["mach"], state["tas_kt"], gs_kt)

                prev_on_ground = state["on_ground"]
                prev_gs_kt = gs_kt

                if phase == "done":
                    break

            if live_display is not None:
                for countdown_s in range(max(int(tick_s), 1), 0, -1):
                    live_display.update(_screen(countdown_s))
                    if _wait_for_key(1 / replay_speed, observe_keys):
                        recorder_note = (_take_observation(obs_log_path, state, elapsed_s,
                                                           live_display) or recorder_note)
                        live_display.update(_screen(countdown_s))
            else:
                if _wait_for_key(tick_s / replay_speed, observe_keys):
                    recorder_note = (_take_observation(obs_log_path, state, elapsed_s, None)
                                     or recorder_note)
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
