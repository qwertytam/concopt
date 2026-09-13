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

from concopt import limits
from concopt.atmos import KT_TO_MS, isa, speed_of_sound
from concopt.route import (build_legs, current_progress_nm,
                            great_circle_nm, parse_pln, project_along_route,
                            supersonic_segment)
from concopt.search import DECEL_DESCENT_S, NM_TO_M, TARGET_FL, _format_hmm
from concopt.verify import _as_atmosphere

DEFAULT_INTERVAL_S = 60.0
DEFAULT_LOOKAHEAD_NM = 100.0
DEFAULT_GAIN_THRESHOLD_KT = 3.0

# On-ground ground speed the take-off roll is judged to have started at --
# "SIM ON GROUND true -> ground speed rising through ~40 kt", per spec.
BRAKE_RELEASE_GS_KT = 40.0

_LB_TO_KG = 0.45359237


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
    aq = AircraftRequests(sm, _time=200)
    print("Connected.")
    return sm, aq


def _read_state(aq):
    """One poll of the sim variables the advisor/recorder need. No
    reconnect/None-handling here -- if Prepar3D stops responding mid-flight
    that should surface as a plain crash, not be swallowed."""
    return dict(
        lat_deg=aq.get("PLANE_LATITUDE"),
        lon_deg=aq.get("PLANE_LONGITUDE"),
        alt_ft=aq.get("PLANE_ALTITUDE"),
        mach=aq.get("AIRSPEED_MACH"),
        tas_kt=aq.get("AIRSPEED_TRUE"),
        gs_kt=aq.get("GPS_GROUND_SPEED") / KT_TO_MS,
        weight_t=aq.get("TOTAL_WEIGHT") * _LB_TO_KG / 1000.0,
        on_ground=bool(aq.get("SIM_ON_GROUND")),
        zulu_s=aq.get("ZULU_TIME"),
    )


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
    isa_t_per_level, _ = isa(TARGET_FL * 100.0 * 0.3048)
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
        fl=TARGET_FL, temp_c=temp_k - 273.15, isa_dev_c=isa_dev_per_level,
        wind_kt=wind_per_level / KT_TO_MS, max_mach=mach_per_level,
        max_tas_kt=tas_ms_per_level / KT_TO_MS, gs_kt=gs_per_level / KT_TO_MS,
        above_ceiling=above_ceiling,
    ))
    return table, best_idx, current_idx, binding_at_best


def _recommendation_line(table, best_idx, current_idx, remaining_nm,
                          gain_threshold_kt, binding_at_best):
    """"CLIMB to FL550 (+14 kt, ~48 s over the remaining 1,240 nm) --
    binding: CAS" or "HOLD FL530" -- suppressed (HOLD) whenever the gain is
    under gain_threshold_kt, so this doesn't nag every tick over noise."""
    current_fl = float(table["fl"].iloc[current_idx])
    best_fl = float(table["fl"].iloc[best_idx])
    gain_kt = float(table["gs_kt"].iloc[best_idx] - table["gs_kt"].iloc[current_idx])

    if best_idx == current_idx or gain_kt < gain_threshold_kt:
        return f"HOLD FL{current_fl:.0f}"

    gs_current_ms = float(table["gs_kt"].iloc[current_idx]) * KT_TO_MS
    gs_best_ms = float(table["gs_kt"].iloc[best_idx]) * KT_TO_MS
    time_saved_s = remaining_nm * NM_TO_M * (1.0 / gs_current_ms - 1.0 / gs_best_ms)
    verb = "CLIMB" if best_fl > current_fl else "DESCEND"
    return (f"{verb} to FL{best_fl:.0f} ({gain_kt:+.0f} kt, ~{time_saved_s:.0f} s "
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
    """_format_hmm, but None -> 'n/a' -- the live panel shows this whenever
    a timing isn't knowable yet (elapsed before the recorder has seen brake
    release, or a pre-flight total with no --compare report given)."""
    return "n/a" if seconds is None else _format_hmm(seconds)


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
                    preflight_predicted_total_s, countdown_s, recorder_note=None):
    """Everything --live (the default) redraws in place each tick, as one
    rich renderable -- pure and testable via rich.console.Console(record=
    True), no terminal or Live instance needed. Layout, top to bottom, is
    the priority order from the spec: the recommendation first (the only
    line that matters at a glance), then current state, the level table,
    progress, and a countdown so a static screen still looks alive rather
    than hung between ticks."""
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

    progress_text = Text(
        f"Distance run {distance_run_nm:,.0f} nm / to run {remaining_nm:,.0f} nm   "
        f"Elapsed {_format_hmm_or_na(elapsed_s)}   "
        f"Remaining {_format_hmm_or_na(predicted_remaining_s)}   "
        f"Predicted total {_format_hmm_or_na(predicted_total_s)}   "
        f"(pre-flight: {_format_hmm_or_na(preflight_predicted_total_s)})"
    )

    countdown_text = Text(f"next update in {countdown_s:.0f}s", style="dim", justify="right")

    parts = [recommendation, Text(""), state_text, Text(""), level_table, Text(""), progress_text]
    if recorder_note:
        parts.append(Text(recorder_note, style="yellow"))
    parts.append(countdown_text)
    return Group(*parts)


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


def compare_to_report(report_df, cpa, touchdown_elapsed_s, accel_id="LINND", decel_id="BARIX"):
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
        predicted_decel_to_touchdown_s=DECEL_DESCENT_S,
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
                  record_path=None, compare_path=None, accel_id="LINND", decel_id="BARIX",
                  host="localhost", port=19285, cruise_mach=limits.CRUISE_MACH,
                  gain_threshold_kt=DEFAULT_GAIN_THRESHOLD_KT, simconnect_dll=None, live=True):
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

    Not covered by the test suite (needs a live SimConnect + Active Sky) --
    see compare_to_report and _level_table/_recommendation_line/
    _render_screen for the pure pieces that are."""
    plan = parse_pln(pln_path)
    legs = build_legs(plan["waypoints"])
    mask = supersonic_segment(legs, accel_id=accel_id, decel_id=decel_id)
    ss_legs = [leg for leg, m in zip(legs, mask) if m]
    route_end_cum_nm = ss_legs[-1].cum_nm

    # Read the compare report up front (not just at touchdown) so its total
    # predicted time can sit in the live panel's progress line throughout
    # the flight, not just in the post-touchdown comparison.
    report_df = None
    preflight_predicted_total_s = None
    if compare_path is not None:
        report_df = pd.read_csv(compare_path)
        preflight_predicted_total_s = (
            _parse_hmm_seconds(report_df["elapsed"].iloc[-1]) + DECEL_DESCENT_S)

    sm, aq = _connect(simconnect_dll)

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
        csv_writer.writerow(["zulu_s", "elapsed_s", "lat_deg", "lon_deg", "alt_ft",
                              "mach", "tas_kt", "gs_kt", "weight_t", "last_waypoint"])

    phase = "ground"
    prev_on_ground = None
    prev_gs_kt = None
    t0_monotonic = None
    touchdown_elapsed_s = None
    recorder_note = None  # live panel only -- --no-live prints these as before

    live_display = Live(console=Console()) if live else None
    if live_display is not None:
        live_display.start()

    interrupted = False
    try:
        while True:
            state = _read_state(aq)
            lat, lon = state["lat_deg"], state["lon_deg"]

            la_lat, la_lon, track_deg = project_along_route(legs, lat, lon, lookahead_nm)
            current_cum_nm, leg_idx, along_nm = current_progress_nm(legs, lat, lon)
            remaining_nm = max(route_end_cum_nm - current_cum_nm, 0.0)
            next_wp_id = legs[leg_idx].to_id
            dist_to_next_nm = max(legs[leg_idx].dist_nm - along_nm, 0.0)

            temp_k, u_ms, v_ms = _as_atmosphere(la_lat, la_lon, host, port)
            current_fl = state["alt_ft"] / 100.0
            table, best_idx, current_idx, binding_at_best = _level_table(
                temp_k, u_ms, v_ms, track_deg, state["weight_t"], cruise_mach, current_fl)

            elapsed_s = (time.monotonic() - t0_monotonic
                         if phase in ("recording", "done") else None)
            current_gs_kt = float(table["gs_kt"].iloc[current_idx])
            predicted_remaining_s = (
                remaining_nm * NM_TO_M / (current_gs_kt * KT_TO_MS) if current_gs_kt > 0 else None)
            predicted_total_s = (
                elapsed_s + predicted_remaining_s
                if elapsed_s is not None and predicted_remaining_s is not None else None)

            def _screen(countdown_s):
                return _render_screen(
                    state, table, best_idx, current_idx, binding_at_best, remaining_nm,
                    gain_threshold_kt, next_wp_id, dist_to_next_nm, current_cum_nm,
                    elapsed_s, predicted_remaining_s, predicted_total_s,
                    preflight_predicted_total_s, countdown_s, recorder_note)

            if live_display is not None:
                live_display.update(_screen(interval_s))
            else:
                _print_advisor(state, la_lat, la_lon, track_deg, table, best_idx, current_idx,
                                binding_at_best, remaining_nm, gain_threshold_kt)

            if recording:
                gs_kt = state["gs_kt"]
                if (phase == "ground" and prev_gs_kt is not None
                        and prev_gs_kt < BRAKE_RELEASE_GS_KT <= gs_kt):
                    phase = "recording"
                    t0_monotonic = time.monotonic()
                    recorder_note = f"Brake release detected -- recording to {record_path}"
                    if live_display is None:
                        print(f"\n{recorder_note}")
                elif phase == "recording" and prev_on_ground is False and state["on_ground"]:
                    phase = "done"
                    touchdown_elapsed_s = time.monotonic() - t0_monotonic
                    recorder_note = (f"Touchdown detected, elapsed "
                                      f"{_format_hmm(touchdown_elapsed_s)} since brake release")
                    if live_display is None:
                        print(f"\n{recorder_note}")

                if phase in ("recording", "done"):
                    elapsed_row_s = time.monotonic() - t0_monotonic
                    last_wp = _last_waypoint_passed(waypoint_cum_nm, current_cum_nm)
                    csv_writer.writerow([state["zulu_s"], round(elapsed_row_s, 1), lat, lon,
                                          state["alt_ft"], state["mach"], state["tas_kt"],
                                          gs_kt, state["weight_t"], last_wp])
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
                for countdown_s in range(int(interval_s), 0, -1):
                    live_display.update(_screen(countdown_s))
                    time.sleep(1)
            else:
                time.sleep(interval_s)
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
