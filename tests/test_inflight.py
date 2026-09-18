"""Acceptance tests for concopt.inflight's pure helpers: the per-level
table/recommendation the advisor prints, the waypoint-progress lookups the
recorder uses, the flight recorder's schema, and the predicted-vs-actual
comparison. No SimConnect, no Active Sky -- run_inflight (the live
orchestration) isn't covered here, same convention as search.run_search/
verify.run_verify. The recorder-schema tests below do touch the filesystem
(tmp_path), because "writes an empty field, not a zero" is a claim about
the bytes on disk and cannot be tested any other way.
"""
import csv

import numpy as np
import pandas as pd
import pytest
from rich.console import Console, Group

from concopt import inflight
from concopt.atmos import KT_TO_MS, isa
from concopt.search import TARGET_FL


def _still_air_atmosphere(isa_dev_c=0.0):
    """temp_k/u_ms/v_ms across TARGET_FL, still air at a uniform ISA
    deviation -- enough to exercise _level_table without needing a real
    Active Sky response."""
    isa_t_k, _ = isa(TARGET_FL * 100.0 * 0.3048)
    temp_k = isa_t_k + isa_dev_c
    u_ms = np.zeros_like(TARGET_FL)
    v_ms = np.zeros_like(TARGET_FL)
    return temp_k, u_ms, v_ms


def test_level_table_still_air_recommends_a_valid_level():
    temp_k, u_ms, v_ms = _still_air_atmosphere()
    table, best_idx, current_idx, binding_at_best = inflight._level_table(
        temp_k, u_ms, v_ms, track_deg=90.0, weight_t=135.0, cruise_mach=2.0, current_fl=450.0)

    assert len(table) == len(TARGET_FL)
    assert 0 <= best_idx < len(table)
    assert current_idx == 0  # current_fl exactly matches TARGET_FL[0]
    assert binding_at_best in ("cruise_mach", "CAS", "total_temp", "ceiling")
    assert not table["gs_kt"].iloc[best_idx] == -np.inf


def test_level_table_current_idx_picks_nearest_grid_point():
    temp_k, u_ms, v_ms = _still_air_atmosphere()
    _table, _best_idx, current_idx, _binding = inflight._level_table(
        temp_k, u_ms, v_ms, track_deg=90.0, weight_t=135.0, cruise_mach=2.0, current_fl=534.0)
    assert TARGET_FL[current_idx] == 530.0


def test_level_table_tailwind_beats_headwind_at_the_same_level():
    """A pure eastward tailwind on an eastward track should give a higher
    ground speed than the same table with a headwind, at every level."""
    temp_k, _u, _v = _still_air_atmosphere()
    tail_u = np.full_like(TARGET_FL, 50.0)
    head_u = np.full_like(TARGET_FL, -50.0)
    zero_v = np.zeros_like(TARGET_FL)

    tail_table, *_ = inflight._level_table(temp_k, tail_u, zero_v, 90.0, 135.0, 2.0, 500.0)
    head_table, *_ = inflight._level_table(temp_k, head_u, zero_v, 90.0, 135.0, 2.0, 500.0)

    assert np.all(tail_table["gs_kt"].to_numpy() > head_table["gs_kt"].to_numpy())


def test_level_table_marks_levels_above_ceiling():
    """At a heavy weight the top of TARGET_FL should be flagged
    above_ceiling -- same ceiling_ft the search uses."""
    temp_k, u_ms, v_ms = _still_air_atmosphere()
    table, *_ = inflight._level_table(temp_k, u_ms, v_ms, 90.0, 165.0, 2.0, 450.0)
    assert bool(table["above_ceiling"].iloc[-1])


def _table_with_levels(current_fl, current_gs_kt, best_fl, best_gs_kt):
    fl = TARGET_FL.copy()
    gs_kt = np.zeros_like(fl)
    ci = int(np.argmin(np.abs(fl - current_fl)))
    bi = int(np.argmin(np.abs(fl - best_fl)))
    gs_kt[ci] = current_gs_kt
    gs_kt[bi] = best_gs_kt
    table = pd.DataFrame(dict(fl=fl, gs_kt=gs_kt))
    return table, bi, ci


def test_recommendation_holds_when_gain_below_threshold():
    table, best_idx, current_idx = _table_with_levels(500.0, 1000.0, 510.0, 1002.0)
    line = inflight._recommendation_line(table, best_idx, current_idx, remaining_nm=1000.0,
                                          gain_threshold_kt=3.0, binding_at_best="cruise_mach")
    assert line == "HOLD FL500"


def test_recommendation_climbs_when_gain_exceeds_threshold():
    table, best_idx, current_idx = _table_with_levels(500.0, 1000.0, 530.0, 1020.0)
    line = inflight._recommendation_line(table, best_idx, current_idx, remaining_nm=1000.0,
                                          gain_threshold_kt=3.0, binding_at_best="CAS")
    assert line.startswith("CLIMB to FL530")
    assert "+20 kt" in line
    assert "binding: CAS" in line


def test_recommendation_descends_when_a_lower_level_is_faster():
    table, best_idx, current_idx = _table_with_levels(550.0, 1000.0, 500.0, 1030.0)
    line = inflight._recommendation_line(table, best_idx, current_idx, remaining_nm=500.0,
                                          gain_threshold_kt=3.0, binding_at_best="total_temp")
    assert line.startswith("DESCEND to FL500")


def test_recommendation_holds_when_already_at_the_best_level():
    table, best_idx, current_idx = _table_with_levels(520.0, 1100.0, 520.0, 1100.0)
    line = inflight._recommendation_line(table, best_idx, current_idx, remaining_nm=500.0,
                                          gain_threshold_kt=3.0, binding_at_best="cruise_mach")
    assert line == "HOLD FL520"


LEGS_FIXTURE_IDS = ["A", "LINND", "MID", "BARIX", "B"]
LEGS_FIXTURE_CUM_NM = [0.0, 100.0, 100.0, 400.0, 500.0]


def _fixture_legs():
    """Minimal Leg-shaped stand-ins (only from_id/to_id/cum_nm/dist_nm
    matter to _waypoint_cum_nm) -- a subdivided middle leg (A->LINND split
    into two sub-legs sharing from_id/to_id, then LINND->MID, MID->BARIX
    both collapsed into one hop, BARIX->B) covering both the "one leg per
    waypoint" and "subdivided parent leg" cases."""
    from concopt.route import Leg
    return [
        Leg("A", "LINND", 0.0, 0.0, 90.0, 50.0, 50.0),
        Leg("A", "LINND", 0.0, 0.0, 90.0, 50.0, 100.0),
        Leg("LINND", "BARIX", 0.0, 0.0, 90.0, 300.0, 400.0),
        Leg("BARIX", "B", 0.0, 0.0, 90.0, 100.0, 500.0),
    ]


def test_waypoint_cum_nm_uses_last_subleg_of_a_parent():
    legs = _fixture_legs()
    table = inflight._waypoint_cum_nm(legs)
    assert table == {"A": 0.0, "LINND": 100.0, "BARIX": 400.0, "B": 500.0}


@pytest.mark.parametrize("cum_nm, expected", [
    (0.0, "A"), (99.0, "A"), (100.0, "LINND"), (250.0, "LINND"),
    (400.0, "BARIX"), (450.0, "BARIX"), (500.0, "B"), (999.0, "B"),
])
def test_last_waypoint_passed(cum_nm, expected):
    table = inflight._waypoint_cum_nm(_fixture_legs())
    assert inflight._last_waypoint_passed(table, cum_nm) == expected


def test_parse_hmm_seconds_inverts_format_hmm():
    from concopt.search import _format_hmm
    assert inflight._parse_hmm_seconds(_format_hmm(3725.0)) == pytest.approx(3720.0)


def test_parse_hmm_seconds_examples():
    assert inflight._parse_hmm_seconds("1:05") == pytest.approx(3900.0)
    assert inflight._parse_hmm_seconds("0:20") == pytest.approx(1200.0)


REPORT_ARRIVAL_S = 1836.0  # 30.6 min -- report.py's real per-day figure, not the flat 35 min


def _sample_report_df(arrival_s=REPORT_ARRIVAL_S):
    """report.py's waypoint CSV, arrival_s populated on the decel waypoint's
    (last) row only -- NaN elsewhere, same convention report.py writes."""
    return pd.DataFrame([
        {"waypoint": "LINND", "elapsed": "0:20", "chosen_fl": 500.0, "gs_kt": 1100.0,
         "arrival_s": np.nan},
        {"waypoint": "MID", "elapsed": "0:45", "chosen_fl": 530.0, "gs_kt": 1150.0,
         "arrival_s": np.nan},
        {"waypoint": "BARIX", "elapsed": "1:10", "chosen_fl": 550.0, "gs_kt": 1160.0,
         "arrival_s": arrival_s},
    ])


def test_compare_to_report_deltas_and_constants():
    report_df = _sample_report_df()
    cpa = {
        "LINND": dict(dist_nm=0.1, elapsed_s=1250.0, fl=500.0, mach=2.0, tas_kt=1140.0, gs_kt=1105.0),
        "MID": dict(dist_nm=0.2, elapsed_s=2760.0, fl=530.0, mach=2.0, tas_kt=1150.0, gs_kt=1140.0),
        "BARIX": dict(dist_nm=0.1, elapsed_s=4260.0, fl=550.0, mach=2.0, tas_kt=1160.0, gs_kt=1170.0),
    }
    touchdown_elapsed_s = 4260.0 + REPORT_ARRIVAL_S + 60.0  # a minute later than predicted

    table, constants = inflight.compare_to_report(report_df, cpa, touchdown_elapsed_s,
                                                    accel_id="LINND", decel_id="BARIX")

    assert table.loc[table["waypoint"] == "LINND", "delta_elapsed_s"].iloc[0] == pytest.approx(50.0)
    assert table.loc[table["waypoint"] == "BARIX", "delta_fl"].iloc[0] == pytest.approx(0.0)

    assert constants["measured_brake_to_accel_s"] == pytest.approx(1250.0)
    assert constants["predicted_brake_to_accel_s"] == pytest.approx(20.0 * 60.0)
    assert constants["measured_supersonic_s"] == pytest.approx(4260.0 - 1250.0)
    assert constants["predicted_supersonic_s"] == pytest.approx((70 - 20) * 60.0)
    assert constants["measured_decel_to_touchdown_s"] == pytest.approx(REPORT_ARRIVAL_S + 60.0)
    # The report's OWN real arrival figure, not the old flat 35 min constant.
    assert constants["predicted_decel_to_touchdown_s"] == pytest.approx(REPORT_ARRIVAL_S)


def test_compare_to_report_missing_waypoint_is_nan_not_a_crash():
    report_df = _sample_report_df()
    cpa = {"LINND": None, "MID": None, "BARIX": None}  # recorder never got a fix near any of them

    table, constants = inflight.compare_to_report(report_df, cpa, touchdown_elapsed_s=5000.0,
                                                    accel_id="LINND", decel_id="BARIX")

    assert table["actual_elapsed_s"].isna().all()
    assert np.isnan(constants["measured_supersonic_s"])


# --- A4: the live in-flight display ----------------------------------------
# rich renderables and _print_advisor's plain output can both be built and
# inspected without a sim -- run_inflight's own loop (SimConnect + Active
# Sky) stays out of scope, same convention as the rest of this file.

def _synthetic_state():
    return dict(lat_deg=40.123, lon_deg=-69.876, alt_ft=53000.0, mach=2.0,
                tas_kt=1150.0, gs_kt=1200.0, weight_t=135.0, on_ground=False, zulu_s=3600.0)


def test_format_hmm_or_na():
    assert inflight._format_hmm_or_na(None) == "n/a"
    assert inflight._format_hmm_or_na(3900.0) == "1:05"


def test_format_hmm_or_na_handles_nan_not_just_none():
    """C3 regression: predicted_remaining_s/predicted_total_s are arithmetic
    on arrival_info['time_min'] (run_inflight: time_to_decel_s +
    arrival_time_s) -- a NaN there is not None (`nan is not None` is True),
    so the old None-only guard let a NaN straight through to _format_hmm's
    int(round(seconds / 60.0)), crashing with ValueError. Found live via
    the C3 replay harness's synthetic flight (a degenerate Active Sky
    reading in _build_live_arrival_wind_fn produced the NaN); this pins the
    fix without needing that whole scenario."""
    assert inflight._format_hmm_or_na(float("nan")) == "n/a"


def test_recommendation_text_hold_is_muted():
    text = inflight._recommendation_text("HOLD FL500")
    assert text.style == "dim"


@pytest.mark.parametrize("line", [
    "CLIMB to FL550 (+14 kt, ~48 s over the remaining 100 nm) -- binding: CAS",
    "DESCEND to FL500 (+30 kt, ~100 s over the remaining 500 nm) -- binding: total_temp",
])
def test_recommendation_text_climb_and_descend_are_not_muted(line):
    text = inflight._recommendation_text(line)
    assert text.style != "dim"


def test_build_level_table_marks_current_and_recommended_rows():
    temp_k, u_ms, v_ms = _still_air_atmosphere()
    table, best_idx, current_idx, _binding = inflight._level_table(
        temp_k, u_ms, v_ms, track_deg=90.0, weight_t=135.0, cruise_mach=2.0, current_fl=530.0)

    rich_table = inflight._build_level_table(table, best_idx, current_idx)

    assert rich_table.row_count == len(TARGET_FL)
    console = Console(width=120, record=True)
    console.print(rich_table)
    text = console.export_text()
    assert "CURRENT" in text
    assert "REC" in text


def test_render_screen_produces_a_renderable_from_a_synthetic_state():
    """The live panel's top-level function -- built from a plain dict and
    plain numbers, no SimConnect/Active Sky/Live instance involved."""
    temp_k, u_ms, v_ms = _still_air_atmosphere()
    table, best_idx, current_idx, binding_at_best = inflight._level_table(
        temp_k, u_ms, v_ms, track_deg=90.0, weight_t=135.0, cruise_mach=2.0, current_fl=530.0)
    state = _synthetic_state()

    screen = inflight._render_screen(
        state, table, best_idx, current_idx, binding_at_best, remaining_nm=1200.0,
        gain_threshold_kt=3.0, next_wp_id="BARIX", dist_to_next_nm=42.0,
        distance_run_nm=300.0, elapsed_s=1800.0, predicted_remaining_s=3600.0,
        predicted_total_s=5400.0, preflight_predicted_total_s=5300.0, countdown_s=45,
        recorder_note="Brake release detected -- recording to out.csv")

    assert isinstance(screen, Group)
    console = Console(width=120, record=True)
    console.print(screen)
    text = console.export_text()
    assert "FL530" in text
    assert "BARIX" in text
    assert "42 nm" in text
    assert "next update in 45s" in text
    assert "Brake release detected" in text
    assert "n/a" not in text  # every timing was supplied in this call


def test_render_screen_shows_na_for_unknown_timings():
    """Before the recorder has started (or with no --compare report), the
    elapsed/predicted-total/pre-flight fields are None -- the panel should
    read 'n/a', not blow up or print 'None'."""
    temp_k, u_ms, v_ms = _still_air_atmosphere()
    table, best_idx, current_idx, binding_at_best = inflight._level_table(
        temp_k, u_ms, v_ms, track_deg=90.0, weight_t=135.0, cruise_mach=2.0, current_fl=530.0)
    state = _synthetic_state()

    screen = inflight._render_screen(
        state, table, best_idx, current_idx, binding_at_best, remaining_nm=1200.0,
        gain_threshold_kt=3.0, next_wp_id="BARIX", dist_to_next_nm=42.0,
        distance_run_nm=300.0, elapsed_s=None, predicted_remaining_s=None,
        predicted_total_s=None, preflight_predicted_total_s=None, countdown_s=60,
        recorder_note=None)

    console = Console(width=120, record=True)
    console.print(screen)
    text = console.export_text()
    assert "None" not in text
    assert text.count("n/a") == 4  # elapsed, remaining, predicted total, pre-flight total


def test_print_advisor_still_produces_plain_scrolling_output(capsys):
    """--no-live's whole job: the original prints, unchanged."""
    temp_k, u_ms, v_ms = _still_air_atmosphere()
    table, best_idx, current_idx, binding_at_best = inflight._level_table(
        temp_k, u_ms, v_ms, track_deg=90.0, weight_t=135.0, cruise_mach=2.0, current_fl=530.0)
    state = _synthetic_state()

    inflight._print_advisor(state, 40.5, -69.0, 90.0, table, best_idx, current_idx,
                             binding_at_best, remaining_nm=1200.0, gain_threshold_kt=3.0)

    out = capsys.readouterr().out
    assert "lookahead point" in out
    assert "FL530" in out


# --- Phase C1: the arrival model, live and pre-flight -----------------------

def _synthetic_wind_at_fl(level_fl):
    """A trivial wind_at_fl for _live_arrival -- still air, a fixed temp
    regardless of the requested level -- enough to exercise the arithmetic
    without needing a real Active Sky wind profile."""
    level_fl = np.atleast_1d(np.asarray(level_fl, dtype=float))
    n = level_fl.shape[0]
    return {"wind_kt": np.zeros(n), "temp_k": np.full(n, 218.0),
            "fl_clamped": np.zeros(n, dtype=bool)}


def test_preflight_predicted_total_s_uses_reports_own_arrival_not_a_flat_constant():
    """The whole point of this task: the report CSV already accounts for a
    real per-day arrival (30.6 min here), so the pre-flight total must be
    elapsed + that figure, NOT elapsed + the old flat 35 min constant."""
    report_df = _sample_report_df()
    total_s = inflight._preflight_predicted_total_s(report_df)
    assert total_s == pytest.approx(_parse_elapsed("1:10") + REPORT_ARRIVAL_S)
    assert total_s != pytest.approx(_parse_elapsed("1:10") + 35.0 * 60.0)


def _parse_elapsed(s):
    return inflight._parse_hmm_seconds(s)


def test_report_arrival_s_raises_on_missing_column():
    """A report CSV from before this task (no arrival_s column) must not
    silently fall back to the flat constant it was written to replace."""
    old_report_df = _sample_report_df().drop(columns=["arrival_s"])
    with pytest.raises(ValueError, match="arrival_s"):
        inflight._report_arrival_s(old_report_df)
    with pytest.raises(ValueError, match="arrival_s"):
        inflight._preflight_predicted_total_s(old_report_df)


def test_live_arrival_returns_a_segment_breakdown():
    info = inflight._live_arrival(cruise_fl=550.0, arrival_nm=307.0, isa_dev_c=0.0,
                                   mass_at_barix_t=118.0, wind_at_fl=_synthetic_wind_at_fl)
    assert info["source"] == "live"
    assert info["time_min"] > 0.0
    assert info["decel_time_min"] > 0.0
    assert info["level_time_min"] >= 0.0
    assert info["descent_time_min"] > 0.0
    assert info["level_fl"] > 0.0
    assert (info["decel_time_min"] + info["level_time_min"] + info["descent_time_min"]
            + inflight.arrival.APPROACH_MIN) == pytest.approx(info["time_min"])


def test_build_arrival_text_renders_the_segment_breakdown():
    info = dict(time_min=30.6, decel_time_min=6.2, level_time_min=12.4,
                descent_time_min=10.5, level_fl=350.0, level_wind_kt=-18.0,
                flags="", source="live")
    console = Console(width=120, record=True)
    console.print(inflight._build_arrival_text(info))
    text = console.export_text()

    assert "decel" in text
    assert "level" in text and "FL350" in text and "-18" in text
    assert "descent" in text
    assert "approach" in text
    assert "30.6 min total" in text


def test_build_arrival_text_none_reads_not_yet_available():
    """Before anything is computable (no --compare report AND Active Sky
    hasn't answered a live query yet) -- must not blow up."""
    console = Console(width=120, record=True)
    console.print(inflight._build_arrival_text(None))
    assert "not yet available" in console.export_text()


def test_build_arrival_text_fallback_shows_note_not_a_segment_breakdown():
    """The Active Sky fallback path: the panel must show the fallback note
    (and the pre-flight total) rather than throwing or silently presenting
    a fabricated segment split."""
    info = dict(time_min=30.6, source="fallback")
    console = Console(width=120, record=True)
    console.print(inflight._build_arrival_text(info))
    text = console.export_text()

    assert "unavailable" in text
    assert "30.6 min" in text
    assert "decel" not in text


def test_render_screen_includes_the_arrival_breakdown_and_live_vs_preflight():
    temp_k, u_ms, v_ms = _still_air_atmosphere()
    table, best_idx, current_idx, binding_at_best = inflight._level_table(
        temp_k, u_ms, v_ms, track_deg=90.0, weight_t=135.0, cruise_mach=2.0, current_fl=530.0)
    state = _synthetic_state()
    arrival_info = dict(time_min=30.6, decel_time_min=6.2, level_time_min=12.4,
                         descent_time_min=10.5, level_fl=350.0, level_wind_kt=-18.0,
                         flags="", source="live")

    screen = inflight._render_screen(
        state, table, best_idx, current_idx, binding_at_best, remaining_nm=1200.0,
        gain_threshold_kt=3.0, next_wp_id="BARIX", dist_to_next_nm=42.0,
        distance_run_nm=300.0, elapsed_s=1800.0, predicted_remaining_s=3600.0,
        predicted_total_s=5400.0, preflight_predicted_total_s=5300.0, countdown_s=45,
        arrival_info=arrival_info)

    console = Console(width=120, record=True)
    console.print(screen)
    text = console.export_text()
    assert "decel" in text and "FL350" in text
    assert "live" in text and "pre-flight" in text


def test_build_live_arrival_wind_fn_returns_none_when_active_sky_errors(monkeypatch):
    """The fallback path's trigger: Active Sky not answering (a point this
    far ahead can be outside the loaded scenario) must come back as None,
    not raise, so the caller can fall back to the pre-flight report figure."""
    def _raise(*args, **kwargs):
        raise RuntimeError("Active Sky not responding")
    monkeypatch.setattr(inflight, "get_atmosphere_np", _raise)
    weather_source = inflight._live_weather_source("localhost", 19285)

    wind_at_fl = inflight._build_live_arrival_wind_fn(weather_source, 40.0, -60.0, 90.0)
    assert wind_at_fl is None


def test_build_live_arrival_wind_fn_returns_none_on_a_short_response(monkeypatch):
    """A malformed/short response (fewer altitudes than requested) is
    another way Active Sky can fail to answer -- also None, not a crash."""
    def _short(lat, lon, alts_ft, host_addr=None, port=None):
        n = len(alts_ft) - 1
        return (np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, 500.0), np.full(n, 15.0))
    monkeypatch.setattr(inflight, "get_atmosphere_np", _short)
    weather_source = inflight._live_weather_source("localhost", 19285)

    wind_at_fl = inflight._build_live_arrival_wind_fn(weather_source, 40.0, -60.0, 90.0)
    assert wind_at_fl is None


def test_build_live_arrival_wind_fn_interpolates_in_log_pressure(monkeypatch):
    """A well-formed response builds a usable wind_at_fl: within the sampled
    span it interpolates (not just repeats an endpoint), and it reports
    fl_clamped outside that span."""
    n = len(inflight.ARRIVAL_WIND_SAMPLE_FL)
    # Pressure falls monotonically with the sampled FL ladder -- roughly
    # ISA-shaped, close enough to exercise the interpolation.
    pressure_hpa = np.linspace(700.0, 50.0, n)
    wind_dir_deg = np.zeros(n)
    wind_speed_kt = np.linspace(20.0, 80.0, n)  # increasing with altitude
    temp_c = np.linspace(0.0, -60.0, n)

    def _fake(lat, lon, alts_ft, host_addr=None, port=None):
        return (alts_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c)
    monkeypatch.setattr(inflight, "get_atmosphere_np", _fake)
    weather_source = inflight._live_weather_source("localhost", 19285)

    wind_at_fl = inflight._build_live_arrival_wind_fn(weather_source, 40.0, -60.0, 0.0)
    assert wind_at_fl is not None

    mid_fl = (inflight.ARRIVAL_WIND_SAMPLE_FL[0] + inflight.ARRIVAL_WIND_SAMPLE_FL[1]) / 2.0
    atm = wind_at_fl(np.array([mid_fl]))
    assert not atm["fl_clamped"][0]

    # Well below the lowest sampled pressure (700 hPa, ~FL99) -- must clamp.
    atm_below = wind_at_fl(np.array([0.0]))
    assert atm_below["fl_clamped"][0]


# --- Phase C2: the flight recorder schema -----------------------------------
# There is no second chance at a given day's weather, so these tests are
# about what the file can and cannot answer afterwards: that the three
# weather sources land in it, that a missing reading stays missing rather
# than becoming a confident zero, and that a notebook can tell what vintage
# of schema it is holding.

def _recorder_state(**overrides):
    """A full _read_state dict: every required key, and every optional one
    present (the all-available case). Pass e.g. sim_wind_kt=None to model a
    sim variable the add-on doesn't wire."""
    state = dict(
        lat_deg=51.1234567, lon_deg=-0.7654321, alt_ft=53000.0, mach=2.0,
        tas_kt=1150.0, gs_kt=1200.0, weight_t=135.0, on_ground=False, zulu_s=3600.0,
        fuel_kg=42000.0, sim_wind_kt=60.0, sim_wind_dir_deg=270.0, sim_temp_c=-55.0,
        sim_pressure_hpa=96.0, agl_ft=52950.0, vs_fpm=0.0, cas_kt=520.0,
        track_deg=91.5,
    )
    state.update(overrides)
    return state


def _as_point(**overrides):
    point = dict(wind_dir_deg=270.0, wind_kt=55.0, temp_c=-54.0, pressure_hpa=95.0)
    point.update(overrides)
    return point


def _advisory_dict(action="HOLD", rec_fl=530.0, gain_kt=1.2, gain_s=40.0):
    return dict(action=action, current_fl=530.0, rec_fl=rec_fl,
                gain_kt=gain_kt, gain_s=gain_s)


def _write_recording(path, rows):
    """rows (list of _record_row dicts) -> a CSV written exactly the way
    run_inflight writes it, header included."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(inflight.RECORD_COLUMNS)
        for row in rows:
            writer.writerow(inflight._row_values(row))
    return path


def _sample_record_row(**overrides):
    row = inflight._record_row(
        elapsed_s=1234.5, state=_recorder_state(), cum_nm=2100.25,
        last_waypoint="BARIX", leg_track_deg=90.0, flight_phase="cruise",
        as_point=_as_point(), advisory=_advisory_dict(),
        binding_at_best="cruise_mach", arrival_info=dict(time_min=30.6, source="live"),
        predicted_remaining_s=3600.0, predicted_total_s=5400.0)
    row.update(overrides)
    return row


def test_record_row_covers_the_declared_schema_exactly():
    """_record_row and RECORD_COLUMNS must not drift apart -- a column in
    one and not the other is either a KeyError at write time or a silently
    dropped measurement."""
    assert set(_sample_record_row()) == set(inflight.RECORD_COLUMNS)
    assert len(inflight.RECORD_COLUMNS) == len(set(inflight.RECORD_COLUMNS))


def test_recording_header_matches_the_declared_schema_version(tmp_path):
    """A notebook must be able to tell which vintage it is reading without
    guessing from the column list."""
    path = _write_recording(tmp_path / "rec.csv", [_sample_record_row()])
    df = pd.read_csv(path)

    assert list(df.columns) == inflight.RECORD_COLUMNS
    assert (df["schema_version"] == inflight.RECORD_SCHEMA_VERSION).all()
    assert inflight.RECORD_SCHEMA_VERSION >= 2  # 1 was the narrow zulu_s..last_waypoint schema


def test_recording_carries_all_three_weather_sources(tmp_path):
    """Question 1: Active Sky at the aircraft, the sim's own ambient, and
    what the aircraft's speeds imply -- all three, per row, or the
    Active-Sky-to-actual error can't be measured at all."""
    path = _write_recording(tmp_path / "rec.csv", [_sample_record_row()])
    row = pd.read_csv(path).iloc[0]

    assert row["as_wind_kt"] == pytest.approx(55.0)
    assert row["sim_wind_kt"] == pytest.approx(60.0)
    assert row["gs_minus_tas_kt"] == pytest.approx(1200.0 - 1150.0)
    # Both sources projected on the SAME (route leg) track, so their
    # difference means something. A 270 deg wind on a 090 deg track is a
    # pure tailwind.
    assert row["as_wind_along_kt"] == pytest.approx(55.0)
    assert row["sim_wind_along_kt"] == pytest.approx(60.0)
    assert row["as_temp_c"] == pytest.approx(-54.0)
    assert row["sim_temp_c"] == pytest.approx(-55.0)


def test_missing_values_write_empty_and_read_back_as_nan_not_zero(tmp_path):
    """The constraint that matters most: a zero wind and a missing wind
    must not look the same. csv.writer turns None into an empty field;
    pandas reads that back as NaN."""
    state = _recorder_state(sim_wind_kt=None, sim_wind_dir_deg=None, sim_temp_c=None,
                             sim_pressure_hpa=None, fuel_kg=None, agl_ft=None,
                             vs_fpm=None, cas_kt=None, track_deg=None)
    row = inflight._record_row(
        elapsed_s=10.0, state=state, cum_nm=1.0, last_waypoint="KJFK",
        leg_track_deg=90.0, flight_phase="climb", as_point=None,
        advisory=_advisory_dict(gain_s=None), binding_at_best="CAS",
        arrival_info=None, predicted_remaining_s=None, predicted_total_s=None)
    path = _write_recording(tmp_path / "rec.csv", [row])

    raw = path.read_text().splitlines()[1].split(",")
    missing_columns = ["sim_wind_kt", "sim_temp_c", "fuel_kg", "agl_ft", "cas_kt",
                       "track_deg", "as_wind_kt", "as_temp_c", "as_wind_along_kt",
                       "rec_gain_s", "predicted_remaining_s", "arrival_time_min"]
    for column in missing_columns:
        assert raw[inflight.RECORD_COLUMNS.index(column)] == "", column

    df = pd.read_csv(path)
    for column in missing_columns:
        assert pd.isna(df[column].iloc[0]), column
        assert df[column].iloc[0] != 0


def test_a_present_zero_wind_is_not_confused_with_a_missing_one(tmp_path):
    """The other half of the same constraint: a genuine calm must survive
    as 0.0, not be swallowed as missing."""
    state = _recorder_state(sim_wind_kt=0.0, sim_wind_dir_deg=0.0)
    row = inflight._record_row(
        elapsed_s=10.0, state=state, cum_nm=1.0, last_waypoint="KJFK",
        leg_track_deg=90.0, flight_phase="climb", as_point=_as_point(wind_kt=0.0),
        advisory=_advisory_dict(), binding_at_best="CAS", arrival_info=None,
        predicted_remaining_s=None, predicted_total_s=None)
    df = pd.read_csv(_write_recording(tmp_path / "rec.csv", [row]))

    assert df["sim_wind_kt"].iloc[0] == 0.0
    assert not pd.isna(df["sim_wind_kt"].iloc[0])
    assert df["as_wind_along_kt"].iloc[0] == 0.0
    assert not pd.isna(df["as_wind_along_kt"].iloc[0])


def _cpa_from_recording(df, wp_positions):
    """Rebuild compare_to_report's cpa dict from a recording -- the
    closest-point-of-approach scan run_inflight does live, redone
    post-hoc from the file. Six lines, using only recorded columns, which
    is the point: the recording is self-sufficient."""
    from concopt.route import great_circle_nm

    cpa = {}
    for wid, (wlat, wlon) in wp_positions.items():
        d = great_circle_nm(df["lat_deg"].to_numpy(), df["lon_deg"].to_numpy(), wlat, wlon)
        i = int(np.argmin(d))
        cpa[wid] = dict(dist_nm=float(d[i]), elapsed_s=float(df["elapsed_s"].iloc[i]),
                        fl=float(df["level_fl"].iloc[i]), mach=float(df["mach"].iloc[i]),
                        tas_kt=float(df["tas_kt"].iloc[i]), gs_kt=float(df["gs_kt"].iloc[i]))
    return cpa


def test_synthetic_recording_round_trips_through_compare_to_report(tmp_path):
    """Widening the schema must not disturb compare_to_report: a recording
    written with the wide schema, read back and reduced to cpa, must give
    exactly what the same flight reduced from the narrow (v1) columns
    gives. The extra columns are inert."""
    wp_positions = {"LINND": (40.0, -70.0), "MID": (45.0, -50.0), "BARIX": (50.0, -10.0)}
    samples = [
        (1250.0, 40.0, -70.0, 50000.0, 2.0, 1140.0, 1105.0),
        (2760.0, 45.0, -50.0, 53000.0, 2.0, 1150.0, 1140.0),
        (4260.0, 50.0, -10.0, 55000.0, 2.0, 1160.0, 1170.0),
    ]
    rows = [
        inflight._record_row(
            elapsed_s=elapsed_s,
            state=_recorder_state(lat_deg=lat, lon_deg=lon, alt_ft=alt_ft, mach=mach,
                                   tas_kt=tas_kt, gs_kt=gs_kt),
            cum_nm=100.0 * i, last_waypoint="MID", leg_track_deg=90.0,
            flight_phase="cruise", as_point=_as_point(), advisory=_advisory_dict(),
            binding_at_best="cruise_mach",
            arrival_info=dict(time_min=30.6, source="live"),
            predicted_remaining_s=3600.0, predicted_total_s=5400.0)
        for i, (elapsed_s, lat, lon, alt_ft, mach, tas_kt, gs_kt) in enumerate(samples)
    ]
    wide = pd.read_csv(_write_recording(tmp_path / "wide.csv", rows))

    # The same flight as a v1-shaped recording -- ONLY the narrow columns,
    # with fl derived the way v1's live cpa derived it (alt_ft / 100).
    narrow = wide[["zulu_s", "elapsed_s", "lat_deg", "lon_deg", "alt_ft", "mach",
                    "tas_kt", "gs_kt", "weight_t", "last_waypoint"]].copy()
    narrow["level_fl"] = narrow["alt_ft"] / 100.0

    report_df = _sample_report_df()
    wide_table, wide_constants = inflight.compare_to_report(
        report_df, _cpa_from_recording(wide, wp_positions), 6096.0)
    narrow_table, narrow_constants = inflight.compare_to_report(
        report_df, _cpa_from_recording(narrow, wp_positions), 6096.0)

    pd.testing.assert_frame_equal(wide_table, narrow_table)
    assert wide_constants == narrow_constants
    # And it actually measured something, rather than matching as all-NaN.
    assert wide_table["actual_elapsed_s"].notna().all()
    assert wide_constants["measured_supersonic_s"] == pytest.approx(4260.0 - 1250.0)


@pytest.mark.parametrize("cum_nm, mach, agl_ft, on_ground, expected", [
    (0.0, 0.3, 30.0, True, "ground"),         # take-off roll
    (50.0, 0.8, 8000.0, False, "climb"),      # before the accel waypoint
    (500.0, 2.0, 55000.0, False, "cruise"),   # accel -> decel
    (3100.0, 1.6, 40000.0, False, "decel"),   # past decel, still supersonic
    (3200.0, 0.8, 20000.0, False, "descent"),  # past decel, subsonic
    (3250.0, 0.4, 1200.0, False, "approach"),  # below arrival.py's 1,500 ft
])
def test_flight_phase_cuts_segments_without_inferring_from_altitude(
        cum_nm, mach, agl_ft, on_ground, expected):
    assert inflight._flight_phase(cum_nm, accel_cum_nm=200.0, decel_cum_nm=3000.0,
                                   mach=mach, agl_ft=agl_ft,
                                   on_ground=on_ground) == expected


def test_flight_phase_without_agl_stays_descent_rather_than_guessing():
    """agl_ft is an optional read; where it's missing the approach cut
    isn't knowable and the sample must not claim to be an approach."""
    assert inflight._flight_phase(3250.0, 200.0, 3000.0, mach=0.4, agl_ft=None,
                                   on_ground=False) == "descent"


@pytest.mark.parametrize("wind_dir_deg, track_deg, expected", [
    (270.0, 90.0, 50.0),    # westerly wind, eastbound -> pure tailwind
    (90.0, 90.0, -50.0),    # easterly wind, eastbound -> pure headwind
    (180.0, 90.0, 0.0),     # southerly wind, eastbound -> pure crosswind
])
def test_wind_along_track_uses_the_meteorological_from_bearing(
        wind_dir_deg, track_deg, expected):
    assert inflight._wind_along_track_kt(50.0, wind_dir_deg, track_deg) == pytest.approx(
        expected, abs=1e-9)


@pytest.mark.parametrize("wind_kt, wind_dir_deg, track_deg", [
    (None, 270.0, 90.0), (50.0, None, 90.0), (50.0, 270.0, None),
])
def test_wind_along_track_none_in_none_out(wind_kt, wind_dir_deg, track_deg):
    """A missing input must not become a confident zero."""
    assert inflight._wind_along_track_kt(wind_kt, wind_dir_deg, track_deg) is None


def test_advisory_matches_the_line_it_formats():
    """_advisory is the recorder's copy of the panel's own decision -- if
    the two ever disagree the recording stops being evidence about the
    advice that was actually shown."""
    table, best_idx, current_idx = _table_with_levels(500.0, 1000.0, 530.0, 1020.0)
    adv = inflight._advisory(table, best_idx, current_idx, remaining_nm=1000.0,
                              gain_threshold_kt=3.0)
    line = inflight._recommendation_line(table, best_idx, current_idx, remaining_nm=1000.0,
                                          gain_threshold_kt=3.0, binding_at_best="CAS")

    assert adv["action"] == "CLIMB"
    assert adv["rec_fl"] == 530.0
    assert f"{adv['action']} to FL{adv['rec_fl']:.0f}" in line
    assert f"{adv['gain_kt']:+.0f} kt" in line
    assert f"~{adv['gain_s']:.0f} s" in line


def test_advisory_reports_the_gain_even_on_a_hold():
    """A HOLD at +2.9 kt is a different event from a HOLD at +0.0 kt, and
    only the recorded gain tells them apart."""
    table, best_idx, current_idx = _table_with_levels(500.0, 1000.0, 510.0, 1002.0)
    adv = inflight._advisory(table, best_idx, current_idx, remaining_nm=1000.0,
                              gain_threshold_kt=3.0)

    assert adv["action"] == "HOLD"
    assert adv["gain_kt"] == pytest.approx(2.0)
    assert adv["gain_s"] is not None and adv["gain_s"] > 0.0
    assert adv["rec_fl"] == 510.0  # the level it would have recommended


def test_advisory_gain_s_is_none_rather_than_a_divide_by_zero():
    table, best_idx, current_idx = _table_with_levels(500.0, 0.0, 530.0, 0.0)
    adv = inflight._advisory(table, best_idx, current_idx, remaining_nm=1000.0,
                              gain_threshold_kt=3.0)
    assert adv["gain_s"] is None


def test_low_altitude_tick_can_resolve_the_approach_allowance():
    """Question 3 is only answerable if the sample interval resolves a
    1.5 min segment -- at the cruise default it gets one or two rows."""
    from concopt.arrival import APPROACH_MIN

    approach_s = APPROACH_MIN * 60.0
    assert approach_s / inflight.DEFAULT_INTERVAL_S < 2  # the problem
    assert approach_s / inflight.LOW_ALT_INTERVAL_S >= 15  # the fix
    assert inflight.LOW_ALT_INTERVAL_S < inflight.DEFAULT_INTERVAL_S
    assert inflight.APPROACH_AGL_FT == 1500.0  # arrival.py's own descent end


def test_round_or_none_passes_none_through():
    assert inflight._round_or_none(None, 2) is None
    assert inflight._round_or_none(1.23456, 2) == pytest.approx(1.23)
