"""Acceptance tests for concopt.inflight's pure helpers: the per-level
table/recommendation the advisor prints, the waypoint-progress lookups the
recorder uses, and the predicted-vs-actual comparison. No SimConnect, no
Active Sky, no file I/O -- run_inflight (the live orchestration) isn't
covered here, same convention as search.run_search/verify.run_verify.
"""
import numpy as np
import pandas as pd
import pytest

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


def _sample_report_df():
    return pd.DataFrame([
        {"waypoint": "LINND", "elapsed": "0:20", "chosen_fl": 500.0, "gs_kt": 1100.0},
        {"waypoint": "MID", "elapsed": "0:45", "chosen_fl": 530.0, "gs_kt": 1150.0},
        {"waypoint": "BARIX", "elapsed": "1:10", "chosen_fl": 550.0, "gs_kt": 1160.0},
    ])


def test_compare_to_report_deltas_and_constants():
    report_df = _sample_report_df()
    cpa = {
        "LINND": dict(dist_nm=0.1, elapsed_s=1250.0, fl=500.0, mach=2.0, tas_kt=1140.0, gs_kt=1105.0),
        "MID": dict(dist_nm=0.2, elapsed_s=2760.0, fl=530.0, mach=2.0, tas_kt=1150.0, gs_kt=1140.0),
        "BARIX": dict(dist_nm=0.1, elapsed_s=4260.0, fl=550.0, mach=2.0, tas_kt=1160.0, gs_kt=1170.0),
    }
    touchdown_elapsed_s = 4260.0 + 35.0 * 60.0 + 60.0  # a minute later than the 35 min default

    table, constants = inflight.compare_to_report(report_df, cpa, touchdown_elapsed_s,
                                                    accel_id="LINND", decel_id="BARIX")

    assert table.loc[table["waypoint"] == "LINND", "delta_elapsed_s"].iloc[0] == pytest.approx(50.0)
    assert table.loc[table["waypoint"] == "BARIX", "delta_fl"].iloc[0] == pytest.approx(0.0)

    assert constants["measured_brake_to_accel_s"] == pytest.approx(1250.0)
    assert constants["predicted_brake_to_accel_s"] == pytest.approx(20.0 * 60.0)
    assert constants["measured_supersonic_s"] == pytest.approx(4260.0 - 1250.0)
    assert constants["predicted_supersonic_s"] == pytest.approx((70 - 20) * 60.0)
    assert constants["measured_decel_to_touchdown_s"] == pytest.approx(35.0 * 60.0 + 60.0)
    assert constants["predicted_decel_to_touchdown_s"] == pytest.approx(35.0 * 60.0)


def test_compare_to_report_missing_waypoint_is_nan_not_a_crash():
    report_df = _sample_report_df()
    cpa = {"LINND": None, "MID": None, "BARIX": None}  # recorder never got a fix near any of them

    table, constants = inflight.compare_to_report(report_df, cpa, touchdown_elapsed_s=5000.0,
                                                    accel_id="LINND", decel_id="BARIX")

    assert table["actual_elapsed_s"].isna().all()
    assert np.isnan(constants["measured_supersonic_s"])
