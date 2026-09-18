"""Phase C3: run_inflight through its full loop -- ground, climb, cruise,
decel, descent, approach, touchdown -- with no SimConnect and no Active Sky,
via concopt.replay's two injectable sources. run_inflight itself had never
executed before this file; every test here is therefore also the first live
proof of the recorder state machine (brake release, touchdown), the
low-altitude tick tightening, and the Ctrl+C exit path.
"""
import time

import numpy as np
import pandas as pd
import pytest

from concopt.atmos import isa, pressure_to_fl
from concopt.inflight import BRAKE_RELEASE_GS_KT, run_inflight
from concopt.replay import (_BRAKE_RELEASE_GS_KT, _row_weather, build_synthetic_flight,
                             replay_sources)
from concopt.route import build_legs, climb_cruise_segment, parse_pln
from tests.test_route import SAMPLE_PLN

# Fast enough that a full ~3-hour synthetic flight replays in well under the
# "under a minute" budget the brief sets, with margin (see the reported
# wall-clock time in the C3 report).
_FAST_REPLAY_SPEED = 900.0


def _still_air_data(levels_hpa, legs):
    """era5.load_legs_npz-shaped dict, ISA+0 and zero wind everywhere --
    same construction as test_report.py's _still_air_npz, built directly in
    memory rather than round-tripped through an actual .npz file (load_legs_npz
    just hands back the same dict of arrays np.savez was given, so the file
    round trip adds nothing here)."""
    fl_at_level = pressure_to_fl(levels_hpa * 100.0)
    temp_at_level, _ = isa(fl_at_level * 30.48)
    n_time = 2
    n_legs = len(legs)
    times = np.array(["2016-01-01T00:00:00", "2026-12-31T00:00:00"], dtype="datetime64[ns]")
    u = np.zeros((n_time, len(levels_hpa), n_legs))
    v = np.zeros((n_time, len(levels_hpa), n_legs))
    t = np.broadcast_to(temp_at_level[None, :, None], (n_time, len(levels_hpa), n_legs)).copy()
    return dict(time=times, level=levels_hpa, u=u, v=v, t=t,
                cum_nm=np.array([leg.cum_nm for leg in legs]),
                track_deg=np.array([leg.track_deg for leg in legs]))


def _synthetic_flight_df(seed=0, zfw_t=92.0, sample_interval_s=15.0):
    """A full KJFK->EGLL synthetic flight (SAMPLE_PLN), still-air/ISA+0
    throughout, built from the real model (search.resolve_tow_and_arrival --
    the same climb/cruise march and arrival.arrival() split `concopt report`
    runs), via build_synthetic_flight."""
    plan = parse_pln(SAMPLE_PLN)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs)
    arrival_legs = [leg for leg, m in zip(legs, mask) if not m]

    data = _still_air_data(np.array([150.0, 125.0, 100.0, 70.0]), legs)
    subsonic_data = _still_air_data(
        np.array([175.0, 200.0, 225.0, 250.0, 300.0, 400.0, 500.0]), arrival_legs)
    arrival_upper_data = _still_air_data(np.array([70.0, 100.0, 125.0, 150.0]), arrival_legs)

    dep_i8 = np.array([1455289200000000000], dtype="int64")  # 2016-02-12 15:00Z
    return build_synthetic_flight(
        SAMPLE_PLN, data, dep_i8, subsonic_data=subsonic_data,
        arrival_upper_data=arrival_upper_data, zfw_t=zfw_t,
        sample_interval_s=sample_interval_s, seed=seed,
    )


# --- pure helpers -------------------------------------------------------------

def test_replay_brake_release_threshold_matches_inflight():
    """replay.py can't import inflight.BRAKE_RELEASE_GS_KT (would create an
    import cycle -- inflight imports concopt.replay's sources at the CLI
    layer), so it keeps its own copy for the ground-roll control points.
    Pinned equal here rather than trusted to stay in sync silently."""
    assert _BRAKE_RELEASE_GS_KT == BRAKE_RELEASE_GS_KT


def test_row_weather_missing_values_become_nan_not_zero():
    """The C2 schema's own rule (a missing reading writes empty, read back
    as NaN, never 0) still has to hold when replay.py is the one reading
    those columns back out -- a missing as_wind_kt must not turn into a
    confident calm."""
    row = pd.Series({"as_wind_dir_deg": np.nan, "as_wind_kt": np.nan,
                      "as_temp_c": np.nan, "as_pressure_hpa": np.nan})
    alt_ft, wind_dir, wind_kt, pressure, temp = _row_weather(row, [45000.0, 50000.0])
    assert list(alt_ft) == [45000.0, 50000.0]
    assert np.all(np.isnan(wind_dir))
    assert np.all(np.isnan(wind_kt))
    assert np.all(np.isnan(pressure))
    assert np.all(np.isnan(temp))


def test_replay_sources_keyed_by_wall_clock_not_call_count():
    """A recording's rows are spaced by real elapsed time, not by how often
    run_inflight happens to call state_source -- the low-altitude tick
    calls it far more often than cruise does, and this still has to answer
    with whatever moment in the flight the (scaled) wall clock says it is,
    not "the next row"."""
    df = pd.DataFrame({
        "elapsed_s": [0.0, 10.0, 20.0],
        "lat_deg": [10.0, 11.0, 12.0], "lon_deg": [0.0, 0.0, 0.0],
        "alt_ft": [0.0, 1000.0, 2000.0], "mach": [0.0, 0.1, 0.2],
        "tas_kt": [0.0, 100.0, 200.0], "gs_kt": [0.0, 100.0, 200.0],
        "weight_t": [180.0, 179.0, 178.0], "on_ground": [1, 0, 0],
        "zulu_s": [0.0, 10.0, 20.0],
    })
    fake_time = {"t": 100.0}
    state_source, _weather_source = replay_sources(df, replay_speed=2.0,
                                                     clock=lambda: fake_time["t"])

    assert state_source()["lat_deg"] == 10.0  # first call anchors t0 -> row 0

    fake_time["t"] = 100.1  # barely any wall time -> still row 0
    assert state_source()["lat_deg"] == 10.0

    fake_time["t"] = 105.0  # 5 s wall * 2.0 speed = 10 s flight -> row 1
    assert state_source()["lat_deg"] == 11.0

    fake_time["t"] = 1000.0  # long past the last row -> holds the last one
    assert state_source()["lat_deg"] == 12.0


def test_replay_sources_rejects_empty_profile():
    with pytest.raises(ValueError):
        replay_sources(pd.DataFrame({"elapsed_s": []}))


# --- build_synthetic_flight ----------------------------------------------------

def test_synthetic_flight_profile_shape():
    profile = _synthetic_flight_df()
    on_ground = profile["on_ground"].to_numpy()
    assert profile["elapsed_s"].is_monotonic_increasing
    assert on_ground[0] == 1
    assert on_ground[-1] == 1
    # exactly two transitions: ground -> air (liftoff), air -> ground (touchdown)
    transitions = np.count_nonzero(np.diff(on_ground) != 0)
    assert transitions == 2
    assert (on_ground[len(on_ground) // 2 - 5:len(on_ground) // 2 + 5] == 0).all()  # airborne mid-flight
    assert profile["cum_nm"].iloc[-1] == pytest.approx(profile["cum_nm"].max())
    assert profile["alt_ft"].max() > 40000.0  # actually reaches a supersonic cruise level


def test_synthetic_flight_noise_is_seed_reproducible_and_seed_sensitive():
    a = _synthetic_flight_df(seed=42)
    b = _synthetic_flight_df(seed=42)
    c = _synthetic_flight_df(seed=43)
    pd.testing.assert_frame_equal(a, b)
    assert not a["mach"].equals(c["mach"])


# --- run_inflight, end to end ---------------------------------------------------

def test_synthetic_flight_replays_end_to_end_with_phases_in_order(tmp_path):
    profile_df = _synthetic_flight_df(seed=1)
    state_source, weather_source = replay_sources(profile_df, replay_speed=_FAST_REPLAY_SPEED)
    record_path = tmp_path / "recording.csv"

    t_start = time.monotonic()
    run_inflight(SAMPLE_PLN, interval_s=60.0, record_path=str(record_path),
                 state_source=state_source, weather_source=weather_source,
                 replay_speed=_FAST_REPLAY_SPEED, live=False)
    wall_s = time.monotonic() - t_start

    recorded = pd.read_csv(record_path)
    assert len(recorded) > 20
    assert recorded["schema_version"].iloc[0] == 2

    phases = recorded["phase"].tolist()
    canonical_order = ["ground", "climb", "cruise", "decel", "descent", "approach"]
    seen_in_canonical_order = [p for p in canonical_order if p in phases]
    first_seen_at = [phases.index(p) for p in seen_in_canonical_order]
    assert first_seen_at == sorted(first_seen_at), (
        f"phases out of order: {seen_in_canonical_order} at {first_seen_at}")
    # cruise is the bulk of a JFK-LHR flight -- if it's missing the phase
    # cut (or the climb model) is badly wrong, not just abbreviated.
    assert "cruise" in phases

    assert recorded["on_ground"].iloc[-1] == 1
    assert wall_s < 60.0, f"replay took {wall_s:.1f}s, wanted well under 60s"


def test_touchdown_detected_and_loop_exits(tmp_path):
    """The loop's only normal exit is the recorder's own touchdown
    detection (on_ground False -> True) -- this is that detection actually
    firing and returning control, not hanging."""
    profile_df = _synthetic_flight_df(seed=2)
    state_source, weather_source = replay_sources(profile_df, replay_speed=_FAST_REPLAY_SPEED)
    record_path = tmp_path / "recording.csv"

    run_inflight(SAMPLE_PLN, interval_s=60.0, record_path=str(record_path),
                 state_source=state_source, weather_source=weather_source,
                 replay_speed=_FAST_REPLAY_SPEED, live=False)

    recorded = pd.read_csv(record_path)
    assert recorded["on_ground"].iloc[-1] == 1
    assert recorded["phase"].iloc[-2] in ("descent", "approach")


def test_no_live_replay_prints_plain_output(tmp_path, capsys):
    profile_df = _synthetic_flight_df(seed=3)
    state_source, weather_source = replay_sources(profile_df, replay_speed=_FAST_REPLAY_SPEED)
    record_path = tmp_path / "recording.csv"

    run_inflight(SAMPLE_PLN, interval_s=60.0, record_path=str(record_path),
                 state_source=state_source, weather_source=weather_source,
                 replay_speed=_FAST_REPLAY_SPEED, live=False)

    out = capsys.readouterr().out
    assert "lat " in out  # _print_advisor's scrolling per-tick line
    assert "Brake release detected" in out
    assert "Touchdown detected" in out


def test_replay_from_recording_reproduces_advisory_sequence(tmp_path):
    """A recorded flight -> --replay -> the SAME advisory calls (rec_action
    per tick), close enough in row count that the two runs walked the same
    flight -- the whole point of turning a recording into a regression
    fixture."""
    profile_df = _synthetic_flight_df(seed=4)

    rec1_path = tmp_path / "rec1.csv"
    s1, w1 = replay_sources(profile_df, replay_speed=_FAST_REPLAY_SPEED)
    run_inflight(SAMPLE_PLN, interval_s=60.0, record_path=str(rec1_path),
                 state_source=s1, weather_source=w1,
                 replay_speed=_FAST_REPLAY_SPEED, live=False)
    rec1 = pd.read_csv(rec1_path)

    rec2_path = tmp_path / "rec2.csv"
    s2, w2 = replay_sources(rec1, replay_speed=_FAST_REPLAY_SPEED)
    run_inflight(SAMPLE_PLN, interval_s=60.0, record_path=str(rec2_path),
                 state_source=s2, weather_source=w2,
                 replay_speed=_FAST_REPLAY_SPEED, live=False)
    rec2 = pd.read_csv(rec2_path)

    # Not bit-identical: two SEPARATE real-time runs, each sleeping through
    # ~200 ticks at compressed speed, accumulate a little real wall-clock
    # scheduler drift between them (observed: a handful of rows' worth by
    # touchdown, out of ~200 -- Windows' ~15 ms timer granularity times
    # replay_speed adds up over that many sleeps). A few rows of drift and
    # an occasional threshold-crossing tick landing on the "wrong" side are
    # both expected; a replay that diverges early or broadly is not, so
    # this tolerates a small fraction of both rather than exact equality
    # across two independently-timed live runs.
    assert abs(len(rec1) - len(rec2)) <= max(6, int(0.05 * max(len(rec1), len(rec2)))), (
        f"row counts {len(rec1)} vs {len(rec2)} diverged more than expected")
    n = min(len(rec1), len(rec2))
    actions1 = rec1["rec_action"].iloc[:n].to_numpy()
    actions2 = rec2["rec_action"].iloc[:n].to_numpy()
    mismatches = int(np.count_nonzero(actions1 != actions2))
    assert mismatches <= max(4, n // 10), (
        f"{mismatches}/{n} advisory actions differed between the recorded "
        "flight and its --replay -- too many to be tick-boundary jitter")


def test_ctrl_c_mid_flight_closes_csv_with_partial_rows(tmp_path):
    """A KeyboardInterrupt raised mid-loop (Ctrl+C while flying) must still
    leave a readable CSV with the rows recorded up to that point -- not a
    partially-flushed or unclosed file."""
    profile_df = _synthetic_flight_df(seed=5)
    real_state_source, weather_source = replay_sources(profile_df, replay_speed=_FAST_REPLAY_SPEED)
    record_path = tmp_path / "recording.csv"

    calls = {"n": 0}

    def flaky_state_source():
        calls["n"] += 1
        if calls["n"] == 40:
            raise KeyboardInterrupt
        return real_state_source()

    run_inflight(SAMPLE_PLN, interval_s=60.0, record_path=str(record_path),
                 state_source=flaky_state_source, weather_source=weather_source,
                 replay_speed=_FAST_REPLAY_SPEED, live=False)

    recorded = pd.read_csv(record_path)
    assert 0 < len(recorded) < 300  # truncated, not a full touchdown-to-touchdown flight
    assert recorded["on_ground"].iloc[-1] != 1 or len(recorded) < 5


def test_live_display_handles_none_values_early_in_flight(tmp_path):
    """rich.Live's redraw, with elapsed_s/predicted_*_s/arrival_info all
    still None (before brake release, before Active Sky's first arrival
    answer) -- must not crash. Interrupted after a few ticks; only the
    early, mostly-None state matters here."""
    profile_df = _synthetic_flight_df(seed=6)
    real_state_source, weather_source = replay_sources(profile_df, replay_speed=5.0)

    calls = {"n": 0}

    def stopping_state_source():
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt
        return real_state_source()

    record_path = tmp_path / "recording.csv"
    run_inflight(SAMPLE_PLN, interval_s=5.0, record_path=str(record_path),
                 state_source=stopping_state_source, weather_source=weather_source,
                 replay_speed=5.0, live=True)
    # No exception -> rich.Live tolerated the early None values.
