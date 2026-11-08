"""The bridge (Windows side) and the remote state_source (advisor side),
exercised together over real localhost HTTP with no sim: replay.py's
synthetic flight stands in for P3D."""
import json
import threading
import time

import pandas as pd
import pytest

from concopt import bridge, inflight
from concopt.inflight import run_inflight
from concopt.remote import STATE_KEYS, remote_state_source
from concopt.replay import replay_sources
from tests.test_replay import _synthetic_flight_df
from tests.test_route import SAMPLE_ACCEL_ID, SAMPLE_DECEL_ID, SAMPLE_PLN

_FAST = 900.0


class _Served:
    """A real bridge handler on a free localhost port, serving read_state."""

    def __init__(self, read_state):
        self.server = bridge.make_server(read_state, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.server_port}/state"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _state_with_nones():
    """None in required (weight_t, zulu_s is also legitimately None-able) and
    optional (vs_fpm, eng2_n2) fields, as _read_state/_read_n2 produce them."""
    state = dict.fromkeys(STATE_KEYS)
    state.update(lat_deg=40.64, lon_deg=-73.78, alt_ft=51000.0, mach=2.0, tas_kt=1150.0,
                 gs_kt=1200.5, weight_t=None, on_ground=False, zulu_s=45296.0,
                 fuel_kg=80000.0, vs_fpm=None, eng1_n2=95.0, eng2_n2=None)
    return state


# --- the one that matters ---------------------------------------------------------

def test_none_survives_the_round_trip_whole_dict_equal():
    original = _state_with_nones()
    assert original["weight_t"] is None and original["vs_fpm"] is None
    assert json.loads(json.dumps(original)) == original
    with _Served(lambda: original) as served:
        received = remote_state_source(served.url)()
    assert received == original          # whole dict, Nones included
    assert received["weight_t"] is None  # not 0, not dropped


# --- remote source against the real handler ---------------------------------------

def test_remote_source_returns_the_served_dict():
    known = _state_with_nones()
    known.update(weight_t=150.25, vs_fpm=-12.5)
    with _Served(lambda: known) as served:
        assert remote_state_source(served.url)() == known


def test_handler_serialises_none_numpy_and_nan_without_raising():
    import numpy as np
    state = _state_with_nones()
    state.update(mach=np.float64(2.0), on_ground=np.bool_(False), vs_fpm=np.float32(1.5),
                 cas_kt=float("nan"))
    with _Served(lambda: state) as served:
        got = remote_state_source(served.url)()
    assert got["weight_t"] is None and got["mach"] == 2.0 and got["on_ground"] is False
    assert got["cas_kt"] != got["cas_kt"]  # NaN stays NaN, not 0 or null


def test_read_state_keys_match_the_contract():
    class _Aq:
        def get(self, name):
            return 1.0

    keys = set(inflight._read_state(_Aq())) | set(inflight._read_n2(None))
    assert keys == set(STATE_KEYS)


def test_bridge_state_comes_from_inflights_own_reads_with_fsuipc_absent():
    """Unit conversion is _read_state's alone: 2204.62 lb of TOTAL_WEIGHT is
    1.0 t however it got here. FSUIPC None -> N2 null, still served."""
    class _Aq:
        def get(self, name):
            return {"TOTAL_WEIGHT": 2204.62262, "SIM_ON_GROUND": 0}.get(name, 1.0)

    with _Served(inflight._live_state_source(_Aq(), None)) as served:
        got = remote_state_source(served.url)()
    assert got["weight_t"] == pytest.approx(1.0, rel=1e-6)
    assert [got[f"eng{i}_n2"] for i in (1, 2, 3, 4)] == [None] * 4


def test_unknown_path_is_404():
    import requests
    with _Served(_state_with_nones) as served:
        assert requests.get(served.url.replace("/state", "/nope"), timeout=2).status_code == 404


# --- failed polls present as dropped reads ---------------------------------------

def test_failed_polls_raise_runtime_error():
    """RuntimeError is what run_inflight's per-tick except already catches."""
    def boom():
        raise OSError("SimConnect went away")

    with _Served(boom) as served:  # 500
        with pytest.raises(RuntimeError, match="HTTP 500"):
            remote_state_source(served.url)()
    with pytest.raises(RuntimeError, match="did not answer"):  # server now closed
        remote_state_source(served.url, timeout_s=0.5)()

    def slow():
        time.sleep(0.6)
        return _state_with_nones()

    with _Served(slow) as served:  # timeout
        with pytest.raises(RuntimeError, match="did not answer"):
            remote_state_source(served.url, timeout_s=0.2)()
    with _Served(lambda: {"lat_deg": 1.0}) as served:  # not the contract
        with pytest.raises(RuntimeError, match="missing"):
            remote_state_source(served.url)()


def test_run_inflight_skips_ticks_on_failed_polls_and_finishes(tmp_path, capsys):
    """500s and a timeout in the middle of the synthetic flight: the loop
    prints its notice and carries on, instead of crashing."""
    state_source, _ = replay_sources(_synthetic_flight_df(seed=4), replay_speed=_FAST)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] in (3, 4):
            raise OSError("dropped")
        if calls["n"] == 5:
            time.sleep(0.5)
        return state_source()

    # weather still comes from the replay; only state crosses the bridge
    _, weather_source = replay_sources(_synthetic_flight_df(seed=4), replay_speed=_FAST)
    record_path = tmp_path / "recording.csv"
    with _Served(flaky) as served:
        run_inflight(SAMPLE_PLN, SAMPLE_ACCEL_ID, SAMPLE_DECEL_ID, interval_s=60.0,
                     record_path=str(record_path),
                     state_source=remote_state_source(served.url, timeout_s=0.2),
                     weather_source=weather_source, replay_speed=_FAST, live=False)

    assert capsys.readouterr().out.count("No usable data this tick") >= 3
    recorded = pd.read_csv(record_path)
    assert len(recorded) > 20 and recorded["on_ground"].iloc[-1] == 1


def test_whole_chain_synthetic_flight_over_http(tmp_path):
    """replay -> bridge handler -> HTTP -> remote source -> run_inflight."""
    df = _synthetic_flight_df(seed=5)
    state_source, weather_source = replay_sources(df, replay_speed=_FAST)
    record_path = tmp_path / "recording.csv"
    with _Served(state_source) as served:
        run_inflight(SAMPLE_PLN, SAMPLE_ACCEL_ID, SAMPLE_DECEL_ID, interval_s=60.0,
                     record_path=str(record_path),
                     state_source=remote_state_source(served.url),
                     weather_source=weather_source, replay_speed=_FAST, live=False)
    recorded = pd.read_csv(record_path)
    phases = recorded["phase"].tolist()
    assert "cruise" in phases and recorded["on_ground"].iloc[-1] == 1
