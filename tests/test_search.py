"""Acceptance tests for search.run_shortlist -- the pure CSV-in,
printed-commands-out helper behind `concopt shortlist` -- plus, from B3, a
couple of fast end-to-end acceptance tests for run_search itself (arrival.py
wired in via resolve_tow_and_arrival), on a monkeypatched two-candidate
scan rather than the real ~31,000-row one.
"""
import datetime as dt

import numpy as np
import pandas as pd

from concopt import search
from concopt.search import DEFAULT_TOW_T, run_shortlist
from tests.test_report import _still_air_npz, _still_air_subsonic_npz
from tests.test_route import SAMPLE_PLN


def _sample_results_csv(path, n=15):
    """A minimal concopt search --out CSV -- just the columns run_shortlist
    reads (date, local_departure, tow_t, total_time, mean_fl, mean_wind_kt,
    flags), already sorted best-first like the real output."""
    df = pd.DataFrame({
        "date": [f"2026-01-{i + 1:02d}" for i in range(n)],
        "local_departure": ["14:00"] * n,
        "total_time": ["3:0" + str(i % 10) for i in range(n)],
        "mean_fl": [507] * n,
        "mean_wind_kt": [76.3] * n,
        "flags": [""] * n,
        "tow_t": [185.0] * n,
    })
    df.to_csv(path, index=False)
    return df


def test_run_shortlist_returns_top_n_rows(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    _sample_results_csv(csv_path, n=15)

    out = run_shortlist(csv_path, "route.pln", "route_legs.npz", top=5)

    assert len(out) == 5
    assert out["date"].tolist() == [f"2026-01-{i + 1:02d}" for i in range(5)]


def test_run_shortlist_prints_ready_to_run_verify_command(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    _sample_results_csv(csv_path, n=3)

    run_shortlist(csv_path, "route.pln", "route_legs.npz", top=3, decel_id="BARIX")

    printed = capsys.readouterr().out
    assert "concopt verify --pln route.pln --npz route_legs.npz" in printed
    assert "--date 2026-01-01 --hour 14" in printed
    assert f"--tow {DEFAULT_TOW_T:.0f}" in printed
    assert "--decel BARIX" in printed


def test_run_shortlist_uses_each_row_own_tow(tmp_path, capsys):
    """A shortlist spanning search runs made under different --tow values
    should generate a command matching each row's own tow_t, not a shared
    default."""
    csv_path = tmp_path / "results.csv"
    df = _sample_results_csv(csv_path, n=2)
    df.loc[1, "tow_t"] = 175.0
    df.to_csv(csv_path, index=False)

    run_shortlist(csv_path, "route.pln", "route_legs.npz", top=2)
    printed = capsys.readouterr().out

    assert "--tow 185" in printed
    assert "--tow 175" in printed


def test_run_shortlist_blank_flags_shown_as_dash(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    _sample_results_csv(csv_path, n=1)

    run_shortlist(csv_path, "route.pln", "route_legs.npz", top=1)

    assert "flags: -" in capsys.readouterr().out


def _still_air_surface_npz(tmp_path):
    """A minimal era5.reduce_surface_to_npz-shaped .npz, zero wind/gust at
    both KJFK and EGLL, wide enough in time to bracket any candidate this
    test file uses."""
    times = np.array(["2016-01-01T00:00:00", "2026-12-31T00:00:00"], dtype="datetime64[ns]")
    arrays = {}
    for name in ("KJFK", "EGLL"):
        arrays[f"{name}_time"] = times
        arrays[f"{name}_u10"] = np.zeros(2)
        arrays[f"{name}_v10"] = np.zeros(2)
        arrays[f"{name}_i10fg"] = np.zeros(2)
    path = tmp_path / "surface.npz"
    np.savez(path, **arrays)
    return path


def _tiny_candidates():
    """Two candidate departures, same day -- stands in for
    search.candidate_departures()'s real ~30,933-row scan so
    run_search's own end-to-end tests stay fast; the real scan is what
    `concopt search` runs for real, not what these need to prove the
    arrival.py wiring works."""
    rows = [
        (dt.date(2016, 2, 12), 10,
         np.datetime64(search.local_to_departure_utc(dt.date(2016, 2, 12), 10))),
        (dt.date(2016, 2, 12), 11,
         np.datetime64(search.local_to_departure_utc(dt.date(2016, 2, 12), 11))),
    ]
    return pd.DataFrame(rows, columns=["local_date", "local_hour", "departure_utc"])


def test_run_search_end_to_end_with_zfw_and_subsonic_npz(tmp_path, monkeypatch):
    """search's real path: --zfw + --subsonic-npz, no legacy override --
    the real per-day arrival.arrival() model runs inside the fixed point
    for every candidate, the same machinery test_report.py's run_report
    tests exercise for one candidate at a time."""
    npz_path = _still_air_npz(tmp_path)
    subsonic_npz_path = _still_air_subsonic_npz(tmp_path)
    surface_npz_path = _still_air_surface_npz(tmp_path)
    out_path = tmp_path / "results.csv"

    monkeypatch.setattr(search, "candidate_departures", _tiny_candidates)

    candidates = search.run_search(
        SAMPLE_PLN, npz_path, surface_npz_path, out_path=out_path,
        zfw_t=92.0, min_landing_fuel_t=10.0, subsonic_npz_path=subsonic_npz_path,
    )

    assert len(candidates) == 2
    assert "arrival_time_s" in candidates.columns
    assert "arrival_fuel_t" in candidates.columns
    assert np.all(np.isfinite(candidates["total_time_s"]))
    # Real per-day arrival fuel, not the old flat 2.0 t placeholder.
    assert np.all(candidates["arrival_fuel_t"] > 3.0)
    assert np.all(candidates["arrival_fuel_t"] < 12.0)


def test_run_search_decel_descent_min_reproduces_old_flat_behaviour(tmp_path, monkeypatch):
    """--decel-descent-min forces arrival.flat_arrival for every candidate:
    total_time_s == accumulated_s + the given minutes + runway penalties,
    exactly, no --subsonic-npz needed."""
    npz_path = _still_air_npz(tmp_path)
    surface_npz_path = _still_air_surface_npz(tmp_path)
    out_path = tmp_path / "results.csv"

    monkeypatch.setattr(search, "candidate_departures", _tiny_candidates)

    candidates = search.run_search(
        SAMPLE_PLN, npz_path, surface_npz_path, out_path=out_path,
        tow_t=DEFAULT_TOW_T, decel_descent_min=35.0,
    )

    expect_s = (candidates["supersonic_time_s"] + 35.0 * 60.0
                + candidates["jfk_penalty_s"] + candidates["lhr_penalty_s"])
    assert np.allclose(candidates["total_time_s"], expect_s)
    assert np.allclose(candidates["arrival_time_s"], 35.0 * 60.0)
