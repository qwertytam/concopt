"""Acceptance tests for search.run_shortlist -- the pure CSV-in,
printed-commands-out helper behind `concopt shortlist` -- plus, from B3, a
couple of fast end-to-end acceptance tests for run_search itself (arrival.py
wired in via resolve_tow_and_arrival), on a monkeypatched two-candidate
scan rather than the real ~31,000-row one. From B6, tests for
search._build_arrival_wind_fn's stitched FL183-FL605 wind profile.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from concopt import search
from concopt.atmos import pressure_to_fl
from concopt.era5 import SUBSONIC_LEVELS, UPPER_AIR_LEVELS
from concopt.route import Leg
from concopt.search import DEFAULT_TOW_T, run_shortlist
from tests.test_report import (_still_air_arrival_upper_npz, _still_air_npz,
                                _still_air_subsonic_npz)
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
    arrival_upper_npz_path = _still_air_arrival_upper_npz(tmp_path)
    surface_npz_path = _still_air_surface_npz(tmp_path)
    out_path = tmp_path / "results.csv"

    monkeypatch.setattr(search, "candidate_departures", _tiny_candidates)

    candidates = search.run_search(
        SAMPLE_PLN, npz_path, surface_npz_path, out_path=out_path,
        zfw_t=92.0, min_landing_fuel_t=10.0, subsonic_npz_path=subsonic_npz_path,
        arrival_upper_npz_path=arrival_upper_npz_path,
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


def _arrival_wind_data(n_legs=3, n_time=2):
    """(subsonic_data, arrival_upper_data) dicts shaped like two
    era5.reduce_to_legs .npz loads against the SAME arrival legs -- one at
    era5.SUBSONIC_LEVELS (175-500 hPa, FL183-FL414), one at
    era5.UPPER_AIR_LEVELS (70-150 hPa, FL447-FL605) -- for
    search._build_arrival_wind_fn directly, no real .npz files needed.

    u is set to the level's own pressure in hPa, constant across time/leg --
    monotonic in log(pressure), so any two distinct flight levels resolve to
    distinct interpolated wind, and the exact clamped value at a span edge
    is easy to predict (it's just that edge level's own hPa figure)."""
    times = np.array(["2016-01-01T00:00:00", "2026-12-31T00:00:00"], dtype="datetime64[ns]")
    # Matches _arrival_wind_fn's own legs below -- B8's cum_nm identity check
    # needs a real leg axis here, not just a leg count.
    cum_nm = np.array([50.0 * (i + 1) for i in range(n_legs)])

    def _fake(levels_hpa):
        n_lvl = len(levels_hpa)
        u = np.broadcast_to(
            levels_hpa[None, :, None], (n_time, n_lvl, n_legs)
        ).astype(float).copy()
        v = np.zeros((n_time, n_lvl, n_legs))
        t = np.full((n_time, n_lvl, n_legs), 220.0)
        return dict(time=times, level=levels_hpa, u=u, v=v, t=t, cum_nm=cum_nm)

    subsonic_data = _fake(np.array([float(x) for x in SUBSONIC_LEVELS]))
    arrival_upper_data = _fake(np.array([float(x) for x in UPPER_AIR_LEVELS]))
    return subsonic_data, arrival_upper_data


def _arrival_wind_fn(subsonic_data, arrival_upper_data, n_legs=3, n_cand=1):
    legs = [
        Leg(f"A{i}", f"B{i}", 45.0, -30.0 + i, 90.0, 50.0, 50.0 * (i + 1))
        for i in range(n_legs)
    ]
    dep_i8 = np.full(n_cand, subsonic_data["time"][0].astype("int64"))
    builder = search._build_arrival_wind_fn(subsonic_data, arrival_upper_data, dep_i8, legs)
    return builder(np.zeros(n_cand))


class TestArrivalWindStitching:
    """B6: subsonic_data alone (era5.SUBSONIC_LEVELS, FL183-FL414) doesn't
    reach the decel segment's own wind-sampling midpoint from a realistic
    cruise level (roughly FL446-491) -- _build_arrival_wind_fn stitches
    arrival_upper_data (era5.UPPER_AIR_LEVELS, FL447-FL605) on top."""

    def test_stitched_span_is_about_fl183_to_fl605(self):
        subsonic_data, arrival_upper_data = _arrival_wind_data()
        wind_at_fl = _arrival_wind_fn(subsonic_data, arrival_upper_data)

        fl_min = pressure_to_fl(500.0 * 100.0)  # bottom of SUBSONIC_LEVELS
        fl_max = pressure_to_fl(70.0 * 100.0)   # top of UPPER_AIR_LEVELS
        assert fl_min == pytest.approx(182.9, abs=0.1)
        assert fl_max == pytest.approx(605.0, abs=0.1)

        assert not wind_at_fl(np.array([fl_min]))["fl_clamped"][0]
        assert not wind_at_fl(np.array([fl_max]))["fl_clamped"][0]
        assert wind_at_fl(np.array([fl_min - 5.0]))["fl_clamped"][0]
        assert wind_at_fl(np.array([fl_max + 5.0]))["fl_clamped"][0]

    def test_fl480_differs_from_fl414(self):
        """The pre-fix bug: FL480 (above SUBSONIC_LEVELS' own FL414 top)
        clamped to FL414's own value, silently, because subsonic_data alone
        was the whole span. With arrival_upper_data stitched on, FL480 sits
        genuinely inside the (now wider) unclamped span and must read a
        different wind than FL414."""
        subsonic_data, arrival_upper_data = _arrival_wind_data()
        wind_at_fl = _arrival_wind_fn(subsonic_data, arrival_upper_data)

        at_414 = wind_at_fl(np.array([414.0]))
        at_480 = wind_at_fl(np.array([480.0]))

        assert not at_414["fl_clamped"][0]
        assert not at_480["fl_clamped"][0]
        assert at_480["wind_kt"][0] != pytest.approx(at_414["wind_kt"][0])

    def test_wind_fl_clamped_false_at_fl450_true_at_fl150(self):
        subsonic_data, arrival_upper_data = _arrival_wind_data()
        wind_at_fl = _arrival_wind_fn(subsonic_data, arrival_upper_data)

        assert not wind_at_fl(np.array([450.0]))["fl_clamped"][0]
        assert wind_at_fl(np.array([150.0]))["fl_clamped"][0]

    def test_mismatched_time_axes_raise(self):
        """A silent time misalignment between the two stitched level sets
        would be this bug's own twin -- assert, don't assume, they match."""
        subsonic_data, arrival_upper_data = _arrival_wind_data()
        arrival_upper_data = dict(arrival_upper_data)
        arrival_upper_data["time"] = np.array(
            ["2015-06-01T00:00:00", "2027-06-01T00:00:00"], dtype="datetime64[ns]"
        )
        dep_i8 = np.full(1, subsonic_data["time"][0].astype("int64"))
        legs = [Leg("A", "B", 45.0, -30.0, 90.0, 50.0, 50.0)]

        with pytest.raises(ValueError, match="time"):
            search._build_arrival_wind_fn(subsonic_data, arrival_upper_data, dep_i8, legs)

    def test_wrong_leg_axis_raises_with_leg_count(self):
        """B8: passing a cruise-leg npz as --arrival-upper-npz (the same
        era5_upper_*.nc files as the cruise --npz, just never re-reduced
        onto arrival_legs) must raise, not silently index a leg axis it was
        never built against -- a ~32-leg cruise axis happily accepts a small
        positional index like len(arrival_legs)//2 and would otherwise read
        mid-Atlantic wind for the arrival segment."""
        n_legs = 3
        subsonic_data, arrival_upper_data = _arrival_wind_data(n_legs=n_legs)
        # Stand in for a cruise-leg npz: same shape family, but a leg axis
        # (and cum_nm) sized like the ~32-leg cruise span, not arrival_legs.
        arrival_upper_data = dict(arrival_upper_data)
        n_cruise_legs = 32
        arrival_upper_data["u"] = np.zeros((2, arrival_upper_data["u"].shape[1], n_cruise_legs))
        arrival_upper_data["v"] = np.zeros((2, arrival_upper_data["v"].shape[1], n_cruise_legs))
        arrival_upper_data["t"] = np.full((2, arrival_upper_data["t"].shape[1], n_cruise_legs), 220.0)
        arrival_upper_data["cum_nm"] = np.array([10.0 * (i + 1) for i in range(n_cruise_legs)])

        dep_i8 = np.full(1, subsonic_data["time"][0].astype("int64"))
        legs = [
            Leg(f"A{i}", f"B{i}", 45.0, -30.0 + i, 90.0, 50.0, 50.0 * (i + 1))
            for i in range(n_legs)
        ]

        with pytest.raises(ValueError, match=f"{n_cruise_legs}.*{n_legs}|{n_legs}.*{n_cruise_legs}"):
            search._build_arrival_wind_fn(subsonic_data, arrival_upper_data, dep_i8, legs)
