"""Acceptance tests for concopt.verify's pure helpers: point selection,
nearest-point ground-speed grouping, the Active Sky wind/temp query, the
snapshot guard, and the level-mismatch TAS cost (all mocked -- no live
Active Sky needed).
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from concopt import verify
from concopt.atmos import isa
from concopt.era5 import load_legs_npz
from concopt.route import build_legs, climb_cruise_segment, parse_pln
from concopt.search import (DEFAULT_TOW_T, TARGET_FL, local_to_departure_utc,
                             resolve_tow_and_arrival)
from tests.test_report import (_still_air_arrival_upper_npz, _still_air_npz,
                                _still_air_subsonic_npz)
from tests.test_route import SAMPLE_PLN, SAMPLE_DECEL_ID


def test_select_points_includes_both_ends():
    idx = verify._select_points(n_legs=30, n_points=6)
    assert idx[0] == 0
    assert idx[-1] == 29
    assert len(idx) == 6
    assert np.all(np.diff(idx) > 0)  # sorted, deduplicated


def test_select_points_clamps_to_n_legs():
    """More points requested than legs available -> every leg, no
    duplicates, no out-of-range index."""
    idx = verify._select_points(n_legs=4, n_points=10)
    assert idx.tolist() == [0, 1, 2, 3]


def test_select_points_single_point_is_first_leg():
    idx = verify._select_points(n_legs=10, n_points=1)
    assert idx.tolist() == [0]


def test_group_gs_ms_assigns_by_nearest_point():
    """Legs nearer point 0 than point 5 get point 0's ground speed, and
    vice versa; the split falls at the midpoint."""
    point_idx = np.array([0, 5])
    gs_at_points = np.array([100.0, 200.0])

    gs_per_leg = verify._group_gs_ms(point_idx, gs_at_points, n_legs=6)

    assert gs_per_leg.tolist() == [100.0, 100.0, 100.0, 200.0, 200.0, 200.0]


def test_group_gs_ms_single_point_covers_every_leg():
    gs_per_leg = verify._group_gs_ms(np.array([3]), np.array([150.0]), n_legs=7)
    assert gs_per_leg.tolist() == [150.0] * 7


class _FakeAtmosphere:
    """Stand-in for asky.get_atmosphere_np, keyed by requested altitude."""
    def __init__(self, by_alt_ft, permute=False):
        self.by_alt_ft = by_alt_ft
        self.permute = permute

    def __call__(self, lat, lon, alts_ft, host_addr="localhost", port=19285):
        rows = [self.by_alt_ft[a] for a in alts_ft]
        if self.permute:
            rows = rows[::-1]
            alts_ft = list(alts_ft)[::-1]
        alt_ft = np.array(alts_ft, dtype=float)
        wind_dir_deg = np.array([r[0] for r in rows], dtype=float)
        wind_speed_kt = np.array([r[1] for r in rows], dtype=float)
        pressure_hpa = np.full(len(rows), 200.0)
        temp_c = np.array([r[2] for r in rows], dtype=float)
        return alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c


def test_as_atmosphere_converts_from_bearing_to_uv(monkeypatch):
    """A due-north wind (FROM 360/0 deg) blows toward the south: u ~ 0,
    v < 0 -- the standard meteorological FROM-bearing inversion."""
    by_alt = {ft: (0.0, 50.0, -56.0) for ft in TARGET_FL * 100.0}
    monkeypatch.setattr(verify, "get_atmosphere_np", _FakeAtmosphere(by_alt))

    temp_k, u_ms, v_ms = verify._as_atmosphere(40.0, -30.0, "localhost", 19285)

    assert np.allclose(u_ms, 0.0, atol=1e-6)
    assert np.all(v_ms < 0.0)
    assert temp_k[0] == pytest.approx(-56.0 + 273.15)


def test_as_atmosphere_reindexes_by_altitude_not_position(monkeypatch):
    """Active Sky's response order doesn't matter -- results are aligned
    onto TARGET_FL by altitude."""
    by_alt = {ft: (90.0, 10.0 + i, -55.0) for i, ft in enumerate(TARGET_FL * 100.0)}
    monkeypatch.setattr(verify, "get_atmosphere_np", _FakeAtmosphere(by_alt, permute=True))

    temp_k, u_ms, v_ms = verify._as_atmosphere(40.0, -30.0, "localhost", 19285)

    # Wind FROM 90 (due east) blows toward the west: u < 0, magnitude == speed.
    expected_u_ms = -(10.0 + np.arange(len(TARGET_FL))) * (1852.0 / 3600.0)
    assert np.allclose(u_ms, expected_u_ms, atol=1e-6)


def test_as_atmosphere_raises_on_altitude_mismatch(monkeypatch):
    """Active Sky returning altitudes that don't match the requested
    TARGET_FL grid is a clear error, not a silent misalignment."""
    by_alt = {ft: (0.0, 10.0, -55.0) for ft in TARGET_FL * 100.0}
    fake = _FakeAtmosphere(by_alt)

    def _wrong_alt(lat, lon, alts_ft, host_addr="localhost", port=19285):
        alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c = fake(
            lat, lon, alts_ft, host_addr, port)
        alt_ft = alt_ft + 50.0  # nudge every altitude off the requested grid
        return alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c

    monkeypatch.setattr(verify, "get_atmosphere_np", _wrong_alt)

    with pytest.raises(RuntimeError, match="can't align"):
        verify._as_atmosphere(40.0, -30.0, "localhost", 19285)


def _sample_atmospheres(n=3, seed=0):
    """n synthetic (temp_k, u_ms, v_ms) triples, each (len(TARGET_FL),) --
    stand-ins for what _as_atmosphere would return per queried point."""
    rng = np.random.default_rng(seed)
    return [
        (rng.uniform(210.0, 220.0, len(TARGET_FL)),
         rng.uniform(-50.0, 50.0, len(TARGET_FL)),
         rng.uniform(-50.0, 50.0, len(TARGET_FL)))
        for _ in range(n)
    ]


def test_atmosphere_fingerprint_deterministic():
    atmospheres = _sample_atmospheres()
    assert verify._atmosphere_fingerprint(atmospheres) == verify._atmosphere_fingerprint(atmospheres)


def test_atmosphere_fingerprint_differs_on_different_data():
    fp1 = verify._atmosphere_fingerprint(_sample_atmospheres(seed=1))
    fp2 = verify._atmosphere_fingerprint(_sample_atmospheres(seed=2))
    assert fp1 != fp2


def test_guard_snapshot_allows_first_run(tmp_path):
    cache_path = tmp_path / "cache.json"
    verify._guard_snapshot("abc123", dt.date(2016, 2, 12), 14, cache_path)
    assert cache_path.exists()


def test_guard_snapshot_allows_repeat_of_same_date(tmp_path):
    """Re-verifying the same date/hour without reloading Active Sky is a
    legitimate thing to do -- an identical fingerprint under the SAME key
    is not an error."""
    cache_path = tmp_path / "cache.json"
    verify._guard_snapshot("abc123", dt.date(2016, 2, 12), 14, cache_path)
    verify._guard_snapshot("abc123", dt.date(2016, 2, 12), 14, cache_path)  # no raise


def test_guard_snapshot_raises_on_stale_load(tmp_path):
    """The actual bug this guards against: a second date's run returns the
    same fingerprint as an earlier, different date's run -- Active Sky
    wasn't reloaded."""
    cache_path = tmp_path / "cache.json"
    verify._guard_snapshot("abc123", dt.date(2016, 2, 12), 14, cache_path)

    with pytest.raises(RuntimeError, match="2016-02-12 14:00.*wasn't reloaded"):
        verify._guard_snapshot("abc123", dt.date(2016, 1, 6), 11, cache_path)


def test_guard_snapshot_different_weather_does_not_raise(tmp_path):
    cache_path = tmp_path / "cache.json"
    verify._guard_snapshot("abc123", dt.date(2016, 2, 12), 14, cache_path)
    verify._guard_snapshot("def456", dt.date(2016, 1, 6), 11, cache_path)  # no raise


def test_mismatch_tas_cost_zero_when_levels_agree():
    temp_k_grid = np.full(len(TARGET_FL), 220.0)
    cost = verify._mismatch_tas_cost_kt(3, 3, temp_k_grid, 135.0, 2.00)
    assert cost == 0.0


def test_mismatch_tas_cost_positive_when_levels_differ():
    isa_t, _ = isa(TARGET_FL * 100.0 * 0.3048)
    cost = verify._mismatch_tas_cost_kt(0, 5, isa_t, 135.0, 2.00)
    assert cost > 0.0


def test_mismatch_tas_cost_symmetric():
    isa_t, _ = isa(TARGET_FL * 100.0 * 0.3048)
    assert verify._mismatch_tas_cost_kt(2, 6, isa_t, 135.0, 2.00) == pytest.approx(
        verify._mismatch_tas_cost_kt(6, 2, isa_t, 135.0, 2.00)
    )


def test_mismatch_tas_cost_bigger_near_cas_knee_than_above_it():
    """A mismatch low in the CAS-limited part of the envelope should cost
    more than one confined to the top of the grid, where cruise_mach alone
    binds and TAS is close to flat with altitude."""
    isa_t, _ = isa(TARGET_FL * 100.0 * 0.3048)
    low_cost = verify._mismatch_tas_cost_kt(0, 2, isa_t, 165.0, 2.00)  # FL450 vs FL470
    high_cost = verify._mismatch_tas_cost_kt(10, 12, isa_t, 165.0, 2.00)  # FL550 vs FL570
    assert low_cost > high_cost


def _still_air_data(legs):
    """A minimal era5-shaped dict, ISA+0 and zero wind -- same construction
    test_perf_table.py's still-air fixture uses, reused here so
    run_verify's snapshot guard can be exercised end to end without a real
    .npz."""
    levels_hpa = np.array([150.0, 125.0, 100.0, 70.0])
    from concopt.atmos import pressure_to_fl
    fl_at_level = pressure_to_fl(levels_hpa * 100.0)
    temp_at_level, _ = isa(fl_at_level * 30.48)

    n_time = 2
    n_legs_total = len(legs)
    times = np.array(["2016-01-01T00:00:00", "2026-12-31T00:00:00"], dtype="datetime64[ns]")
    u = np.zeros((n_time, 4, n_legs_total))
    v = np.zeros((n_time, 4, n_legs_total))
    t = np.broadcast_to(temp_at_level[None, :, None], (n_time, 4, n_legs_total)).copy()

    return dict(time=times, level=levels_hpa, u=u, v=v, t=t,
                cum_nm=np.array([leg.cum_nm for leg in legs]),
                track_deg=np.array([leg.track_deg for leg in legs]))


class _ConstantAtmosphere:
    """Stand-in for asky.get_atmosphere_np returning the SAME atmosphere
    regardless of lat/lon -- reproduces the live bug: Active Sky returning
    an earlier historical load's data because the date wasn't reloaded."""
    def __call__(self, lat, lon, alts_ft, host_addr="localhost", port=19285):
        alt_ft = np.asarray(alts_ft, dtype=float)
        wind_dir_deg = np.full(len(alt_ft), 270.0)
        wind_speed_kt = np.full(len(alt_ft), 80.0)
        pressure_hpa = np.full(len(alt_ft), 200.0)
        temp_c = np.full(len(alt_ft), -55.0)
        return alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c


def test_run_verify_snapshot_guard_end_to_end(monkeypatch, tmp_path):
    """The actual failure mode this feature exists for: Active Sky not
    reloaded between two verify runs for different dates returns identical
    weather both times -- the second run must refuse to proceed."""
    monkeypatch.setattr(verify, "get_atmosphere_np", _ConstantAtmosphere())

    plan = parse_pln(SAMPLE_PLN)
    legs = build_legs(plan["waypoints"])
    data = _still_air_data(legs)
    npz_path = tmp_path / "route_legs.npz"
    np.savez(npz_path, **data)

    cache_path = tmp_path / "cache.json"

    verify.run_verify(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 14, decel_id=SAMPLE_DECEL_ID,
                       n_points=2, zfw_t=92.0, tow_t=DEFAULT_TOW_T,
                       snapshot_cache_path=cache_path)

    with pytest.raises(RuntimeError, match="wasn't reloaded"):
        verify.run_verify(SAMPLE_PLN, npz_path, dt.date(2016, 1, 6), 11, decel_id=SAMPLE_DECEL_ID,
                           n_points=2, zfw_t=92.0, tow_t=DEFAULT_TOW_T,
                       snapshot_cache_path=cache_path)


def test_run_verify_zfw_without_subsonic_npz_raises(monkeypatch, tmp_path):
    """--zfw needs the SAME fixed point search --zfw uses, which needs
    subsonic wind data -- a clear failure, not a silently-wrong TOW."""
    monkeypatch.setattr(verify, "get_atmosphere_np", _ConstantAtmosphere())

    npz_path = _still_air_npz(tmp_path)
    cache_path = tmp_path / "cache.json"

    with pytest.raises(ValueError, match="subsonic_npz_path"):
        verify.run_verify(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 14, decel_id=SAMPLE_DECEL_ID,
                           n_points=2, zfw_t=92.0, snapshot_cache_path=cache_path)


def test_run_verify_zfw_without_arrival_upper_npz_raises(monkeypatch, tmp_path):
    """--zfw needs BOTH --subsonic-npz and --arrival-upper-npz (B6) -- given
    only the first, this must still fail loud rather than silently stitching
    against a missing second level set."""
    monkeypatch.setattr(verify, "get_atmosphere_np", _ConstantAtmosphere())

    npz_path = _still_air_npz(tmp_path)
    subsonic_npz_path = _still_air_subsonic_npz(tmp_path)
    cache_path = tmp_path / "cache.json"

    with pytest.raises(ValueError, match="arrival_upper_npz_path"):
        verify.run_verify(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 14, decel_id=SAMPLE_DECEL_ID,
                           n_points=2, zfw_t=92.0, snapshot_cache_path=cache_path,
                           subsonic_npz_path=subsonic_npz_path)


def test_verify_zfw_produces_same_tow_as_search_zfw(monkeypatch, tmp_path):
    """verify --zfw must run the SAME fixed point search --zfw does (via
    the shared search.resolve_tow_and_arrival), so an Active Sky check is
    flown at the weight that day was actually found under -- the whole
    point of B3's part 4. Spies on verify's own call to
    resolve_tow_and_arrival to read the TOW it actually used internally,
    then reproduces search's own call independently and checks they agree."""
    real_resolve = verify.resolve_tow_and_arrival
    captured = {}

    def _spy(*args, **kwargs):
        result = real_resolve(*args, **kwargs)
        captured["tow_t"] = result[0]
        return result

    monkeypatch.setattr(verify, "resolve_tow_and_arrival", _spy)
    monkeypatch.setattr(verify, "get_atmosphere_np", _ConstantAtmosphere())

    npz_path = _still_air_npz(tmp_path)
    subsonic_npz_path = _still_air_subsonic_npz(tmp_path)
    arrival_upper_npz_path = _still_air_arrival_upper_npz(tmp_path)
    cache_path = tmp_path / "cache.json"

    verify.run_verify(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 14, decel_id=SAMPLE_DECEL_ID,
                       n_points=2, zfw_t=92.0, min_landing_fuel_t=10.0,
                       subsonic_npz_path=subsonic_npz_path,
                       arrival_upper_npz_path=arrival_upper_npz_path,
                       snapshot_cache_path=cache_path)

    verify_tow = captured["tow_t"]
    assert verify_tow is not None

    # Independently reproduce search's own fixed point for the same day/data
    # (not read back from verify's spy -- a fresh call, same construction
    # run_search itself does).
    plan = parse_pln(SAMPLE_PLN)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs, SAMPLE_DECEL_ID)
    cc_idx = np.flatnonzero(mask)
    cc_legs = [legs[i] for i in cc_idx]
    arrival_idx = np.flatnonzero(~mask)
    arrival_legs = [legs[i] for i in arrival_idx]
    arrival_nm = legs[-1].cum_nm - cc_legs[-1].cum_nm

    data = load_legs_npz(npz_path)
    subsonic_data = load_legs_npz(subsonic_npz_path)
    arrival_upper_data = load_legs_npz(arrival_upper_npz_path)
    departure_utc = local_to_departure_utc(dt.date(2016, 2, 12), 14)
    dep_i8 = np.array([pd.Timestamp(departure_utc).value], dtype="int64")

    search_tow, *_ = resolve_tow_and_arrival(
        cc_legs, cc_idx, arrival_legs, arrival_nm, data, subsonic_data, dep_i8,
        zfw_t=92.0, min_landing_fuel_t=10.0,
        arrival_upper_data=arrival_upper_data,
    )

    assert np.allclose(verify_tow, search_tow)
