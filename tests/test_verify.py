"""Acceptance tests for concopt.verify's pure helpers: point selection,
nearest-point ground-speed grouping, and the Active Sky wind/temp query
(mocked -- no live Active Sky needed).
"""
import numpy as np
import pytest

from concopt import verify
from concopt.search import TARGET_FL


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
