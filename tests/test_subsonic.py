"""Tests for the subsonic (M0.95) arrival cruise table:
conc_data.subsonic_cruise, trilinear over (level_fl, mass_t, isa_dev_c).
B5.
"""
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from concopt import atmos
from concopt.data import conc_data


@pytest.fixture
def subsonic_csv():
    """Load conc_subsonic_cruise.csv directly for reference."""
    fp = files("concopt").joinpath("data/conc_subsonic_cruise.csv")
    return pd.read_csv(fp, encoding="utf-8-sig")


class TestRowCount:
    def test_row_count(self, subsonic_csv):
        assert len(subsonic_csv) == 751


class TestMach:
    def test_constant_095_throughout(self, subsonic_csv):
        assert (subsonic_csv["mach"] == 0.95).all()


class TestExactHits:
    """Exact grid hits return the table value, even where a ragged
    neighbour is NaN."""

    def test_exact_hit_fl350_mass110_isa0(self):
        # conc_subsonic_cruise.csv, level_fl=350, mass_t=110, isa_dev_c=0
        r = conc_data.subsonic_cruise(350.0, 110.0, 0.0)
        assert r["fuel_total_kgh"] == pytest.approx(11092.0)
        assert r["specific_range_nm_per_t"] == pytest.approx(49.40)

    def test_exact_hit_with_nan_neighbour(self):
        """FL410, mass 125 t is only published for isa_dev_c up to 10 --
        isa=15/20 are NaN gaps. An exact hit at isa=10 must not be
        poisoned by that neighbour."""
        r = conc_data.subsonic_cruise(410.0, 125.0, 10.0)
        assert r["specific_range_nm_per_t"] == pytest.approx(36.14)
        assert r["fuel_total_kgh"] == pytest.approx(15412.0)

    def test_exact_hit_at_table_corner(self):
        # level_fl=290, mass_t=180, isa_dev_c=-20 -- the table's own corner.
        r = conc_data.subsonic_cruise(290.0, 180.0, -20.0)
        assert r["fuel_total_kgh"] == pytest.approx(18212.0)
        assert r["specific_range_nm_per_t"] == pytest.approx(29.49)

    def test_every_published_row_round_trips(self, subsonic_csv):
        """Every one of the 751 rows, read back exactly."""
        level_fl = subsonic_csv["level_fl"].to_numpy(float)
        mass_t = subsonic_csv["mass_t"].to_numpy(float)
        isa_dev_c = subsonic_csv["isa_dev_c"].to_numpy(float)
        r = conc_data.subsonic_cruise(level_fl, mass_t, isa_dev_c)
        assert np.allclose(r["fuel_total_kgh"], subsonic_csv["fuel_total_kgh"])
        assert np.allclose(
            r["specific_range_nm_per_t"], subsonic_csv["specific_range_nm_per_t"]
        )


class TestMidpointInterpolation:
    def test_midpoint_isa_interpolates_to_the_chord(self):
        """FL350, mass 110 t, isa 0 -> 49.40 nm/t (11092 kg/h); isa 5 ->
        49.36 nm/t (11224 kg/h). The midpoint (isa 2.5) must land on the
        straight chord between them, not the table's own curve."""
        lo = conc_data.subsonic_cruise(350.0, 110.0, 0.0)
        hi = conc_data.subsonic_cruise(350.0, 110.0, 5.0)
        mid = conc_data.subsonic_cruise(350.0, 110.0, 2.5)
        assert mid["specific_range_nm_per_t"] == pytest.approx(
            (lo["specific_range_nm_per_t"] + hi["specific_range_nm_per_t"]) / 2
        )
        assert mid["fuel_total_kgh"] == pytest.approx(
            (lo["fuel_total_kgh"] + hi["fuel_total_kgh"]) / 2
        )


class TestSpecificRangeAtArrivalMasses:
    def test_range_41_to_52_nm_per_t_at_fl310_390_105_120t(self, subsonic_csv):
        sub = subsonic_csv[
            subsonic_csv["level_fl"].between(310, 390)
            & subsonic_csv["mass_t"].between(105, 120)
        ]
        assert len(sub) > 0
        assert sub["specific_range_nm_per_t"].min() >= 41.0
        assert sub["specific_range_nm_per_t"].max() <= 52.1


class TestRaggedEnvelope:
    """A ragged-grid lookup outside the published envelope returns NaN, not
    a value silently clamped to a mass the aircraft cannot hold there."""

    def test_mass_too_heavy_for_level(self):
        # FL410 tops out at 125 t; the global mass axis runs to 180.
        r = conc_data.subsonic_cruise(410.0, 150.0, 0.0)
        assert np.isnan(r["specific_range_nm_per_t"])
        assert np.isnan(r["fuel_total_kgh"])

    def test_missing_isa_cell_at_ragged_corner(self):
        # FL410, mass 125 t: isa_dev_c=20 has no row (only up to 10 is
        # published there).
        r = conc_data.subsonic_cruise(410.0, 125.0, 20.0)
        assert np.isnan(r["specific_range_nm_per_t"])

    def test_low_level_floor_is_110t_not_100t(self):
        """FL290-330 are only published down to 110 t -- unlike FL350+,
        which goes to 100 t. A query at 105 t there must be NaN, not
        clamped to the table's global 100 t floor."""
        r = conc_data.subsonic_cruise(310.0, 105.0, 0.0)
        assert np.isnan(r["specific_range_nm_per_t"])

    def test_not_clamped_to_a_published_neighbour(self):
        """The NaN must be a genuine NaN, not silently substituted with the
        nearest in-envelope value."""
        outside = conc_data.subsonic_cruise(410.0, 150.0, 0.0)["specific_range_nm_per_t"]
        nearest_published = conc_data.subsonic_cruise(410.0, 125.0, 0.0)["specific_range_nm_per_t"]
        assert np.isnan(outside)
        assert not np.isnan(nearest_published)

    def test_mass_within_global_bounds_clamps_not_nans(self):
        """A mass beyond the table's own global axis (>180 t) clamps to
        180 t like every other loader in conc_data.py -- it is only a
        per-level/per-ISA gap inside the published axis range that returns
        NaN."""
        clamped = conc_data.subsonic_cruise(310.0, 500.0, 0.0)
        at_180 = conc_data.subsonic_cruise(310.0, 180.0, 0.0)
        assert clamped["specific_range_nm_per_t"] == pytest.approx(
            at_180["specific_range_nm_per_t"]
        )


class TestTas:
    """tas_kt is computed from atmos.py, not read off the (mass-independent)
    table column -- see subsonic_cruise's docstring."""

    def test_matches_atmos_speed_of_sound(self):
        isa_t_k, _ = atmos.isa(350.0 * 100.0 * 0.3048)
        expected_kt = atmos.speed_of_sound(isa_t_k + 5.0) * 0.95 / atmos.KT_TO_MS
        r = conc_data.subsonic_cruise(350.0, 110.0, 5.0)
        assert r["tas_kt"] == pytest.approx(expected_kt)

    def test_matches_all_printed_values_within_half_a_knot(self, subsonic_csv):
        level_fl = subsonic_csv["level_fl"].to_numpy(float)
        isa_dev_c = subsonic_csv["isa_dev_c"].to_numpy(float)
        mass_t = subsonic_csv["mass_t"].to_numpy(float)
        r = conc_data.subsonic_cruise(level_fl, mass_t, isa_dev_c)
        assert np.max(np.abs(r["tas_kt"] - subsonic_csv["tas_kt"].to_numpy(float))) < 0.5

    def test_tas_does_not_depend_on_mass(self):
        """tas_kt is mach * speed of sound -- mass never enters it."""
        light = conc_data.subsonic_cruise(350.0, 100.0, 0.0)["tas_kt"]
        heavy = conc_data.subsonic_cruise(350.0, 165.0, 0.0)["tas_kt"]
        assert light == pytest.approx(heavy)


class TestBroadcasting:
    def test_broadcasts_across_candidates(self):
        level_fl = np.array([350.0, 410.0])
        mass_t = np.array([110.0, 150.0])
        isa_dev_c = np.array([0.0, 0.0])
        r = conc_data.subsonic_cruise(level_fl, mass_t, isa_dev_c)
        assert r["specific_range_nm_per_t"].shape == (2,)
        assert not np.isnan(r["specific_range_nm_per_t"][0])
        assert np.isnan(r["specific_range_nm_per_t"][1])

    def test_scalar_level_broadcasts_against_array_mass(self):
        mass_t = np.array([105.0, 110.0, 115.0, 120.0])
        r = conc_data.subsonic_cruise(350.0, mass_t, 0.0)
        assert r["specific_range_nm_per_t"].shape == (4,)
        assert np.all(np.isfinite(r["specific_range_nm_per_t"]))
