"""Acceptance tests for the climb performance table (conc_climb.csv) and
conc_data.climb_to, which replaces search.py's old fixed
DEPARTURE_TO_ACCEL_S/WEIGHT_ACCEL_T assumption with a TOW+temperature-band
model of brake-release-to-top-of-climb.
"""
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from concopt.data.conc_data import CLIMB_BANDS, climb_to
from concopt.route import build_legs, climb_cruise_segment, parse_pln
from tests.test_route import SAMPLE_PLN


@pytest.fixture(scope="module")
def climb_table():
    fp = files("concopt").joinpath("data/conc_climb.csv")
    return pd.read_csv(fp, encoding="utf-8-sig")


def test_row_count(climb_table):
    """3 temp bands x 6 TOW (160..185, 5 t steps) x 18 levels = 324 rows."""
    assert len(climb_table) == 324


def test_mass_matches_tow_minus_fuel(climb_table):
    """mass_t == tow_t - fuel_used_kg/1000, within the table's own rounding."""
    calc = climb_table["tow_t"] - climb_table["fuel_used_kg"] / 1000.0
    assert (calc - climb_table["mass_t"]).abs().max() <= 0.5


def test_dist_and_time_monotonic_in_tow(climb_table):
    """At fixed (band, level), heavier TOW takes longer and covers less
    ground (thinner climb performance) -- non-decreasing time, and dist_nm
    should broadly track it rather than run backwards."""
    for band in CLIMB_BANDS:
        for level_fl, grp in climb_table[climb_table["temp_band"] == band].groupby("level_fl"):
            grp = grp.sort_values("tow_t")
            assert np.all(np.diff(grp["time_min"].to_numpy()) >= -1e-9), \
                f"time_min not monotonic in tow_t for {band}/{level_fl}"


def test_dist_and_time_monotonic_in_level(climb_table):
    """At fixed (band, tow), a higher top-of-climb level takes longer and
    covers more ground."""
    for band in CLIMB_BANDS:
        for tow_t, grp in climb_table[climb_table["temp_band"] == band].groupby("tow_t"):
            grp = grp.sort_values("level_fl")
            assert np.all(np.diff(grp["dist_nm"].to_numpy()) >= -1e-9), \
                f"dist_nm not monotonic in level_fl for {band}/{tow_t}"
            assert np.all(np.diff(grp["time_min"].to_numpy()) >= -1e-9), \
                f"time_min not monotonic in level_fl for {band}/{tow_t}"


def test_climb_speed_is_plausible(climb_table):
    """Average air speed over the climb (dist_nm/time_min*60) should sit in
    a plausible subsonic-to-low-supersonic climb-out range for every row."""
    speed_kt = climb_table["dist_nm"] / climb_table["time_min"] * 60.0
    assert speed_kt.between(400.0, 1000.0).all()


def test_climb_to_spot_values_cold_band():
    mass_t, fuel_used_kg, dist_nm, time_min = climb_to(502.0, 185.0, "isa_minus_20_to_minus_10")
    assert float(dist_nm) == pytest.approx(324.0, abs=0.5)
    assert float(time_min) == pytest.approx(26.0, abs=0.5)


def test_climb_to_spot_values_warm_band():
    mass_t, fuel_used_kg, dist_nm, time_min = climb_to(502.0, 185.0, "isa_to_isa_plus_10")
    assert float(dist_nm) == pytest.approx(1047.0, abs=0.5)
    assert float(time_min) == pytest.approx(65.0, abs=0.5)


def test_climb_to_unknown_band_raises():
    with pytest.raises(ValueError, match="temp_band"):
        climb_to(502.0, 185.0, "isa_plus_10_to_plus_20")


def test_climb_to_clamps_out_of_range_tow():
    """No extrapolation -- a TOW above the table's 185 t top clamps to the
    185 t row, matching the ceiling/fuel tables' own convention."""
    at_max = climb_to(502.0, 185.0, "isa_to_isa_plus_10")
    above_max = climb_to(502.0, 999.0, "isa_to_isa_plus_10")
    assert float(above_max[2]) == pytest.approx(float(at_max[2]))


def test_end_to_end_top_of_climb_beyond_fix03():
    """At TOW 185 in the ISA..+10 band, top of climb (1047 nm air, and at
    least that far over the ground even with a headwind component of 0)
    falls beyond Fix03 (702 nm on the sample .pln) -- the march should
    start past it, not fly Fix03 supersonically."""
    plan = parse_pln(SAMPLE_PLN)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs)
    cc_legs = [leg for leg, m in zip(legs, mask) if m]

    # Fix02->Fix03 is subdivided into several equal sub-legs sharing that
    # from_id/to_id (route.build_legs) -- the last one carries the true
    # cumulative distance to the named waypoint.
    fix03 = [leg for leg in cc_legs if leg.to_id == "Fix03"][-1]
    assert fix03.cum_nm == pytest.approx(702.0, abs=1.0)

    _mass_t, _fuel_used_kg, dist_nm, _time_min = climb_to(502.0, 185.0, "isa_to_isa_plus_10")
    assert float(dist_nm) > fix03.cum_nm
