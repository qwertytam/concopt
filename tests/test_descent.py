"""Tests for descent table interpolators: decel_to_mach1, descent_to_1500ft,
descent_direct_from_cruise, and dist_with_wind."""

import numpy as np
import pandas as pd
import pytest

from concopt.data import conc_data


@pytest.fixture
def descent_csv():
    """Load conc_descent.csv directly for reference."""
    from importlib.resources import files
    fp = files("concopt").joinpath("data/conc_descent.csv")
    return pd.read_csv(fp, encoding="utf-8-sig")


class TestDescentRowCount:
    """Row count of the descent CSV."""

    def test_row_count(self, descent_csv):
        assert len(descent_csv) == 291


class TestDescentMonotonicity:
    """Every (speed, temp_band, table, from_supersonic_cruise) group is monotonic
    in level_fl for fuel_t, time_min, and dist_zero_wind_nm."""

    def test_monotonic_time(self, descent_csv):
        for (speed, band, table, from_super), group in descent_csv.groupby(
            ["descent_speed_kt", "temp_band", "table", "from_supersonic_cruise"]
        ):
            group = group.sort_values("level_fl")
            time = group["time_min"].to_numpy(float)
            diffs = np.diff(time)
            assert np.all(diffs >= 0) or np.all(diffs <= 0), \
                f"time_min not monotonic for speed={speed}, band={band}, table={table}, from_super={from_super}"

    def test_monotonic_distance(self, descent_csv):
        for (speed, band, table, from_super), group in descent_csv.groupby(
            ["descent_speed_kt", "temp_band", "table", "from_supersonic_cruise"]
        ):
            group = group.sort_values("level_fl")
            dist = group["dist_zero_wind_nm"].to_numpy(float)
            diffs = np.diff(dist)
            assert np.all(diffs >= 0) or np.all(diffs <= 0), \
                f"dist_zero_wind_nm not monotonic for speed={speed}, band={band}, table={table}, from_super={from_super}"


class TestDecelToMach1ExactHits:
    """Exact table hits return the table value."""

    def test_exact_hit_325_above(self, descent_csv):
        # 325 kt, above_isa_minus_10, decel_to_mach1, level 600
        row = descent_csv[
            (descent_csv["descent_speed_kt"] == 325) &
            (descent_csv["temp_band"] == "above_isa_minus_10") &
            (descent_csv["table"] == "decel_to_mach1") &
            (descent_csv["level_fl"] == 600)
        ].iloc[0]
        result = conc_data.decel_to_mach1(600, 325, "above_isa_minus_10")
        assert np.isclose(result["fuel_t"], row["fuel_t"])
        assert np.isclose(result["time_min"], row["time_min"])
        assert np.isclose(result["dist_zero_wind_nm"], row["dist_zero_wind_nm"])

    def test_exact_hit_350_below(self, descent_csv):
        # 350 kt, isa_minus_10_and_below, decel_to_mach1, level 530
        row = descent_csv[
            (descent_csv["descent_speed_kt"] == 350) &
            (descent_csv["temp_band"] == "isa_minus_10_and_below") &
            (descent_csv["table"] == "decel_to_mach1") &
            (descent_csv["level_fl"] == 530)
        ].iloc[0]
        result = conc_data.decel_to_mach1(530, 350, "isa_minus_10_and_below")
        assert np.isclose(result["fuel_t"], row["fuel_t"])
        assert np.isclose(result["time_min"], row["time_min"])
        assert np.isclose(result["dist_zero_wind_nm"], row["dist_zero_wind_nm"])

    def test_exact_hit_380_above(self, descent_csv):
        # 380 kt, above_isa_minus_10, decel_to_mach1, level 470
        row = descent_csv[
            (descent_csv["descent_speed_kt"] == 380) &
            (descent_csv["temp_band"] == "above_isa_minus_10") &
            (descent_csv["table"] == "decel_to_mach1") &
            (descent_csv["level_fl"] == 470)
        ].iloc[0]
        result = conc_data.decel_to_mach1(470, 380, "above_isa_minus_10")
        assert np.isclose(result["fuel_t"], row["fuel_t"])
        assert np.isclose(result["time_min"], row["time_min"])
        assert np.isclose(result["dist_zero_wind_nm"], row["dist_zero_wind_nm"])


class TestDescentTo1500FtExactHits:
    """Exact table hits for descent_to_1500ft (subsonic)."""

    def test_exact_hit_325_above(self, descent_csv):
        # 325 kt, above_isa_minus_10, descent_to_1500ft, from_supersonic_cruise=False, level 550
        row = descent_csv[
            (descent_csv["descent_speed_kt"] == 325) &
            (descent_csv["temp_band"] == "above_isa_minus_10") &
            (descent_csv["table"] == "descent_to_1500ft") &
            (descent_csv["from_supersonic_cruise"] == False) &
            (descent_csv["level_fl"] == 550)
        ].iloc[0]
        result = conc_data.descent_to_1500ft(550, 325, "above_isa_minus_10")
        assert np.isclose(result["fuel_t"], row["fuel_t"])
        assert np.isclose(result["time_min"], row["time_min"])
        assert np.isclose(result["dist_zero_wind_nm"], row["dist_zero_wind_nm"])

    def test_exact_hit_350_below(self, descent_csv):
        # 350 kt, isa_minus_10_and_below, descent_to_1500ft, from_supersonic_cruise=False, level 350
        row = descent_csv[
            (descent_csv["descent_speed_kt"] == 350) &
            (descent_csv["temp_band"] == "isa_minus_10_and_below") &
            (descent_csv["table"] == "descent_to_1500ft") &
            (descent_csv["from_supersonic_cruise"] == False) &
            (descent_csv["level_fl"] == 350)
        ].iloc[0]
        result = conc_data.descent_to_1500ft(350, 350, "isa_minus_10_and_below")
        assert np.isclose(result["fuel_t"], row["fuel_t"])
        assert np.isclose(result["time_min"], row["time_min"])
        assert np.isclose(result["dist_zero_wind_nm"], row["dist_zero_wind_nm"])

    def test_exact_hit_380_above(self, descent_csv):
        # 380 kt, above_isa_minus_10, descent_to_1500ft, from_supersonic_cruise=False, level 470
        row = descent_csv[
            (descent_csv["descent_speed_kt"] == 380) &
            (descent_csv["temp_band"] == "above_isa_minus_10") &
            (descent_csv["table"] == "descent_to_1500ft") &
            (descent_csv["from_supersonic_cruise"] == False) &
            (descent_csv["level_fl"] == 470)
        ].iloc[0]
        result = conc_data.descent_to_1500ft(470, 380, "above_isa_minus_10")
        assert np.isclose(result["fuel_t"], row["fuel_t"])
        assert np.isclose(result["time_min"], row["time_min"])
        assert np.isclose(result["dist_zero_wind_nm"], row["dist_zero_wind_nm"])


class TestDescentDirectFromCruiseExactHits:
    """Exact table hits for descent_direct_from_cruise (from_supersonic_cruise=True)."""

    def test_exact_hit_325_above(self, descent_csv):
        # 325 kt, above_isa_minus_10, descent_to_1500ft, from_supersonic_cruise=True, level 600
        row = descent_csv[
            (descent_csv["descent_speed_kt"] == 325) &
            (descent_csv["temp_band"] == "above_isa_minus_10") &
            (descent_csv["table"] == "descent_to_1500ft") &
            (descent_csv["from_supersonic_cruise"] == True) &
            (descent_csv["level_fl"] == 600)
        ].iloc[0]
        result = conc_data.descent_direct_from_cruise(600, 325, "above_isa_minus_10")
        assert np.isclose(result["fuel_t"], row["fuel_t"])
        assert np.isclose(result["time_min"], row["time_min"])
        assert np.isclose(result["dist_zero_wind_nm"], row["dist_zero_wind_nm"])

    def test_exact_hit_350_below(self, descent_csv):
        # 350 kt, isa_minus_10_and_below, descent_to_1500ft, from_supersonic_cruise=True, level 530
        row = descent_csv[
            (descent_csv["descent_speed_kt"] == 350) &
            (descent_csv["temp_band"] == "isa_minus_10_and_below") &
            (descent_csv["table"] == "descent_to_1500ft") &
            (descent_csv["from_supersonic_cruise"] == True) &
            (descent_csv["level_fl"] == 530)
        ].iloc[0]
        result = conc_data.descent_direct_from_cruise(530, 350, "isa_minus_10_and_below")
        assert np.isclose(result["fuel_t"], row["fuel_t"])
        assert np.isclose(result["time_min"], row["time_min"])
        assert np.isclose(result["dist_zero_wind_nm"], row["dist_zero_wind_nm"])


class TestInterpolation:
    """Midpoint interpolates to the chord."""

    def test_midpoint_interpolation(self, descent_csv):
        # Get a group with at least 2 points to interpolate between
        group = descent_csv[
            (descent_csv["descent_speed_kt"] == 325) &
            (descent_csv["temp_band"] == "above_isa_minus_10") &
            (descent_csv["table"] == "decel_to_mach1")
        ].sort_values("level_fl")

        # Take the first two rows
        row1 = group.iloc[0]
        row2 = group.iloc[1]

        level1 = float(row1["level_fl"])
        level2 = float(row2["level_fl"])
        mid_level = (level1 + level2) / 2

        # Interpolate at midpoint
        result = conc_data.decel_to_mach1(mid_level, 325, "above_isa_minus_10")

        # Linear interpolation should give the chord
        expected_fuel = (float(row1["fuel_t"]) + float(row2["fuel_t"])) / 2
        expected_time = (float(row1["time_min"]) + float(row2["time_min"])) / 2
        expected_dist = (float(row1["dist_zero_wind_nm"]) + float(row2["dist_zero_wind_nm"])) / 2

        assert np.isclose(result["fuel_t"], expected_fuel, rtol=1e-6)
        assert np.isclose(result["time_min"], expected_time, rtol=1e-6)
        assert np.isclose(result["dist_zero_wind_nm"], expected_dist, rtol=1e-6)


class TestClamping:
    """Out-of-range level_fl clamps rather than extrapolates."""

    def test_clamp_low(self):
        # Request a level below the table's minimum (470)
        result_low = conc_data.decel_to_mach1(400, 325, "above_isa_minus_10")
        result_min = conc_data.decel_to_mach1(470, 325, "above_isa_minus_10")
        # Should get the same result (clamped to minimum)
        assert np.isclose(result_low["fuel_t"], result_min["fuel_t"])
        assert np.isclose(result_low["time_min"], result_min["time_min"])

    def test_clamp_high(self):
        # Request a level above the table's maximum (600)
        result_high = conc_data.decel_to_mach1(650, 325, "above_isa_minus_10")
        result_max = conc_data.decel_to_mach1(600, 325, "above_isa_minus_10")
        # Should get the same result (clamped to maximum)
        assert np.isclose(result_high["fuel_t"], result_max["fuel_t"])
        assert np.isclose(result_high["time_min"], result_max["time_min"])


class TestDecelEndFl:
    """decel_end_fl is constant within each speed schedule."""

    def test_constant_325(self):
        result1 = conc_data.decel_to_mach1(600, 325, "above_isa_minus_10")
        result2 = conc_data.decel_to_mach1(500, 325, "above_isa_minus_10")
        result3 = conc_data.decel_to_mach1(470, 325, "isa_minus_10_and_below")
        assert result1["decel_end_fl"] == 383
        assert result2["decel_end_fl"] == 383
        assert result3["decel_end_fl"] == 383

    def test_constant_350(self):
        result1 = conc_data.decel_to_mach1(600, 350, "above_isa_minus_10")
        result2 = conc_data.decel_to_mach1(500, 350, "above_isa_minus_10")
        result3 = conc_data.decel_to_mach1(470, 350, "isa_minus_10_and_below")
        assert result1["decel_end_fl"] == 350
        assert result2["decel_end_fl"] == 350
        assert result3["decel_end_fl"] == 350

    def test_constant_380(self):
        result1 = conc_data.decel_to_mach1(600, 380, "above_isa_minus_10")
        result2 = conc_data.decel_to_mach1(500, 380, "above_isa_minus_10")
        result3 = conc_data.decel_to_mach1(470, 380, "isa_minus_10_and_below")
        assert result1["decel_end_fl"] == 312
        assert result2["decel_end_fl"] == 312
        assert result3["decel_end_fl"] == 312

    def test_level_correction_constant(self):
        # level_correction_nm_per_2000ft should also be constant per speed
        result_325_1 = conc_data.decel_to_mach1(600, 325, "above_isa_minus_10")
        result_325_2 = conc_data.decel_to_mach1(470, 325, "above_isa_minus_10")
        assert result_325_1["level_correction_nm_per_2000ft"] == result_325_2["level_correction_nm_per_2000ft"]
        assert result_325_1["level_correction_nm_per_2000ft"] == 5


class TestDistWithWind:
    """Wind correction function behavior."""

    def test_zero_wind(self):
        # Zero wind should return distance unchanged
        dist_result = conc_data.dist_with_wind(100.0, 10.0, 0.0)
        assert np.isclose(dist_result, 100.0)

    def test_tailwind_addition(self):
        # 60 kt tailwind over 10 min should add 10 nm
        # dist = 0 + 60 * 10 / 60 = 10
        dist_result = conc_data.dist_with_wind(0.0, 10.0, 60.0)
        assert np.isclose(dist_result, 10.0)

    def test_headwind_subtraction(self):
        # -60 kt (headwind) over 10 min should subtract 10 nm
        dist_result = conc_data.dist_with_wind(100.0, 10.0, -60.0)
        assert np.isclose(dist_result, 90.0)

    def test_broadcasting(self):
        # Test array broadcasting
        dist_array = np.array([100.0, 200.0, 300.0])
        time_array = np.array([10.0, 20.0, 30.0])
        wind = 60.0
        result = conc_data.dist_with_wind(dist_array, time_array, wind)
        expected = dist_array + wind * time_array / 60.0
        assert np.allclose(result, expected)


class TestGroundSpeed:
    """Implied mean ground speed dist_zero_wind_nm/time_min*60 is 200-900 kt."""

    def test_ground_speed_bounds(self, descent_csv):
        # Compute implied ground speed for every row (ignoring wind)
        # TAS mean is given, but we'll check the basic kinematics
        ground_speeds = descent_csv["dist_zero_wind_nm"] / descent_csv["time_min"] * 60

        assert ground_speeds.min() >= 200, f"Min ground speed {ground_speeds.min()} < 200 kt"
        assert ground_speeds.max() <= 900, f"Max ground speed {ground_speeds.max()} > 900 kt"
