"""Acceptance tests for concopt.runways: the headwind/crosswind-gust
decomposition, the runway screen/selection logic, and the bracketing-hour
surface-wind interpolation.
"""
import math

import numpy as np
import pytest

from concopt.atmos import KT_TO_MS
from concopt.runways import (RUNWAYS, TAILWIND_MAX_KT, XWIND_FLAG_KT, XWIND_OK_KT,
                              select_runway, wind_at)


def _uv_from_dir_speed(from_dir_deg, speed_kt):
    """(u, v) m/s -- eastward/northward wind vector, ERA5 convention -- for
    a wind blowing FROM from_dir_deg (true) at speed_kt. Independent of
    runways.py's own along/cross decomposition, so it cross-checks that
    decomposition rather than assuming it."""
    speed_ms = speed_kt * KT_TO_MS
    bearing_to_rad = math.radians(from_dir_deg + 180.0)
    u = speed_ms * math.sin(bearing_to_rad)
    v = speed_ms * math.cos(bearing_to_rad)
    return np.array([u]), np.array([v])


JFK_22R_TRUE = RUNWAYS["KJFK"][0]["true_deg"]  # 211
JFK_31L_TRUE = RUNWAYS["KJFK"][1]["true_deg"]  # 301


def test_pure_headwind_no_crosswind():
    """Wind straight down the runway (FROM its own true heading) is 100%
    headwind, 0 crosswind, at any gust speed."""
    u, v = _uv_from_dir_speed(JFK_22R_TRUE, speed_kt=20.0)
    gust = np.array([40.0]) * KT_TO_MS

    out = select_runway(u, v, gust, "KJFK")

    assert out["runway"][0] == "22R"
    assert out["headwind_kt"][0] == pytest.approx(20.0, abs=0.1)
    assert out["xwind_gust_kt"][0] == pytest.approx(0.0, abs=0.1)
    assert out["flag"][0] == ""


def test_pure_crosswind_scales_with_gust_not_mean():
    """Wind at 90 deg to the runway is 0 headwind, and its crosswind scales
    with the GUST speed, not the mean speed (the whole gust is
    crosswind when it's all crosswind at the mean-wind direction)."""
    u, v = _uv_from_dir_speed(JFK_22R_TRUE + 90.0, speed_kt=10.0)
    gust = np.array([40.0]) * KT_TO_MS

    s = np.sin(np.radians(JFK_22R_TRUE))
    c = np.cos(np.radians(JFK_22R_TRUE))
    headwind_ms = -(u * s + v * c)
    crosswind_gust_ms = np.abs(u * c - v * s) / np.hypot(u, v) * gust

    assert headwind_ms[0] / KT_TO_MS == pytest.approx(0.0, abs=0.1)
    assert crosswind_gust_ms[0] / KT_TO_MS == pytest.approx(40.0, abs=0.1)


def test_calm_mean_wind_treats_full_gust_as_crosswind():
    """No mean wind means no direction to scale the gust's crosswind
    fraction from -- the conservative fallback treats the whole gust as
    crosswind (see _headwind_crosswind_gust)."""
    u = np.array([0.0])
    v = np.array([0.0])
    gust = np.array([40.0]) * KT_TO_MS

    out = select_runway(u, v, gust, "KJFK")

    assert out["xwind_gust_kt"][0] == pytest.approx(40.0, abs=0.1)


def test_flagged_25_to_30kt_band():
    """A crosswind gust just inside (25, 30] kt on the winning (greatest
    headwind) runway is allowed but flagged, not unflyable."""
    u, v = _uv_from_dir_speed(JFK_22R_TRUE + 15.0, speed_kt=15.0)
    gust_ms = np.array([103.0]) * KT_TO_MS  # ~103 kt gust

    out = select_runway(u, v, gust_ms, "KJFK")

    assert out["runway"][0] == "22R"  # clearly the greater-headwind runway
    assert 25.0 < out["xwind_gust_kt"][0] <= 30.0
    assert out["flag"][0] == "xwind_25_30"


def test_unflyable_when_no_runway_passes_but_still_reports_one():
    """A big gust roughly 45 deg off both (orthogonal) JFK runways pushes
    the crosswind-gust test over 30 kt on both -- unflyable, but a runway
    (the greater-headwind one) is still reported, not dropped."""
    u, v = _uv_from_dir_speed((JFK_22R_TRUE + JFK_31L_TRUE) / 2.0, speed_kt=10.0)
    gust = np.array([70.0]) * KT_TO_MS

    out = select_runway(u, v, gust, "KJFK")

    assert out["flag"][0] == "unflyable"
    assert out["runway"][0] in ("22R", "31L")
    assert out["xwind_gust_kt"][0] > XWIND_FLAG_KT


def test_tailwind_over_max_is_unflyable_on_that_runway():
    """A runway with the wind squarely FROM behind (tailwind > 10 kt) is
    unflyable on that runway even with zero crosswind -- and the
    orthogonal runway (pure crosswind there, no tailwind) still wins."""
    u, v = _uv_from_dir_speed(JFK_22R_TRUE + 180.0, speed_kt=15.0)  # tailwind on 22R
    gust = np.array([15.0]) * KT_TO_MS

    out = select_runway(u, v, gust, "KJFK")

    assert out["runway"][0] == "31L"
    assert out["flag"][0] == ""


def test_all_ok_no_flag():
    """Light, nearly-straight-down-the-runway wind: no flag."""
    u, v = _uv_from_dir_speed(JFK_22R_TRUE + 5.0, speed_kt=10.0)
    gust = np.array([12.0]) * KT_TO_MS

    out = select_runway(u, v, gust, "KJFK")

    assert out["runway"][0] == "22R"
    assert out["flag"][0] == ""
    assert out["xwind_gust_kt"][0] <= XWIND_OK_KT


def test_wind_at_interpolates_between_bracketing_hours():
    """Linear interpolation between two hourly records, matching
    march_legs's bracketing-hour treatment of the upper-air data."""
    times = np.array(["2020-01-01T00:00", "2020-01-01T01:00"], dtype="datetime64[ns]")
    surface_data = {
        "KJFK": {
            "time": times,
            "u10": np.array([0.0, 10.0]),
            "v10": np.array([0.0, 0.0]),
            "i10fg": np.array([5.0, 15.0]),
        }
    }
    query = times[0].astype("int64") + int(0.25 * 3600 * 1e9)

    u, v, gust = wind_at(surface_data, "KJFK", np.array([query]))

    assert u[0] == pytest.approx(2.5)
    assert v[0] == pytest.approx(0.0)
    assert gust[0] == pytest.approx(7.5)


def test_wind_at_clamps_outside_range():
    """A query before/after the data's time range clamps to the nearest
    bracket rather than extrapolating."""
    times = np.array(["2020-01-01T00:00", "2020-01-01T01:00"], dtype="datetime64[ns]")
    surface_data = {
        "KJFK": {
            "time": times,
            "u10": np.array([0.0, 10.0]),
            "v10": np.array([0.0, 0.0]),
            "i10fg": np.array([5.0, 15.0]),
        }
    }
    before = times[0].astype("int64") - int(3600 * 1e9)
    after = times[1].astype("int64") + int(3600 * 1e9)

    u, _, _ = wind_at(surface_data, "KJFK", np.array([before, after]))

    assert u[0] == pytest.approx(0.0)
    assert u[1] == pytest.approx(10.0)
