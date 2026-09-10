"""Acceptance tests for concopt.route against the sample KJFK-EGLL plan."""
from pathlib import Path

import numpy as np
import pytest

from concopt.route import (
    build_legs,
    great_circle_nm,
    intermediate_point,
    parse_pln,
    supersonic_segment,
)

SAMPLE_PLN = Path(__file__).parent / "data" / "KJFKEGLL_CONC_01.pln"

EXP_DIST_NM = [
    30.6, 89.8, 162.3, 66.2, 352.7, 345.6, 114.0, 430.9, 402.4, 389.3,
    194.1, 273.3, 218.0, 51.0, 27.2, 10.4,
]
EXP_TRACK_DEG = [
    127.3, 127.5, 79.2, 59.7, 60.4, 66.6, 66.4, 68.0, 75.4, 82.8,
    90.5, 78.1, 70.4, 102.5, 90.7, 357.1,
]


@pytest.fixture(scope="module")
def plan():
    return parse_pln(SAMPLE_PLN)


@pytest.fixture(scope="module")
def legs(plan):
    return build_legs(plan["waypoints"])


def test_parse_pln_header(plan):
    assert len(plan["waypoints"]) == 17
    assert plan["departure_id"] == "KJFK"
    assert plan["destination_id"] == "EGLL"
    assert plan["cruising_alt_ft"] == 58000


def test_parse_pln_endpoint_coords(plan):
    ids = [w[0] for w in plan["waypoints"]]
    kjfk = plan["waypoints"][ids.index("KJFK")]
    egll = plan["waypoints"][ids.index("EGLL")]

    assert kjfk[1] == pytest.approx(40.6398, abs=1e-4)
    assert kjfk[2] == pytest.approx(-73.7790, abs=1e-4)
    assert egll[1] == pytest.approx(51.4775, abs=1e-4)
    assert egll[2] == pytest.approx(-0.4613, abs=1e-4)


def test_unsubdivided_leg_distances(plan):
    legs = build_legs(plan["waypoints"], max_leg_nm=1e9)
    dist_nm = [leg.dist_nm for leg in legs]
    assert dist_nm == pytest.approx(EXP_DIST_NM, abs=0.1)


def test_unsubdivided_leg_tracks(plan):
    legs = build_legs(plan["waypoints"], max_leg_nm=1e9)
    track_deg = [leg.track_deg for leg in legs]
    assert track_deg == pytest.approx(EXP_TRACK_DEG, abs=0.1)


def test_total_route_and_direct_distance(plan):
    legs = build_legs(plan["waypoints"], max_leg_nm=1e9)
    assert legs[-1].cum_nm == pytest.approx(3158.0, abs=1.0)

    kjfk = plan["waypoints"][0]
    egll = plan["waypoints"][-1]
    direct_nm = great_circle_nm(kjfk[1], kjfk[2], egll[1], egll[2])
    assert direct_nm == pytest.approx(2991.0, abs=1.0)


def test_cumulative_to_linnd_and_barix(plan):
    legs = build_legs(plan["waypoints"], max_leg_nm=1e9)
    linnd_leg = next(leg for leg in legs if leg.to_id == "LINND")
    barix_leg = next(leg for leg in legs if leg.to_id == "BARIX")

    assert linnd_leg.cum_nm == pytest.approx(120.0, abs=1.0)
    assert barix_leg.cum_nm == pytest.approx(2851.0, abs=1.0)


def test_subdivided_legs(legs):
    assert all(leg.dist_nm <= 100.0 + 1e-9 for leg in legs)
    assert len(legs) <= 40
    assert legs[-1].cum_nm == pytest.approx(3158.0, abs=0.5)


def test_supersonic_segment(legs):
    mask = supersonic_segment(legs)
    span_nm = sum(leg.dist_nm for leg in np.array(legs)[mask])

    cum = np.array([leg.cum_nm for leg in legs])
    start_cum = cum[mask][0] - np.array(legs)[mask][0].dist_nm
    end_cum = cum[mask][-1]

    assert start_cum == pytest.approx(120.0, abs=1.0)
    assert end_cum == pytest.approx(2851.0, abs=1.0)
    assert span_nm == pytest.approx(2731.0, abs=2.0)
    assert span_nm / legs[-1].cum_nm == pytest.approx(0.86, abs=0.02)


def test_supersonic_segment_unknown_accel_id_raises(legs):
    with pytest.raises(ValueError, match="NOPE.*LINND"):
        supersonic_segment(legs, accel_id="NOPE", decel_id="BARIX")


def test_supersonic_segment_unknown_decel_id_raises(legs):
    with pytest.raises(ValueError, match="NOPE.*BARIX"):
        supersonic_segment(legs, accel_id="LINND", decel_id="NOPE")


def test_supersonic_segment_swapped_raises(legs):
    with pytest.raises(ValueError, match="swapped"):
        supersonic_segment(legs, accel_id="BARIX", decel_id="LINND")


def test_intermediate_point_coincident_waypoints():
    lat, lon = intermediate_point(40.0, -73.0, 40.0, -73.0, np.array([0.0, 0.5, 1.0]))
    assert lat == pytest.approx([40.0, 40.0, 40.0])
    assert lon == pytest.approx([-73.0, -73.0, -73.0])


def test_build_legs_coincident_waypoints_no_nan():
    waypoints = [("A", 40.0, -73.0), ("A", 40.0, -73.0), ("B", 51.0, 0.0)]
    legs = build_legs(waypoints, max_leg_nm=1e9)
    assert not any(np.isnan(leg.lat_mid) or np.isnan(leg.lon_mid) for leg in legs)
    assert legs[0].dist_nm == pytest.approx(0.0)
