"""Acceptance tests for concopt.route against the sample KJFK-EGLL plan."""
from pathlib import Path

import numpy as np
import pytest

from concopt.route import (
    build_legs,
    current_progress_nm,
    destination_point,
    great_circle_nm,
    initial_bearing_deg,
    intermediate_point,
    parse_pln,
    project_along_route,
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


def test_destination_point_round_trips_with_great_circle_nm():
    """destination_point is the direct geodesic problem, the inverse of
    great_circle_nm/initial_bearing_deg: going dist_nm along the bearing to
    a point lands back on it."""
    lat1, lon1, lat2, lon2 = 40.0, -73.0, 51.0, 0.0
    dist_nm = great_circle_nm(lat1, lon1, lat2, lon2)
    brng = initial_bearing_deg(lat1, lon1, lat2, lon2)

    lat_out, lon_out = destination_point(lat1, lon1, brng, dist_nm)

    assert lat_out == pytest.approx(lat2, abs=1e-6)
    assert lon_out == pytest.approx(lon2, abs=1e-6)


def test_destination_point_zero_distance_is_identity():
    lat_out, lon_out = destination_point(40.0, -73.0, 90.0, 0.0)
    assert lat_out == pytest.approx(40.0)
    assert lon_out == pytest.approx(-73.0)


def test_current_progress_nm_at_a_leg_midpoint(legs):
    """Standing exactly on a leg's own midpoint, progress is that leg's
    along-track midpoint distance -- half the leg's own length short of its
    cum_nm."""
    leg = legs[5]
    cum_nm, leg_idx, along_nm = current_progress_nm(legs, leg.lat_mid, leg.lon_mid)

    assert leg_idx == 5
    assert along_nm == pytest.approx(leg.dist_nm / 2.0, abs=0.5)
    assert cum_nm == pytest.approx(leg.cum_nm - leg.dist_nm / 2.0, abs=0.5)


def test_current_progress_nm_monotonic_along_route(legs):
    """Progress increases leg by leg down the route."""
    progress = [current_progress_nm(legs, leg.lat_mid, leg.lon_mid)[0] for leg in legs]
    assert np.all(np.diff(progress) > 0)


def test_project_along_route_advances_by_lookahead_nm(legs):
    """From a leg's own midpoint, the projected point should be roughly
    lookahead_nm of path length further down the route (chord distance can
    fall a bit short across a turn, but not by much over a short hop)."""
    leg = legs[3]
    lat, lon, track_deg = project_along_route(legs, leg.lat_mid, leg.lon_mid, 20.0)

    d = great_circle_nm(leg.lat_mid, leg.lon_mid, lat, lon)
    assert d == pytest.approx(20.0, rel=0.1)
    assert 0.0 <= track_deg < 360.0


def test_project_along_route_clamps_past_route_end(legs):
    """A lookahead far beyond the route's end clamps to (approximately) the
    last leg's own endpoint, rather than extrapolating or raising."""
    last = legs[-1]
    lat, lon, _track_deg = project_along_route(legs, last.lat_mid, last.lon_mid, 1_000_000.0)

    end_lat, end_lon = destination_point(last.lat_mid, last.lon_mid, last.track_deg, last.dist_nm / 2.0)
    assert great_circle_nm(lat, lon, end_lat, end_lon) < 1.0
