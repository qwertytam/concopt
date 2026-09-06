"""Parse a P3D .pln flight plan into legs with distances and tracks.
Vectorised numpy, SI plus nautical miles at the interface, no pint.
"""
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

R_NM = 3440.065  # earth radius, nautical miles

# WorldPosition / *LLA token: HEMdeg° min' sec", e.g. N40° 38' 23.39" or
# W68° 15' 0" (minutes/seconds may be single-digit, seconds may lack a
# decimal part).
_DMS_RE = re.compile(r'([NSEW])(\d+)°\s*(\d+)\'\s*([\d.]+)"')


def _dms_to_decimal(token):
    """Signed decimal degrees from one 'N40° 38' 23.39"' style DMS token."""
    hemi, deg, minute, sec = _DMS_RE.match(token.strip()).groups()
    dd = float(deg) + float(minute) / 60.0 + float(sec) / 3600.0
    return -dd if hemi in ('S', 'W') else dd


def _parse_world_position(text):
    """lat_deg, lon_deg from a WorldPosition/*LLA string. Ignores the
    trailing altitude field, so the DepartureLLA variant's extra leading
    space before that field doesn't matter."""
    lat_str, lon_str, _alt_str = text.split(',')
    return _dms_to_decimal(lat_str), _dms_to_decimal(lon_str)


def parse_pln(path):
    """Parse a P3D .pln file into title, cruising altitude, endpoint IDs,
    and waypoints (id, lat_deg, lon_deg) in file order.

    ATCWaypointType is not used to identify the endpoints (it is unreliable
    in practice, e.g. every fix typed "Airport") - DepartureID/DestinationID
    are used instead.
    """
    root = ET.parse(path).getroot()
    plan = root.find('FlightPlan.FlightPlan')

    waypoints = [
        (wp.get('id'), *_parse_world_position(wp.find('WorldPosition').text))
        for wp in plan.findall('ATCWaypoint')
    ]

    return {
        'title': plan.find('Title').text,
        'cruising_alt_ft': float(plan.find('CruisingAlt').text),
        'departure_id': plan.find('DepartureID').text,
        'destination_id': plan.find('DestinationID').text,
        'waypoints': waypoints,
    }


def _central_angle_rad(lat1, lon1, lat2, lon2):
    """Great-circle angular separation (rad) via the haversine formula."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def great_circle_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance (nm), haversine, R = 3440.065 nm."""
    return R_NM * _central_angle_rad(lat1, lon1, lat2, lon2)


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """True initial bearing (deg, 0-360) from point 1 to point 2."""
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))

    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def intermediate_point(lat1, lon1, lat2, lon2, f):
    """Great-circle interpolation (slerp) between point 1 and point 2,
    f in [0, 1]. f may be an array, broadcasting against scalar endpoints."""
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lon1r = np.radians(np.asarray(lon1, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    lon2r = np.radians(np.asarray(lon2, dtype=float))
    f = np.asarray(f, dtype=float)

    delta = _central_angle_rad(lat1, lon1, lat2, lon2)
    a = np.sin((1.0 - f) * delta) / np.sin(delta)
    b = np.sin(f * delta) / np.sin(delta)

    x = a * np.cos(lat1r) * np.cos(lon1r) + b * np.cos(lat2r) * np.cos(lon2r)
    y = a * np.cos(lat1r) * np.sin(lon1r) + b * np.cos(lat2r) * np.sin(lon2r)
    z = a * np.sin(lat1r) + b * np.sin(lat2r)

    lat_i = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    lon_i = np.arctan2(y, x)
    return np.degrees(lat_i), np.degrees(lon_i)


@dataclass
class Leg:
    from_id: str
    to_id: str
    lat_mid: float
    lon_mid: float
    track_deg: float
    dist_nm: float
    cum_nm: float


def build_legs(waypoints, max_leg_nm=100.0):
    """Waypoints (id, lat_deg, lon_deg) in file order -> list of Leg.

    Legs longer than max_leg_nm are subdivided into equal great-circle
    sub-legs, each with its own midpoint and its own initial bearing (the
    track rotates along a long great circle, so a sub-leg's bearing is
    computed at that sub-leg, not carried over from the parent). Sub-legs
    inherit their parent's from_id/to_id, so supersonic_segment's mask still
    lines up after subdivision.
    """
    legs = []
    cum_nm = 0.0
    for (from_id, lat1, lon1), (to_id, lat2, lon2) in zip(waypoints, waypoints[1:]):
        dist_nm = float(great_circle_nm(lat1, lon1, lat2, lon2))
        n_sub = max(1, int(np.ceil(dist_nm / max_leg_nm)))
        sub_dist_nm = dist_nm / n_sub

        bounds = np.arange(n_sub + 1) / n_sub
        sub_lat, sub_lon = intermediate_point(lat1, lon1, lat2, lon2, bounds)
        tracks = initial_bearing_deg(sub_lat[:-1], sub_lon[:-1], sub_lat[1:], sub_lon[1:])
        mid_lat, mid_lon = intermediate_point(lat1, lon1, lat2, lon2, (np.arange(n_sub) + 0.5) / n_sub)

        for k in range(n_sub):
            cum_nm += sub_dist_nm
            legs.append(Leg(from_id, to_id, float(mid_lat[k]), float(mid_lon[k]),
                            float(tracks[k]), sub_dist_nm, cum_nm))
    return legs


def supersonic_segment(legs, accel_id="LINND", decel_id="BARIX"):
    """Boolean mask over legs, True from the leg departing accel_id through
    the leg arriving at decel_id (inclusive). Sub-legs of a subdivided
    parent leg all share that parent's from_id/to_id, so this still finds
    the right span after subdivision."""
    from_ids = np.array([leg.from_id for leg in legs])
    to_ids = np.array([leg.to_id for leg in legs])

    start_idx = np.flatnonzero(from_ids == accel_id)[0]
    end_idx = np.flatnonzero(to_ids == decel_id)[-1]

    mask = np.zeros(len(legs), dtype=bool)
    mask[start_idx:end_idx + 1] = True
    return mask
