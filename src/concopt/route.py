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


def destination_point(lat, lon, bearing_deg, dist_nm):
    """The point dist_nm along true bearing_deg (deg clockwise from north)
    from (lat, lon) -- the direct geodesic problem, inverse of
    great_circle_nm/initial_bearing_deg. Used to reconstruct a leg's
    endpoints from its stored midpoint/track/dist_nm (Leg keeps only the
    midpoint), and to walk a lookahead distance along the route in
    project_along_route."""
    lat1 = np.radians(np.asarray(lat, dtype=float))
    lon1 = np.radians(np.asarray(lon, dtype=float))
    brng = np.radians(np.asarray(bearing_deg, dtype=float))
    delta = np.asarray(dist_nm, dtype=float) / R_NM

    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) + np.cos(lat1) * np.sin(delta) * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(delta) * np.cos(lat1),
        np.cos(delta) - np.sin(lat1) * np.sin(lat2),
    )
    return np.degrees(lat2), np.degrees(lon2)


def intermediate_point(lat1, lon1, lat2, lon2, f):
    """Great-circle interpolation (slerp) between point 1 and point 2,
    f in [0, 1]. f may be an array, broadcasting against scalar endpoints.

    Coincident endpoints (delta == 0) return point 1 for every f, rather than
    NaN from dividing by sin(0)."""
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lon1r = np.radians(np.asarray(lon1, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    lon2r = np.radians(np.asarray(lon2, dtype=float))
    f = np.asarray(f, dtype=float)

    delta = _central_angle_rad(lat1, lon1, lat2, lon2)
    if delta == 0.0:
        lat_i = np.broadcast_to(lat1r, f.shape) if f.shape else lat1r
        lon_i = np.broadcast_to(lon1r, f.shape) if f.shape else lon1r
        return np.degrees(lat_i), np.degrees(lon_i)

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


def current_progress_nm(legs, lat, lon):
    """(lat, lon)'s along-route progress (cum_nm), for Phase 6's advisor and
    recorder. Finds the leg whose midpoint is nearest (lat, lon),
    reconstructs that leg's endpoints (destination_point against its own
    midpoint/track/dist_nm -- Leg keeps only the midpoint), and projects
    (lat, lon) onto its track to get an along-track offset from the leg's
    start -- clamped to [0, dist_nm], so being off to the side of the route
    doesn't run the projection backwards or past the leg.

    Not vectorised across candidates -- there is only ever one live
    aircraft position, unlike search.py's ~31,000-candidate scan.

    Returns (cum_nm, leg_idx, along_nm): leg_idx/along_nm let
    project_along_route resume the walk from exactly this point without
    redoing the nearest-leg search."""
    mid_lat = np.array([leg.lat_mid for leg in legs])
    mid_lon = np.array([leg.lon_mid for leg in legs])
    i = int(np.argmin(great_circle_nm(lat, lon, mid_lat, mid_lon)))
    leg = legs[i]

    start_lat, start_lon = destination_point(leg.lat_mid, leg.lon_mid, leg.track_deg + 180.0, leg.dist_nm / 2.0)
    d_from_start = great_circle_nm(start_lat, start_lon, lat, lon)
    brng_from_start = initial_bearing_deg(start_lat, start_lon, lat, lon)
    along_nm = float(np.clip(
        d_from_start * np.cos(np.radians(brng_from_start - leg.track_deg)),
        0.0, leg.dist_nm,
    ))
    cum_nm = (leg.cum_nm - leg.dist_nm) + along_nm
    return cum_nm, i, along_nm


def project_along_route(legs, lat, lon, lookahead_nm):
    """The point lookahead_nm ahead of (lat, lon) along the leg sequence, for
    Phase 6's advisor (project forward from the live aircraft position to
    get an Active Sky query point). Starts from current_progress_nm's
    along-track offset into the current leg, adds lookahead_nm, and carries
    any overflow past that leg's end into however many further legs it
    takes -- clamping to the route's last point if lookahead_nm overruns it.

    Returns (lookahead_lat, lookahead_lon, track_deg) -- track_deg is the
    track of the leg the lookahead point ends up on, for feeding
    limits.best_level's wind decomposition."""
    _cum_nm, i, along_nm = current_progress_nm(legs, lat, lon)
    leg = legs[i]
    target_along_nm = along_nm + lookahead_nm

    while target_along_nm > leg.dist_nm and i < len(legs) - 1:
        target_along_nm -= leg.dist_nm
        i += 1
        leg = legs[i]
    target_along_nm = min(target_along_nm, leg.dist_nm)

    leg_start_lat, leg_start_lon = destination_point(leg.lat_mid, leg.lon_mid, leg.track_deg + 180.0, leg.dist_nm / 2.0)
    cur_lat, cur_lon = destination_point(leg_start_lat, leg_start_lon, leg.track_deg, target_along_nm)
    return float(cur_lat), float(cur_lon), float(leg.track_deg)


def supersonic_segment(legs, accel_id="LINND", decel_id="BARIX"):
    """Boolean mask over legs, True from the leg departing accel_id through
    the leg arriving at decel_id (inclusive). Sub-legs of a subdivided
    parent leg all share that parent's from_id/to_id, so this still finds
    the right span after subdivision."""
    from_ids = np.array([leg.from_id for leg in legs])
    to_ids = np.array([leg.to_id for leg in legs])

    accel_matches = np.flatnonzero(from_ids == accel_id)
    if accel_matches.size == 0:
        raise ValueError(
            f"accel_id {accel_id!r} not found; available waypoint ids: "
            f"{sorted(set(from_ids))}"
        )
    decel_matches = np.flatnonzero(to_ids == decel_id)
    if decel_matches.size == 0:
        raise ValueError(
            f"decel_id {decel_id!r} not found; available waypoint ids: "
            f"{sorted(set(to_ids))}"
        )

    start_idx = accel_matches[0]
    end_idx = decel_matches[-1]
    if start_idx > end_idx:
        raise ValueError(
            f"accel_id {accel_id!r} (leg {start_idx}) comes after decel_id "
            f"{decel_id!r} (leg {end_idx}); accel and decel look swapped"
        )

    mask = np.zeros(len(legs), dtype=bool)
    mask[start_idx:end_idx + 1] = True
    return mask
