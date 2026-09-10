"""Phase 4: runway selection and crosswind/tailwind screening for the two
route ends (KJFK departure, EGLL arrival), wired into search.run_search so
the day-scan ranks on total block time (brakes-release to landing) rather
than supersonic-segment time alone.

Consumes the combined per-airport surface-wind npz from
era5.reduce_surface_to_npz (10 m u/v wind plus instantaneous gust,
box-averaged over each airport's small ERA5 area) -- never touches the
surface netCDF directly. Vectorised across candidates exactly like
search.march_legs: every function here takes/returns (n_cand,) arrays, no
per-candidate Python loop.

Wind geometry, all bearings TRUE (not magnetic): ERA5's u/v are in the true
frame, and JFK's ~13 deg W variation would otherwise put every runway
number 13 deg off. headwind/crosswind are computed via the same
along-track/cross-track decomposition search.limits.ground_speed already
uses (along = u*sin(track)+v*cos(track), cross = u*cos(track)-v*sin(track)),
not by round-tripping through a "wind FROM direction" in degrees -- it's
algebraically identical (see _headwind_crosswind_gust) and avoids
degrees/atan2 wraparound. headwind is the along-track component with the
sign flipped (positive = wind opposing the direction of travel down the
runway); crosswind is the magnitude of the cross-track component.

ERA5's instantaneous gust (i10fg, the max gust in the preceding hour, the
conservative reading per the CDS docs) carries no direction of its own, so
the crosswind *gust* test reuses the mean wind's direction and rescales
just the magnitude onto the gust speed -- standard practice absent a
reported gust direction.
"""
from pathlib import Path

import numpy as np

from concopt.atmos import KT_TO_MS

# Runway geometry + JFK/LHR-specific missed-approach-slot/taxi penalty, true
# bearings (see module docstring). Only 22R/31L are JFK candidates -- not
# 04L/13R, which this route never uses. EGLL's runways are parallel (09L/
# 09R and 27R/27L share one true bearing each), so there is no crosswind
# relief from choosing between the pair -- only the westerly/easterly
# choice matters here.
RUNWAYS = {
    "KJFK": (
        {"name": "22R", "true_deg": 211.0, "penalty_s": 0.0},
        {"name": "31L", "true_deg": 301.0, "penalty_s": 2.0 * 60.0},
    ),
    "EGLL": (
        {"name": "09L/09R", "true_deg": 89.0, "penalty_s": 0.0},
        {"name": "27R/27L", "true_deg": 269.0, "penalty_s": 5.0 * 60.0},
    ),
}

# Screen thresholds, kt. 25-30 kt crosswind (gust) is allowed but flagged;
# above 30 kt gust, or any mean tailwind above 10 kt, is unflyable on that
# runway. No buffer on the tailwind test -- it's a hard 10 kt.
XWIND_OK_KT = 25.0
XWIND_FLAG_KT = 30.0
TAILWIND_MAX_KT = 10.0


def _headwind_crosswind_gust(u_ms, v_ms, gust_ms, runway_true_deg):
    """Headwind (mean wind) and crosswind (gust) components, kt, for one
    runway -- (n_cand,) in, (n_cand,) out.

    along/cross are the same along-track/cross-track decomposition as
    limits.ground_speed, evaluated against the runway's true heading
    instead of a leg's track: along = u*sin(hdg)+v*cos(hdg) is the
    component of the wind blowing *with* the direction of travel down the
    runway (positive = tailwind), so headwind is its negation. |cross| is
    algebraically equal to W*|sin(wind_from_dir - runway_true)| (the
    textbook crosswind formula in terms of a wind-FROM bearing) without
    ever computing that bearing.

    Gust: i10fg has no direction, so the gust's crosswind is the mean
    wind's crosswind *fraction* (|cross|/W) rescaled onto the gust speed --
    i.e. assume the gust blows from the same direction as the mean wind.
    When the mean wind is (near) calm there's no direction to scale from;
    treat the whole gust as crosswind then (conservative, and rare in
    practice)."""
    s = np.sin(np.radians(runway_true_deg))
    c = np.cos(np.radians(runway_true_deg))
    along = u_ms * s + v_ms * c
    cross = u_ms * c - v_ms * s
    w_mean = np.hypot(u_ms, v_ms)

    headwind_ms = -along
    frac = np.divide(np.abs(cross), w_mean,
                      out=np.ones_like(np.asarray(w_mean, dtype=float)),
                      where=w_mean > 1e-6)
    crosswind_gust_ms = frac * gust_ms

    return headwind_ms / KT_TO_MS, crosswind_gust_ms / KT_TO_MS


def select_runway(u_ms, v_ms, gust_ms, airport):
    """Best runway at `airport` for each candidate, given that candidate's
    interpolated surface wind (u_ms, v_ms, gust_ms, all (n_cand,), m/s --
    departure-hour KJFK wind for the JFK call, touchdown-time EGLL wind for
    the LHR call).

    Screens every runway at this airport, then picks the greatest-headwind
    one among those that pass (crosswind gust <= XWIND_FLAG_KT and tailwind
    mean <= TAILWIND_MAX_KT). If none pass, the candidate is unflyable at
    this airport -- rather than dropping the row, still reports the
    greatest-headwind runway overall (so total_time stays computable) with
    flag "unflyable".

    Returns a dict of (n_cand,) arrays: runway (str), headwind_kt,
    xwind_gust_kt, tailwind_kt, penalty_s, flag (str: "" / "xwind_25_30" /
    "unflyable")."""
    runways = RUNWAYS[airport]
    u_ms = np.asarray(u_ms, dtype=float)
    n_cand = u_ms.shape[0]

    headwind_kt = np.empty((n_cand, len(runways)))
    xwind_gust_kt = np.empty((n_cand, len(runways)))
    for j, rwy in enumerate(runways):
        headwind_kt[:, j], xwind_gust_kt[:, j] = _headwind_crosswind_gust(
            u_ms, v_ms, gust_ms, rwy["true_deg"]
        )
    tailwind_kt = -headwind_kt

    passes = (xwind_gust_kt <= XWIND_FLAG_KT) & (tailwind_kt <= TAILWIND_MAX_KT)
    any_pass = passes.any(axis=1)

    # Among passing runways, greatest headwind wins; where none pass, fall
    # back to the greatest headwind overall (see docstring) rather than
    # dropping the candidate.
    masked_headwind = np.where(passes, headwind_kt, -np.inf)
    chosen_idx = np.where(any_pass,
                           np.argmax(masked_headwind, axis=1),
                           np.argmax(headwind_kt, axis=1))

    row = np.arange(n_cand)
    chosen_headwind_kt = headwind_kt[row, chosen_idx]
    chosen_xwind_gust_kt = xwind_gust_kt[row, chosen_idx]
    chosen_tailwind_kt = tailwind_kt[row, chosen_idx]

    flagged_band = ((chosen_xwind_gust_kt > XWIND_OK_KT)
                     & (chosen_xwind_gust_kt <= XWIND_FLAG_KT))
    unflyable = ~any_pass
    flag = np.where(unflyable, "unflyable", np.where(flagged_band, "xwind_25_30", ""))

    names = np.array([rwy["name"] for rwy in runways])
    penalties_s = np.array([rwy["penalty_s"] for rwy in runways])

    return {
        "runway": names[chosen_idx],
        "headwind_kt": chosen_headwind_kt,
        "xwind_gust_kt": chosen_xwind_gust_kt,
        "tailwind_kt": chosen_tailwind_kt,
        "penalty_s": penalties_s[chosen_idx],
        "flag": flag,
    }


def wind_at(surface_data, airport, query_i8):
    """u10, v10, i10fg (m/s), linearly interpolated between the bracketing
    hourly records in surface_data[airport] at query_i8 (int64 ns UTC
    timestamps, (n_cand,)) -- the same bracketing-hour interpolation
    march_legs uses for the upper-air data. query_i8 outside the data's
    time range is clamped to the nearest bracket, not extrapolated."""
    d = surface_data[airport]
    times_i8 = d["time"].astype("datetime64[ns]").astype("int64")

    idx1 = np.searchsorted(times_i8, query_i8, side="right") - 1
    idx1 = np.clip(idx1, 0, len(times_i8) - 2)
    idx2 = idx1 + 1
    t0, t1 = times_i8[idx1], times_i8[idx2]
    frac = np.clip((query_i8 - t0) / (t1 - t0), 0.0, 1.0)

    def _interp(values):
        return values[idx1] + frac * (values[idx2] - values[idx1])

    return _interp(d["u10"]), _interp(d["v10"]), _interp(d["i10fg"])


def runway_screen(surface_data, airport, query_i8):
    """wind_at + select_runway combined: the interpolated wind at
    query_i8, screened against `airport`'s runways. See select_runway for
    the returned dict's shape."""
    u_ms, v_ms, gust_ms = wind_at(surface_data, airport, query_i8)
    return select_runway(u_ms, v_ms, gust_ms, airport)
