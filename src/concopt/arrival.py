"""Arrival model: decel waypoint (BARIX) to touchdown.

Replaces the flat DECEL_DESCENT_S = 35 min / 2.0 t placeholder. Four segments
covering a ground distance fixed by the route:

  1. decel    cruise Mach -> M1.0, cruise_fl -> decel_end_fl  (conc_descent.csv)
  2. level    M0.95 at decel_end_fl, for whatever distance is left over
  3. descent  decel_end_fl -> 1,500 ft                        (conc_descent.csv)
  4. approach 1,500 ft -> touchdown, a fixed allowance

      level_nm = arrival_nm - decel_nm - descent_nm - APPROACH_NM

`conc_data.descent_direct_from_cruise` is deliberately NOT used: it reaches
1,500 ft in ~194 nm and this route has ~307, which would leave ~113 nm of
cruise at 1,500 ft.

Everything is vectorised across candidates: every argument is a (n_cand,)
array (or a scalar broadcast to one) and every returned array is too. There is
no per-candidate and no per-schedule Python loop -- all three speed schedules
and both temperature bands are evaluated as whole-array table lookups and
selected with `np.where`.

This module never reads era5 or .npz files. Winds arrive through the
`wind_at_fl` callable so that wiring it into search.py stays a caller concern.
"""

import numpy as np

from . import atmos
from .data import conc_data

# --- Approach allowance, 1,500 ft to touchdown -------------------------------
# ASSUMPTIONS, not table values. The Phase C flight recorder measures these.
APPROACH_NM = 5.0
APPROACH_MIN = 1.5
APPROACH_FUEL_T = 0.3

# --- Level segment at M0.95 --------------------------------------------------
LEVEL_MACH = 0.95

# PLACEHOLDER -- THE WEAKEST NUMBER IN THIS MODEL.
# No subsonic cruise table exists. ~23 t/h at 550 kt TAS, mid-range against
# published Concorde subsonic figures of 21-26 t/h. Sizes a ~4 t term, so the
# 21-26 t/h spread is worth roughly +/-0.4 t of arrival fuel, which the fuel.py
# fixed point then amplifies into climb time. Replace with a real table if one
# ever turns up.
SUBSONIC_SR_NM_PER_T = 24.0

# conc_descent.csv has TWO temperature bands, not the three in conc_climb.csv.
BAND_WARM = "above_isa_minus_10"
BAND_COLD = "isa_minus_10_and_below"

SCHEDULES_KT = (325, 350, 380)

# Decel table bounds (conc_descent.csv decel_to_mach1 rows).
CRUISE_FL_MIN = 470.0
CRUISE_FL_MAX = 600.0

# Altitude the descent segment ends at, in FL units, for its mid-level wind.
_DESCENT_END_FL = 15.0

_SEGMENT_KEYS = (
    "time_min", "fuel_t", "level_fl", "level_nm", "decel_nm", "descent_nm",
    "decel_time_min", "level_time_min", "descent_time_min",
    "decel_fuel_t", "level_fuel_t", "descent_fuel_t", "level_wind_kt",
)

_FLAG_KEYS = ("level_nm_clamped", "cruise_fl_clamped", "level_gs_nonpositive")

# Legacy comparison only. The flat (DECEL_DESCENT_S=35 min, DESCENT_FUEL_T=
# 2.0 t) pair this module replaces -- see flat_arrival, which --decel-
# descent-min forces instead of the real model below, for comparing old and
# new numbers on equal terms.
LEGACY_FLAT_FUEL_T = 2.0


def _band_is_warm(isa_dev_at_cruise):
    """True where the arrival is in conc_descent.csv's `above_isa_minus_10`
    band. Two bands only -- do not reuse the three-band climb selector."""
    return np.asarray(isa_dev_at_cruise, float) > -10.0


def _wind_temp(wind_at_fl, level_fl, n_cand):
    """Pull (wind_kt, temp_k) out of `wind_at_fl`, both as (n_cand,) float.

    `level_fl` is always broadcast to (n_cand,) before the call, so the
    callable never has to guess which candidates it is being asked about --
    that ambiguity is what made an earlier band-masked version crash the
    moment candidates straddled the ISA-10 boundary."""
    level_fl = np.broadcast_to(np.asarray(level_fl, float), (n_cand,))
    atm = wind_at_fl(level_fl)
    if isinstance(atm, dict):
        wind_kt, temp_k = atm["wind_kt"], atm["temp_k"]
    else:
        wind_kt, temp_k = atm[0], atm[1]
    wind_kt = np.broadcast_to(np.asarray(wind_kt, float), (n_cand,))
    temp_k = np.broadcast_to(np.asarray(temp_k, float), (n_cand,))
    return wind_kt, temp_k


def _table_by_band(fn, level_fl, speed_kt, warm, keys):
    """`fn` evaluated in BOTH temperature bands over all candidates, then
    selected per candidate with np.where. Two whole-array lookups instead of
    masking the candidate axis -- masking would hand `wind_at_fl` a subset it
    cannot identify."""
    hot = fn(level_fl, speed_kt, BAND_WARM)
    cold = fn(level_fl, speed_kt, BAND_COLD)
    return {k: np.where(warm, hot[k], cold[k]) for k in keys}


def _arrival_for_speed(cruise_fl, arrival_nm, wind_at_fl, warm, speed_kt, n_cand):
    """One descent speed schedule, vectorised over every candidate at once."""
    cols = ("fuel_t", "time_min", "dist_zero_wind_nm")

    # --- 1. decel: cruise_fl -> decel_end_fl -------------------------------
    decel = _table_by_band(conc_data.decel_to_mach1, cruise_fl, speed_kt, warm, cols)
    # decel_end_fl is constant within a speed schedule AND identical in both
    # bands (325 -> FL383, 350 -> FL350, 380 -> FL312), so it is a true scalar.
    decel_end_fl = float(
        conc_data.decel_to_mach1(cruise_fl.flat[0], speed_kt, BAND_WARM)["decel_end_fl"]
    )

    decel_wind, _ = _wind_temp(wind_at_fl, (cruise_fl + decel_end_fl) / 2.0, n_cand)
    decel_nm = conc_data.dist_with_wind(
        decel["dist_zero_wind_nm"], decel["time_min"], decel_wind
    )

    # --- 3. descent: decel_end_fl -> 1,500 ft ------------------------------
    # Computed before the level segment because level_nm is the leftover.
    descent = _table_by_band(
        conc_data.descent_to_1500ft, decel_end_fl, speed_kt, warm, cols
    )
    descent_mid_fl = (decel_end_fl + _DESCENT_END_FL) / 2.0
    descent_wind, _ = _wind_temp(wind_at_fl, descent_mid_fl, n_cand)
    descent_nm = conc_data.dist_with_wind(
        descent["dist_zero_wind_nm"], descent["time_min"], descent_wind
    )

    # --- 2. level: M0.95 at decel_end_fl for the leftover distance ---------
    level_nm_raw = arrival_nm - APPROACH_NM - decel_nm - descent_nm
    level_nm_clamped = level_nm_raw < 0.0
    level_nm = np.maximum(level_nm_raw, 0.0)

    level_wind, level_temp_k = _wind_temp(wind_at_fl, decel_end_fl, n_cand)
    # atmos.py is the single validated speed-of-sound path -- do not add another.
    level_tas_kt = atmos.speed_of_sound(level_temp_k) * LEVEL_MACH / atmos.KT_TO_MS
    level_gs_kt = level_tas_kt + level_wind

    # A headwind exceeding TAS means the aircraft never arrives. Flag it rather
    # than dividing by a non-positive ground speed.
    level_gs_nonpositive = level_gs_kt <= 0.0
    safe_gs = np.where(level_gs_nonpositive, np.nan, level_gs_kt)
    with np.errstate(invalid="ignore"):
        level_time_min = np.where(
            level_gs_nonpositive, np.inf, level_nm / safe_gs * 60.0
        )
    # Zero leftover distance costs no time even at a non-positive ground speed.
    level_time_min = np.where(level_nm == 0.0, 0.0, level_time_min)
    level_gs_nonpositive &= level_nm > 0.0

    # Spec'd as ground distance / specific range. Note this makes level fuel
    # wind-independent: a headwind lengthens the time aloft without raising the
    # burn. Immaterial against SUBSONIC_SR_NM_PER_T's own 21-26 t/h spread.
    level_fuel_t = level_nm / SUBSONIC_SR_NM_PER_T

    out = {
        "decel_nm": decel_nm,
        "descent_nm": descent_nm,
        "level_nm": level_nm,
        "level_fl": np.full(n_cand, decel_end_fl),
        "level_wind_kt": np.broadcast_to(level_wind, (n_cand,)).astype(float),
        "decel_time_min": decel["time_min"],
        "level_time_min": level_time_min,
        "descent_time_min": descent["time_min"],
        "decel_fuel_t": decel["fuel_t"],
        "level_fuel_t": level_fuel_t,
        "descent_fuel_t": descent["fuel_t"],
        "level_nm_clamped": level_nm_clamped,
        "level_gs_nonpositive": level_gs_nonpositive,
    }
    out["time_min"] = (
        out["decel_time_min"] + out["level_time_min"]
        + out["descent_time_min"] + APPROACH_MIN
    )
    out["fuel_t"] = (
        out["decel_fuel_t"] + out["level_fuel_t"]
        + out["descent_fuel_t"] + APPROACH_FUEL_T
    )
    return out


def arrival(cruise_fl, arrival_nm, wind_at_fl, isa_dev_at_cruise, speed="auto"):
    """Arrival time and fuel from the decel waypoint to touchdown.

    Vectorised across candidates throughout: every argument is a (n_cand,)
    array (scalars broadcast) and every returned array is (n_cand,).

    Args:
        cruise_fl: cruise flight level. Clamped to 470-600, the decel table's
            span, and flagged where clamping bit.
        arrival_nm: ground distance from the decel waypoint to touchdown. Comes
            from the route -- sum the post-decel legs; do NOT pass a constant,
            since the .pln is a command-line input and can change.
        wind_at_fl: callable taking a (n_cand,) array of flight levels and
            returning either a dict with "wind_kt"/"temp_k" or a
            (wind_kt, temp_k) pair, each (n_cand,). Wind is the along-track
            component in kt, positive for a tailwind; temperature is static, K.
        isa_dev_at_cruise: ISA deviation in the arrival area at cruise level,
            °C. Picks the descent temperature band (two bands, not the climb
            table's three).
        speed: "auto" (default) picks, per candidate, the schedule minimising
            ARRIVAL time. 325/350/380 forces one. Note that the time-optimal
            schedule is not the total-time-optimal one once fuel feeds back
            through the climb -- 380 kt buys ~1.6 min here for ~1.4 t, which
            costs ~0.3 min of climb on a cold day but ~1.9 min on a warm one.
            Use `by_schedule` to make that choice on total time instead.

    Returns:
        dict with (n_cand,) arrays:
            time_min, fuel_t, schedule_kt, level_fl, level_nm,
            decel_nm, descent_nm,
            decel_time_min, level_time_min, descent_time_min,
            decel_fuel_t, level_fuel_t, descent_fuel_t,
            flags (str, "" when clean; names joined by ";")
        plus per-flag booleans level_nm_clamped / cruise_fl_clamped /
        level_gs_nonpositive, and `by_schedule`: {325: {...}, 350: {...},
        380: {...}}, each the full breakdown for that forced schedule.

    Flags rather than exceptions:
        level_nm_clamped      descent did not fit the available distance;
                              level_nm clamped to 0
        cruise_fl_clamped     cruise_fl outside the decel table's 470-600
        level_gs_nonpositive  headwind >= M0.95 TAS; level_time_min is inf
    """
    if speed != "auto" and speed not in SCHEDULES_KT:
        raise ValueError(
            f"speed must be 'auto' or one of {SCHEDULES_KT}; got {speed!r}"
        )

    arrival_nm = np.atleast_1d(np.asarray(arrival_nm, float))
    isa_dev_at_cruise = np.atleast_1d(np.asarray(isa_dev_at_cruise, float))
    cruise_fl = np.atleast_1d(np.asarray(cruise_fl, float))
    cruise_fl, arrival_nm, isa_dev_at_cruise = (
        np.array(a) for a in
        np.broadcast_arrays(cruise_fl, arrival_nm, isa_dev_at_cruise)
    )
    n_cand = arrival_nm.shape[0]

    cruise_fl_clamped_to = np.clip(cruise_fl, CRUISE_FL_MIN, CRUISE_FL_MAX)
    cruise_fl_clamped = cruise_fl != cruise_fl_clamped_to
    cruise_fl = cruise_fl_clamped_to

    warm = _band_is_warm(isa_dev_at_cruise)

    by_schedule = {
        spd: _arrival_for_speed(
            cruise_fl, arrival_nm, wind_at_fl, warm, spd, n_cand
        )
        for spd in SCHEDULES_KT
    }

    if speed == "auto":
        times = np.stack([by_schedule[s]["time_min"] for s in SCHEDULES_KT], axis=-1)
        schedule_kt = np.asarray(SCHEDULES_KT)[np.argmin(times, axis=-1)]
    else:
        schedule_kt = np.full(n_cand, speed)

    # Gather the chosen schedule per candidate. Three np.where passes, not a
    # per-candidate loop.
    result = {}
    for key in _SEGMENT_KEYS:
        picked = by_schedule[SCHEDULES_KT[0]][key]
        for spd in SCHEDULES_KT[1:]:
            picked = np.where(schedule_kt == spd, by_schedule[spd][key], picked)
        result[key] = picked

    flags = {"cruise_fl_clamped": cruise_fl_clamped}
    for key in ("level_nm_clamped", "level_gs_nonpositive"):
        picked = by_schedule[SCHEDULES_KT[0]][key]
        for spd in SCHEDULES_KT[1:]:
            picked = np.where(schedule_kt == spd, by_schedule[spd][key], picked)
        flags[key] = picked

    result.update(flags)
    result["schedule_kt"] = schedule_kt
    result["by_schedule"] = by_schedule

    flag_str = np.full(n_cand, "", dtype=object)
    for name in _FLAG_KEYS:
        flag_str = np.where(
            flags[name],
            np.where(flag_str == "", name, flag_str + ";" + name),
            flag_str,
        )
    result["flags"] = flag_str.astype(str)

    return result


def flat_arrival(cruise_fl, arrival_nm, wind_at_fl, isa_dev_at_cruise, speed="auto",
                  *, decel_descent_min):
    """Drop-in replacement for arrival() with the SAME call signature (so
    callers -- fuel.fixed_point_fuel_iteration in particular -- don't need
    to know which one they're holding), but returning the flat legacy pair
    this module replaces: decel_descent_min minutes at LEGACY_FLAT_FUEL_T
    tonnes, everything else zeroed. wind_at_fl and speed are accepted and
    ignored -- --decel-descent-min forces this instead of the real model,
    for comparing old and new numbers on equal terms.

    n_cand is read off cruise_fl/arrival_nm/isa_dev_at_cruise the same way
    arrival() itself does, so scalars broadcast to one candidate."""
    cruise_fl, arrival_nm, isa_dev_at_cruise = (
        np.array(a) for a in np.broadcast_arrays(
            np.atleast_1d(np.asarray(cruise_fl, float)),
            np.atleast_1d(np.asarray(arrival_nm, float)),
            np.atleast_1d(np.asarray(isa_dev_at_cruise, float)),
        )
    )
    n_cand = cruise_fl.shape[0]
    zeros = np.zeros(n_cand)
    falses = np.zeros(n_cand, dtype=bool)

    return {
        "time_min": np.full(n_cand, float(decel_descent_min)),
        "fuel_t": np.full(n_cand, LEGACY_FLAT_FUEL_T),
        "schedule_kt": np.zeros(n_cand, dtype=int),  # N/A -- flat override
        "level_fl": zeros, "level_nm": zeros, "decel_nm": zeros, "descent_nm": zeros,
        "level_wind_kt": zeros,
        "decel_time_min": zeros,
        "level_time_min": np.full(n_cand, float(decel_descent_min)),
        "descent_time_min": zeros,
        "decel_fuel_t": zeros,
        "level_fuel_t": np.full(n_cand, LEGACY_FLAT_FUEL_T),
        "descent_fuel_t": zeros,
        "level_nm_clamped": falses, "cruise_fl_clamped": falses,
        "level_gs_nonpositive": falses,
        "flags": np.full(n_cand, "", dtype=object).astype(str),
        "by_schedule": {},
    }
