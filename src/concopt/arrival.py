"""Arrival model: decel waypoint (BARIX) to touchdown.

Replaces the flat DECEL_DESCENT_S = 35 min / 2.0 t placeholder. Four segments
covering a ground distance fixed by the route:

  1. decel    cruise Mach -> M1.0, cruise_fl -> decel_end_fl  (conc_descent.csv)
  2. level    M0.95 at decel_end_fl, for whatever distance is left over
              (conc_subsonic_cruise.csv, indexed by the mass actually
              flying it -- mass_at_barix_t minus the decel burn, not TOW)
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

# Typical mass at the decel waypoint -- only used as mass_at_barix_t's
# default, for the many tests below that exercise wind/band/flag behaviour
# and don't care what the level segment's fuel table lookup lands on. Real
# callers (search.py, via fuel._arrival_from_march) always pass the march's
# own weight_at_barix; MASS IS THE TRAP here (see conc_data.subsonic_cruise
# and _arrival_for_speed below) so nothing downstream of a real call should
# ever rely on this default firing.
#
# 118, not 110: the subsonic table's lowest levels (FL290-330, which cover
# the 380 kt schedule's FL312 decel_end_fl) are only published down to
# 110 t, and the ~1-1.5 t decel burn taken off mass_at_barix_t before that
# lookup would otherwise land BELOW the table's own floor -- a real, table-
# shaped envelope edge, not a bug, but not what an arbitrary test default
# should be tripping over.
DEFAULT_MASS_AT_BARIX_T = 118.0

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

_FLAG_KEYS = ("level_nm_clamped", "cruise_fl_clamped", "level_gs_nonpositive",
              "level_mass_outside_envelope", "wind_fl_clamped")

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
    """Pull (wind_kt, temp_k, fl_clamped) out of `wind_at_fl`, all as
    (n_cand,) arrays.

    `level_fl` is always broadcast to (n_cand,) before the call, so the
    callable never has to guess which candidates it is being asked about --
    that ambiguity is what made an earlier band-masked version crash the
    moment candidates straddled the ISA-10 boundary.

    fl_clamped (B6): True where the requested level_fl fell outside
    wind_at_fl's own source data span and was clamped rather than genuinely
    read -- search._build_arrival_wind_fn's stitched profile reports this
    via an optional "fl_clamped" dict key; a wind_at_fl that doesn't
    (a bare (wind_kt, temp_k) pair, or a dict without the key -- every
    test mock in tests/test_arrival.py, for instance) is treated as never
    clamped rather than erroring, so this stays backward compatible."""
    level_fl = np.broadcast_to(np.asarray(level_fl, float), (n_cand,))
    atm = wind_at_fl(level_fl)
    if isinstance(atm, dict):
        wind_kt, temp_k = atm["wind_kt"], atm["temp_k"]
        fl_clamped = atm.get("fl_clamped", False)
    else:
        wind_kt, temp_k = atm[0], atm[1]
        fl_clamped = atm[2] if len(atm) > 2 else False
    wind_kt = np.broadcast_to(np.asarray(wind_kt, float), (n_cand,))
    temp_k = np.broadcast_to(np.asarray(temp_k, float), (n_cand,))
    fl_clamped = np.broadcast_to(np.asarray(fl_clamped, bool), (n_cand,))
    return wind_kt, temp_k, fl_clamped


def _table_by_band(fn, level_fl, speed_kt, warm, keys):
    """`fn` evaluated in BOTH temperature bands over all candidates, then
    selected per candidate with np.where. Two whole-array lookups instead of
    masking the candidate axis -- masking would hand `wind_at_fl` a subset it
    cannot identify."""
    hot = fn(level_fl, speed_kt, BAND_WARM)
    cold = fn(level_fl, speed_kt, BAND_COLD)
    return {k: np.where(warm, hot[k], cold[k]) for k in keys}


def _arrival_for_speed(cruise_fl, arrival_nm, wind_at_fl, warm, speed_kt, n_cand,
                        mass_at_barix_t):
    """One descent speed schedule, vectorised over every candidate at once."""
    cols = ("fuel_t", "time_min", "dist_zero_wind_nm")

    # --- 1. decel: cruise_fl -> decel_end_fl -------------------------------
    decel = _table_by_band(conc_data.decel_to_mach1, cruise_fl, speed_kt, warm, cols)
    # decel_end_fl is constant within a speed schedule AND identical in both
    # bands (325 -> FL383, 350 -> FL350, 380 -> FL312), so it is a true scalar.
    decel_end_fl = float(
        conc_data.decel_to_mach1(cruise_fl.flat[0], speed_kt, BAND_WARM)["decel_end_fl"]
    )

    decel_wind, _, decel_wind_clamped = _wind_temp(
        wind_at_fl, (cruise_fl + decel_end_fl) / 2.0, n_cand
    )
    decel_nm = conc_data.dist_with_wind(
        decel["dist_zero_wind_nm"], decel["time_min"], decel_wind
    )

    # --- 3. descent: decel_end_fl -> 1,500 ft ------------------------------
    # Computed before the level segment because level_nm is the leftover.
    descent = _table_by_band(
        conc_data.descent_to_1500ft, decel_end_fl, speed_kt, warm, cols
    )
    descent_mid_fl = (decel_end_fl + _DESCENT_END_FL) / 2.0
    descent_wind, _, descent_wind_clamped = _wind_temp(wind_at_fl, descent_mid_fl, n_cand)
    descent_nm = conc_data.dist_with_wind(
        descent["dist_zero_wind_nm"], descent["time_min"], descent_wind
    )

    # --- 2. level: M0.95 at decel_end_fl for the leftover distance ---------
    level_nm_raw = arrival_nm - APPROACH_NM - decel_nm - descent_nm
    level_nm_clamped = level_nm_raw < 0.0
    level_nm = np.maximum(level_nm_raw, 0.0)

    level_wind, level_temp_k, level_wind_clamped = _wind_temp(wind_at_fl, decel_end_fl, n_cand)
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

    # MASS IS THE TRAP: the subsonic table is indexed by aircraft mass, and
    # the aircraft reaches the level segment at roughly mass_at_barix_t minus
    # the decel burn (~105-120 t), not TOW (~160-185 t) and not
    # mass_at_barix_t untouched either -- reading it at the wrong mass here
    # silently halves the specific range and doubles level_fuel_t.
    mass_at_level_t = np.broadcast_to(
        np.asarray(mass_at_barix_t, float), (n_cand,)
    ) - decel["fuel_t"]
    isa_t_k_at_level, _ = atmos.isa(decel_end_fl * 100.0 * 0.3048)
    isa_dev_c_at_level = level_temp_k - isa_t_k_at_level
    subsonic = conc_data.subsonic_cruise(decel_end_fl, mass_at_level_t, isa_dev_c_at_level)
    specific_range_nm_per_t = subsonic["specific_range_nm_per_t"]

    # Spec'd as ground distance / specific range. Note this makes level fuel
    # wind-independent: a headwind lengthens the time aloft without raising
    # the burn.
    level_mass_outside_envelope = ~np.isfinite(specific_range_nm_per_t)
    # Zero leftover distance burns no fuel even where the table has no entry
    # for this mass/level/ISA -- same convention as level_time_min above.
    with np.errstate(invalid="ignore"):
        level_fuel_t = np.where(
            level_nm == 0.0, 0.0, level_nm / specific_range_nm_per_t
        )

    # Any of the three wind_at_fl reads (decel midpoint, descent midpoint,
    # level segment) landing outside the source data's span means this
    # schedule's numbers rest on a clamped read somewhere -- OR them
    # together into one flag rather than three, since the caller cares
    # whether the *segment* used a clamped wind, not which sub-lookup did.
    wind_fl_clamped = decel_wind_clamped | descent_wind_clamped | level_wind_clamped

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
        "level_mass_outside_envelope": level_mass_outside_envelope,
        "wind_fl_clamped": wind_fl_clamped,
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


def arrival(cruise_fl, arrival_nm, wind_at_fl, isa_dev_at_cruise,
            mass_at_barix_t=DEFAULT_MASS_AT_BARIX_T, speed=380):
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
            returning either a dict with "wind_kt"/"temp_k" (and optionally
            "fl_clamped", a (n_cand,) bool -- True where the request fell
            outside the source data's own span and was clamped rather than
            genuinely read) or a (wind_kt, temp_k) / (wind_kt, temp_k,
            fl_clamped) tuple. Wind is the along-track component in kt,
            positive for a tailwind; temperature is static, K. Missing
            fl_clamped is treated as never-clamped (see search.
            _build_arrival_wind_fn for the real implementation, B6).
        isa_dev_at_cruise: ISA deviation in the arrival area at cruise level,
            °C. Picks the descent temperature band (two bands, not the climb
            table's three).
        mass_at_barix_t: aircraft mass (t) AT THE DECEL WAYPOINT, before the
            decel burn -- search.py's march calls this weight_at_barix.
            MASS IS THE TRAP: the subsonic cruise table (conc_data.
            subsonic_cruise) that prices the level segment is indexed by the
            mass actually flying it, which is mass_at_barix_t minus that
            schedule's own decel_fuel_t (roughly 105-120 t, not TOW's
            160-185 t) -- _arrival_for_speed subtracts it internally, per
            schedule, since decel_fuel_t is schedule-dependent. Defaults to
            DEFAULT_MASS_AT_BARIX_T for the many tests here that don't care
            what the level segment's fuel lands on; a real caller always
            passes the march's own weight_at_barix.
        speed: 380 (default) forces the 380 kt schedule. With the real
            subsonic table, 380 kt buys ~1.6 min here for ~0.44 t of extra
            fuel (not the old placeholder's ~1.4 t) -- worth ~0.1 min of
            climb time cold and ~0.6 min warm, against the 1.6 min saved, so
            on TIME AND FUEL ALONE 380 kt wins in every temperature band and
            the trade-off other callers used to have to make on
            `by_schedule` is no longer close. BUT 380's decel_end_fl (FL312)
            sits in conc_subsonic_cruise.csv's FL290-330 band, whose
            published floor is 110 t -- comfortably inside a warm/light
            day's mass_at_level_t, but above it for anything under roughly
            178 t TOW at ISA+0 (see search.py's own worked climb/cruise burn
            numbers), which is a good deal of the real 160-185 t TOW range.
            At those masses 380 kt is simply not computable from this table
            (level_mass_outside_envelope, NaN fuel_t) even though its
            time_min still looks fastest -- "auto" accounts for this (it
            disqualifies a NaN-fuel schedule before picking by time) and
            falls back to 350 or 325 kt (100 t floors, comfortably wider)
            whenever 380 kt is infeasible, so real callers (search.py, via
            fuel._arrival_from_march) request "auto", not this default.
            325/350 force the other two. Use `by_schedule` to re-decide on
            *total* time instead of arrival time.

    Returns:
        dict with (n_cand,) arrays:
            time_min, fuel_t, schedule_kt, level_fl, level_nm,
            decel_nm, descent_nm,
            decel_time_min, level_time_min, descent_time_min,
            decel_fuel_t, level_fuel_t, descent_fuel_t,
            flags (str, "" when clean; names joined by ";")
        plus per-flag booleans level_nm_clamped / cruise_fl_clamped /
        level_gs_nonpositive / level_mass_outside_envelope /
        wind_fl_clamped, and `by_schedule`: {325: {...}, 350: {...},
        380: {...}}, each the full breakdown for that forced schedule.

    Flags rather than exceptions:
        level_nm_clamped      descent did not fit the available distance;
                              level_nm clamped to 0
        cruise_fl_clamped     cruise_fl outside the decel table's 470-600
        level_gs_nonpositive  headwind >= M0.95 TAS; level_time_min is inf
        level_mass_outside_envelope
                              mass_at_level_t (mass_at_barix_t minus the
                              decel burn) fell outside conc_subsonic_cruise
                              .csv's published envelope at this level/ISA;
                              level_fuel_t (and so fuel_t) is NaN. Should
                              never fire at real arrival masses.
        wind_fl_clamped       the decel midpoint, descent midpoint, or
                              level segment read wind/temp outside
                              wind_at_fl's own source span and got a
                              clamped value instead of a genuine one (B6).
                              A REAL number still comes back -- this flag
                              is the only thing that says so isn't a
                              genuine reading; see search.
                              _build_arrival_wind_fn's stitched FL183-605
                              profile, which still clamps below FL183 on
                              the descent segment's own midpoint (as low
                              as FL163.5 on the 380 kt schedule).
    """
    if speed != "auto" and speed not in SCHEDULES_KT:
        raise ValueError(
            f"speed must be 'auto' or one of {SCHEDULES_KT}; got {speed!r}"
        )

    arrival_nm = np.atleast_1d(np.asarray(arrival_nm, float))
    isa_dev_at_cruise = np.atleast_1d(np.asarray(isa_dev_at_cruise, float))
    cruise_fl = np.atleast_1d(np.asarray(cruise_fl, float))
    mass_at_barix_t = np.atleast_1d(np.asarray(mass_at_barix_t, float))
    cruise_fl, arrival_nm, isa_dev_at_cruise, mass_at_barix_t = (
        np.array(a) for a in
        np.broadcast_arrays(cruise_fl, arrival_nm, isa_dev_at_cruise, mass_at_barix_t)
    )
    n_cand = arrival_nm.shape[0]

    cruise_fl_clamped_to = np.clip(cruise_fl, CRUISE_FL_MIN, CRUISE_FL_MAX)
    cruise_fl_clamped = cruise_fl != cruise_fl_clamped_to
    cruise_fl = cruise_fl_clamped_to

    warm = _band_is_warm(isa_dev_at_cruise)

    by_schedule = {
        spd: _arrival_for_speed(
            cruise_fl, arrival_nm, wind_at_fl, warm, spd, n_cand, mass_at_barix_t
        )
        for spd in SCHEDULES_KT
    }

    if speed == "auto":
        times = np.stack([by_schedule[s]["time_min"] for s in SCHEDULES_KT], axis=-1)
        fuels = np.stack([by_schedule[s]["fuel_t"] for s in SCHEDULES_KT], axis=-1)
        # level_time_min never touches the subsonic table (it's TAS + wind
        # only), so a schedule with a NaN fuel_t (mass_at_level_t outside
        # conc_subsonic_cruise.csv's envelope -- 380 kt's low decel_end_fl,
        # FL312, is only published down to 110 t, well inside the real
        # 160-185 t TOW range) can still show the smallest time_min. argmin
        # doesn't know that "fastest" is meaningless without a fuel number,
        # so disqualify it here rather than silently picking an infeasible
        # schedule ahead of a feasible slower one.
        times = np.where(np.isfinite(fuels), times, np.inf)
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
    for key in ("level_nm_clamped", "level_gs_nonpositive", "level_mass_outside_envelope",
                "wind_fl_clamped"):
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


def flat_arrival(cruise_fl, arrival_nm, wind_at_fl, isa_dev_at_cruise,
                  mass_at_barix_t=DEFAULT_MASS_AT_BARIX_T, speed="auto",
                  *, decel_descent_min):
    """Drop-in replacement for arrival() with the SAME call signature (so
    callers -- fuel.fixed_point_fuel_iteration in particular -- don't need
    to know which one they're holding), but returning the flat legacy pair
    this module replaces: decel_descent_min minutes at LEGACY_FLAT_FUEL_T
    tonnes, everything else zeroed. wind_at_fl, mass_at_barix_t and speed are
    accepted and ignored -- --decel-descent-min forces this instead of the
    real model, for comparing old and new numbers on equal terms.

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
        "level_gs_nonpositive": falses, "level_mass_outside_envelope": falses,
        "wind_fl_clamped": falses,
        "flags": np.full(n_cand, "", dtype=object).astype(str),
        "by_schedule": {},
    }
