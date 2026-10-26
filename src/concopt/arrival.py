"""Arrival model: the deceleration, then decel waypoint to touchdown.

The decel waypoint is where the aircraft must be SUBSONIC BY -- the
deceleration ends there, it doesn't start there. So the decel is flown over
the last decel_nm of the climb+cruise span, and the supersonic cruise the
march credited over that stretch is backed out (march_tail, see arrival()).
Four segments:

  1. decel    cruise Mach -> M1.0, cruise_fl -> decel_end_fl, ENDING at the
              decel waypoint                                   (conc_descent.csv)
  2. level    M0.95 at decel_end_fl, for whatever distance is left over
              (conc_subsonic_cruise.csv, indexed by the mass actually
              flying it -- the mass at decel start minus the decel burn, not TOW)
  3. descent  decel_end_fl -> 1,500 ft                        (conc_descent.csv)
  4. approach 1,500 ft -> touchdown, a fixed allowance

      level_nm = arrival_nm - descent_nm - APPROACH_NM

where arrival_nm is the route distance from the decel waypoint to touchdown.
time_min/fuel_t are NET of the replaced cruise -- measured from the march's
own arrival at the decel waypoint -- so a caller's march total plus
time_min is still brake release to touchdown, and the fuel split still adds
up, without either caller knowing where the decel started.

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
from .params import (APPROACH_FUEL_T, APPROACH_MIN, APPROACH_NM, BAND_COLD, BAND_WARM,  # noqa: F401
                     CRUISE_FL_MAX, CRUISE_FL_MIN, DEFAULT_MASS_AT_BARIX_T, DESCENT_END_FL,
                     DESCENT_END_FT, DESCENT_GRID_STEP_FT, DESCENT_GRID_TOP_FT,
                     DESCENT_RATE_SCHEDULE_KT, FASTEST_SCHEDULE_KT, FT_TO_M, GATE_KIAS,
                     LEVEL_MACH, REVERSER_FACTOR, REVERSER_MAX_MIN, SCHEDULES_KT,
                     SPEED_LIMIT_FL, SPEED_LIMIT_KIAS, SUBSONIC_LEVELS_FL)

_SEGMENT_KEYS = (
    "time_min", "fuel_t", "level_fl", "level_nm", "decel_nm", "descent_nm",
    "decel_time_min", "level_time_min", "descent_time_min",
    "decel_fuel_t", "level_fuel_t", "descent_fuel_t", "level_wind_kt",
    "decel_wind_kt", "descent_wind_kt",
    "replaced_cruise_time_min", "replaced_cruise_fuel_t", "mass_at_decel_start_t",
    "after_waypoint_time_min", "decel_end_fl", "level_gs_kt",
    "transition_nm", "transition_time_min", "transition_fuel_t",
    "reverser_min", "descent_top_kias", "approach_time_min",
)

_FLAG_KEYS = ("level_nm_clamped", "cruise_fl_clamped", "level_gs_nonpositive",
              "level_mass_outside_envelope", "wind_fl_clamped", "decel_starts_before_cruise")


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


def _to_1500ft(level_fl, band):
    """conc_descent.csv's subsonic descent (DESCENT_RATE_SCHEDULE_KT) from
    level_fl down to 1,500 ft: cumulative (time_min, fuel_t, zero-wind
    dist_nm). The table's
    lowest row (FL30 warm, FL50 cold) still has time to go; below it this is
    linear down to exactly zero at 1,500 ft, instead of the clamp that would
    leave FL15 reading that row's 0.8-1.4 min."""
    level_fl = np.asarray(level_fl, float)
    floor_fl = conc_data.descent_to_1500ft_floor_fl(DESCENT_RATE_SCHEDULE_KT, band)
    row = conc_data.descent_to_1500ft(np.maximum(level_fl, floor_fl),
                                       DESCENT_RATE_SCHEDULE_KT, band)
    f = np.clip((level_fl - DESCENT_END_FL) / (floor_fl - DESCENT_END_FL), 0.0, 1.0)
    below = level_fl < floor_fl
    return tuple(np.where(below, row[k] * f, row[k])
                 for k in ("time_min", "fuel_t", "dist_zero_wind_nm"))


def _tas_kt(cas_kt, p_pa, temp_k):
    mach = atmos.mach_from_cas_subsonic(np.asarray(cas_kt, float) * atmos.KT_TO_MS, p_pa)
    return mach * atmos.speed_of_sound(temp_k) / atmos.KT_TO_MS


def _max_cas_kt(alt_ft, p_pa, weight_t):
    """The fastest the airframe allows: M0.95 until that meets the CAS limit
    (conc_cas_limit.csv, by weight), the CAS limit below."""
    mach_cas_kt = atmos.cas_from_mach(LEVEL_MACH, p_pa) / atmos.KT_TO_MS
    return np.minimum(mach_cas_kt, conc_data.cas_limit_kt(alt_ft, weight_t))


def _descent_grid(wind_at_fl, warm, n_cand):
    """The subsonic descent on a fixed altitude grid, DESCENT_GRID_TOP_FT down
    to DESCENT_END_FT in DESCENT_GRID_STEP_FT steps, for every candidate:
    T/F (n_cand, K), the table's cumulative time/fuel from each grid
    altitude down to 1,500 ft at the normal descent rate, and cum_nm
    (n_cand, K), ground distance from the top of the grid: the table's own
    zero-wind distance (which, like its time, already includes slowing
    down through FL100 and to the gate) plus wind over each step's time,
    the conc_data.dist_with_wind relation. Built once per arrival() call;
    every schedule and level reads its own stretch of it (_at)."""
    alt_ft = np.arange(DESCENT_GRID_TOP_FT, DESCENT_END_FT - 1.0, -DESCENT_GRID_STEP_FT)
    mid_ft = (alt_ft[:-1] + alt_ft[1:]) / 2.0

    wind_kt = np.empty((n_cand, len(mid_ft)))
    clamped = np.zeros(n_cand, bool)
    for j, ft in enumerate(mid_ft):
        wind_kt[:, j], _, fl_clamped = _wind_temp(wind_at_fl, ft / 100.0, n_cand)
        clamped |= fl_clamped

    tables = {band: _to_1500ft(alt_ft / 100.0, band) for band in (BAND_WARM, BAND_COLD)}
    t, f, d = (np.where(warm[:, None], tables[BAND_WARM][k][None, :], tables[BAND_COLD][k][None, :])
               for k in range(3))
    step_nm = (d[:, :-1] - d[:, 1:]) + wind_kt * (t[:, :-1] - t[:, 1:]) / 60.0
    cum_nm = np.concatenate([np.zeros((n_cand, 1)), np.cumsum(step_nm, axis=1)], axis=1)
    return dict(T=t, F=f, cum_nm=cum_nm, clamped=clamped)


def _at(values, alt_ft):
    """values (n_cand, K) on the descent grid, linearly interpolated at
    alt_ft -- a scalar, or (n_cand,) for a per-candidate altitude."""
    idx = (DESCENT_GRID_TOP_FT - np.asarray(alt_ft, float)) / DESCENT_GRID_STEP_FT
    if idx.ndim == 0:  # one altitude for every candidate: two plain columns
        i0 = int(np.clip(np.floor(idx), 0, values.shape[1] - 2))
        return values[:, i0] + (idx - i0) * (values[:, i0 + 1] - values[:, i0])
    i0 = np.clip(np.floor(idx).astype(int), 0, values.shape[1] - 2)
    frac = idx - i0
    lo = np.take_along_axis(values, i0[:, None], axis=1)[:, 0]
    hi = np.take_along_axis(values, i0[:, None] + 1, axis=1)[:, 0]
    return lo + frac * (hi - lo)


def _alt_where_t(t, target):
    """Per candidate, the grid altitude (ft) at which T (n_cand, K, falling
    down the grid) reaches target (n_cand,)."""
    i0 = np.clip((t > target[:, None]).sum(axis=1) - 1, 0, t.shape[1] - 2)
    lo = np.take_along_axis(t, i0[:, None], axis=1)[:, 0]
    hi = np.take_along_axis(t, i0[:, None] + 1, axis=1)[:, 0]
    frac = np.clip(np.where(lo > hi, (lo - target) / np.where(lo > hi, lo - hi, 1.0), 0.0), 0.0, 1.0)
    return DESCENT_GRID_TOP_FT - (i0 + frac) * DESCENT_GRID_STEP_FT


def _descend(g, top_ft, bottom_ft, budget_min):
    """top_ft -> bottom_ft at the normal rate, with the thrust reversers
    taking up to budget_min (n_cand,) of it from the top: REVERSER_FACTOR x
    the rate while they're in, at the same speed. Returns (nm, time_min,
    fuel_t, reverser_min); fuel is the table's per-minute burn, over the
    time actually taken."""
    t_top, t_bot = _at(g["T"], top_ft), _at(g["T"], bottom_ft)
    normal_min = np.maximum(t_top - t_bot, 0.0)
    reverser_min = np.minimum(budget_min, normal_min / REVERSER_FACTOR)
    time_min = normal_min - (REVERSER_FACTOR - 1.0) * reverser_min
    nm_top = _at(g["cum_nm"], top_ft)
    nm_reversed = 0.0
    if np.any(reverser_min > 0.0):
        nm_reversed = _at(g["cum_nm"], _alt_where_t(g["T"], t_top - REVERSER_FACTOR * reverser_min)) - nm_top
    nm = _at(g["cum_nm"], bottom_ft) - nm_top - (1.0 - 1.0 / REVERSER_FACTOR) * nm_reversed
    fuel_t = (_at(g["F"], top_ft) - _at(g["F"], bottom_ft)) * np.where(
        normal_min > 0.0, time_min / np.where(normal_min > 0.0, normal_min, 1.0), 0.0)
    return nm, time_min, fuel_t, reverser_min


def _after_waypoint(level_fl, decel_end_fl, arrival_nm, g, wind_at_fl, isa_dev_c,
                    mass_after_decel_t, reverser_budget_min, n_cand, descent_wind_at_fl=None):
    """Decel waypoint (M1.0 at decel_end_fl) -> touchdown, at one subsonic
    cruise level_fl <= decel_end_fl:

      transition  decel_end_fl -> level_fl, normal rate (none if equal)
      level       M0.95 at level_fl, for whatever distance is left over
      descent     level_fl -> DESCENT_END_FT at conc_descent.csv's own time,
                  the reversers from the top -- M0.95/the CAS limit
                  (descent_top_kias) down to SPEED_LIMIT_FL, SPEED_LIMIT_KIAS
                  below it, GATE_KIAS by the end. The table's time already
                  includes slowing down on the way; there are no level
                  slow-down segments (an earlier version added ~5 min/~29 nm
                  of them on top, against a sim-computed ~65 nm FL290 ->
                  2,000 ft with a 40 kt tailwind)
      approach    the last APPROACH_NM at GATE_KIAS

      level_nm = arrival_nm - transition_nm - descent_nm - APPROACH_NM"""
    tr_nm, tr_min, tr_fuel_t, _ = _descend(g, decel_end_fl * 100.0, level_fl * 100.0,
                                           np.zeros(n_cand))

    descent_nm, descent_min, descent_fuel_t, reverser_min = _descend(
        g, level_fl * 100.0, DESCENT_END_FT, np.full(n_cand, reverser_budget_min))
    descent_top_kias = _max_cas_kt(level_fl * 100.0, atmos.fl_to_pressure(level_fl),
                                   mass_after_decel_t - tr_fuel_t)
    if descent_wind_at_fl is None:
        descent_wind_at_fl = wind_at_fl
    low_wind, _, _ = _wind_temp(descent_wind_at_fl, DESCENT_END_FL, n_cand)
    low_temp_k = atmos.isa(DESCENT_END_FT * FT_TO_M)[0] + isa_dev_c

    gate_gs_kt = _tas_kt(GATE_KIAS, atmos.fl_to_pressure(DESCENT_END_FL), low_temp_k) + low_wind
    approach_min = np.where(gate_gs_kt > 0.0, APPROACH_NM / np.where(gate_gs_kt > 0.0, gate_gs_kt, 1.0) * 60.0,
                            np.inf)

    level_nm_raw = arrival_nm - tr_nm - descent_nm - APPROACH_NM
    level_nm_clamped = level_nm_raw < 0.0
    level_nm = np.maximum(level_nm_raw, 0.0)

    level_wind, level_temp_k, level_wind_clamped = _wind_temp(wind_at_fl, level_fl, n_cand)
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
    # the aircraft reaches the level segment at the mass at decel start less
    # the decel and transition burns (~105-120 t), not TOW (~160-185 t) and
    # not that mass untouched either -- reading it at the wrong mass here
    # silently halves the specific range and doubles level_fuel_t.
    mass_at_level_t = mass_after_decel_t - tr_fuel_t
    isa_t_k_at_level, _ = atmos.isa(level_fl * 100.0 * FT_TO_M)
    isa_dev_c_at_level = level_temp_k - isa_t_k_at_level
    fuel_total_kgh = conc_data.subsonic_cruise(level_fl, mass_at_level_t, isa_dev_c_at_level)["fuel_total_kgh"]

    # B8: level_fuel_t is flow x time, NOT ground_nm / specific_range_nm_per_t
    # -- specific_range_nm_per_t is TAS-based (still-air) nm/t, so dividing a
    # GROUND distance by it silently dropped the wind term. level_time_min
    # already carries the wind, so flow x time is wind-correct for free, and
    # mirrors its conventions exactly:
    #   level_nm == 0          -> time is already 0 -> fuel 0 too
    #   level_gs_nonpositive   -> time is already inf -> fuel inf too
    with np.errstate(invalid="ignore"):
        level_fuel_t = fuel_total_kgh / 1000.0 * level_time_min / 60.0
    level_fuel_t = np.where(level_nm == 0.0, 0.0, level_fuel_t)
    level_fuel_t = np.where(level_gs_nonpositive, np.inf, level_fuel_t)

    # Envelope flag off fuel_total_kgh: zero leftover distance burns no fuel
    # even where the table has no entry for this mass/level/ISA (B7).
    level_mass_outside_envelope = ~np.isfinite(fuel_total_kgh) & (level_nm > 0.0)

    descent_mid_fl = (level_fl + DESCENT_END_FL) / 2.0
    descent_wind, _, _ = _wind_temp(descent_wind_at_fl, descent_mid_fl, n_cand)

    out = {
        "level_fl": np.full(n_cand, float(level_fl)),
        "level_nm": level_nm,
        "level_time_min": level_time_min,
        "level_fuel_t": level_fuel_t,
        "level_wind_kt": np.asarray(level_wind, float),
        "level_gs_kt": level_gs_kt,
        "transition_nm": tr_nm,
        "transition_time_min": tr_min,
        "transition_fuel_t": tr_fuel_t,
        "descent_nm": descent_nm,
        "descent_time_min": descent_min,
        "descent_fuel_t": descent_fuel_t,
        "descent_wind_kt": np.asarray(descent_wind, float),
        "reverser_min": reverser_min,
        "descent_top_kias": np.broadcast_to(descent_top_kias, (n_cand,)).astype(float),
        "approach_time_min": approach_min,
        "level_nm_clamped": level_nm_clamped,
        "level_gs_nonpositive": level_gs_nonpositive,
        "level_mass_outside_envelope": level_mass_outside_envelope,
        "level_wind_clamped": level_wind_clamped,
    }
    out["after_waypoint_time_min"] = tr_min + level_time_min + descent_min + approach_min
    out["after_waypoint_fuel_t"] = tr_fuel_t + level_fuel_t + descent_fuel_t + APPROACH_FUEL_T
    return out


def _arrival_for_speed(cruise_fl, arrival_nm, wind_at_fl, warm, speed_kt, n_cand,
                        mass_at_barix_t, march_tail, g, isa_dev_c, reverser_budget_min,
                        decel_wind_at_fl=None, descent_wind_at_fl=None):
    """One descent speed schedule, vectorised over every candidate at once.
    The schedule only picks conc_descent.csv's decel (its distance and the
    level it ends at); after the decel waypoint every candidate flies the
    fastest of the subsonic levels at or below that (_after_waypoint)."""
    cols = ("fuel_t", "time_min", "dist_zero_wind_nm")

    # --- decel: cruise_fl -> decel_end_fl, ending at the waypoint ----------
    decel = _table_by_band(conc_data.decel_to_mach1, cruise_fl, speed_kt, warm, cols)
    # decel_end_fl is constant within a speed schedule AND identical in both
    # bands (325 -> FL383, 350 -> FL350, 380 -> FL312), so it is a true scalar.
    decel_end_fl = float(
        conc_data.decel_to_mach1(cruise_fl.flat[0], speed_kt, BAND_WARM)["decel_end_fl"]
    )

    decel_wind, _, decel_wind_clamped = _wind_temp(
        decel_wind_at_fl or wind_at_fl, (cruise_fl + decel_end_fl) / 2.0, n_cand
    )
    decel_nm = conc_data.dist_with_wind(
        decel["dist_zero_wind_nm"], decel["time_min"], decel_wind
    )

    # The decel ENDS at the waypoint, so it replaces the march's own last
    # decel_nm of supersonic cruise -- per schedule, since decel_nm is.
    mass_at_barix_t = np.broadcast_to(np.asarray(mass_at_barix_t, float), (n_cand,))
    if march_tail is None:
        replaced_s = np.zeros(n_cand)
        mass_at_decel_start_t = mass_at_barix_t
        decel_starts_before_cruise = np.zeros(n_cand, bool)
    else:
        replaced_s, mass_at_decel_start_t, decel_starts_before_cruise = march_tail(decel_nm)
    mass_after_decel_t = mass_at_decel_start_t - decel["fuel_t"]

    # --- after the waypoint: the fastest feasible subsonic level -------------
    levels = [decel_end_fl] + [fl for fl in SUBSONIC_LEVELS_FL if fl < decel_end_fl]
    per_level = [
        _after_waypoint(fl, decel_end_fl, arrival_nm, g, wind_at_fl, isa_dev_c,
                        mass_after_decel_t, reverser_budget_min, n_cand, descent_wind_at_fl)
        for fl in levels
    ]
    # Same rule as the schedule choice: a level with no fuel number (outside
    # the subsonic table's envelope at this mass) can't be the fastest.
    times = np.stack([p["after_waypoint_time_min"] for p in per_level], axis=-1)
    fuels = np.stack([p["after_waypoint_fuel_t"] for p in per_level], axis=-1)
    best = np.argmin(np.where(np.isfinite(fuels), times, np.inf), axis=-1)
    out = {key: np.choose(best, [p[key] for p in per_level]) for key in per_level[0]}

    # The decel midpoint and the level segment itself are the reads B6 was
    # written to extend coverage for. The descent's own reads clamp at the
    # BOTTOM on every candidate (below the stitched profile's ~FL183 floor --
    # see search._build_arrival_wind_fn); OR-ing those in would saturate
    # wind_fl_clamped into meaningless, so they're kept in their own key.
    wind_fl_clamped = decel_wind_clamped | out.pop("level_wind_clamped")

    out.update({
        "decel_nm": decel_nm,
        "decel_end_fl": np.full(n_cand, decel_end_fl),
        "decel_wind_kt": np.broadcast_to(decel_wind, (n_cand,)).astype(float),
        "decel_time_min": decel["time_min"],
        "decel_fuel_t": decel["fuel_t"],
        "wind_fl_clamped": wind_fl_clamped,
        "descent_wind_clamped": g["clamped"],
        "replaced_cruise_time_min": replaced_s / 60.0,
        "replaced_cruise_fuel_t": mass_at_decel_start_t - mass_at_barix_t,
        "mass_at_decel_start_t": np.asarray(mass_at_decel_start_t, float),
        "decel_starts_before_cruise": np.asarray(decel_starts_before_cruise, bool),
    })
    out["time_min"] = (
        out["decel_time_min"] + out["after_waypoint_time_min"]
        - out["replaced_cruise_time_min"]
    )
    out["fuel_t"] = (
        out["decel_fuel_t"] + out.pop("after_waypoint_fuel_t")
        - out["replaced_cruise_fuel_t"]
    )
    return out


def arrival(cruise_fl, arrival_nm, wind_at_fl, isa_dev_at_cruise,
            mass_at_barix_t=DEFAULT_MASS_AT_BARIX_T, speed=FASTEST_SCHEDULE_KT,
            march_tail=None, thrust_reversers=False, decel_wind_at_fl=None,
            descent_wind_at_fl=None):
    """Arrival time and fuel, net of the cruise the deceleration replaces:
    measured from the march's own arrival at the decel waypoint to
    touchdown (see the module docstring).

    Vectorised across candidates throughout: every argument is a (n_cand,)
    array (scalars broadcast) and every returned array is (n_cand,).

    Args:
        cruise_fl: cruise flight level. Clamped to 470-600, the decel table's
            span, and flagged where clamping bit.
        arrival_nm: ground distance from the decel waypoint (where the
            aircraft must be subsonic by) to touchdown. Comes
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
        decel_wind_at_fl, descent_wind_at_fl: optional separate sources,
            same shape as wind_at_fl, for the decel (read where it is flown,
            ending at the decel waypoint) and for the descent and approach
            (read at the destination). None means wind_at_fl, as pre-flight
            -- concopt inflight passes Active Sky at the decel waypoint and
            at EGLL; wind_at_fl then only prices the level segment.
        isa_dev_at_cruise: ISA deviation in the arrival area at cruise level,
            °C. Picks the descent temperature band (two bands, not the climb
            table's three).
        mass_at_barix_t: aircraft mass (t) the march arrived at the decel
            waypoint with -- search.py's march calls this weight_at_barix.
            With march_tail, the mass at decel START comes from march_tail
            instead and this only prices the replaced cruise fuel.
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
        march_tail: callable(back_nm (n_cand,)) -> (time_s, mass_t,
            beyond_cruise), the march's own time and mass back_nm before
            the decel waypoint (fuel.march_tail). Each schedule's decel_nm
            is backed out of the march with it. None means nothing is
            backed out -- the decel is simply added on, at
            mass_at_barix_t -- for tests that don't model a march.

    Returns:
        dict with (n_cand,) arrays:
            time_min, fuel_t (both NET of the replaced cruise),
            schedule_kt, level_fl, level_nm, decel_nm, descent_nm,
            decel_time_min, level_time_min, descent_time_min,
            decel_fuel_t, level_fuel_t, descent_fuel_t,
            replaced_cruise_time_min, replaced_cruise_fuel_t (the march's
            own time/fuel over the decel_nm the decel replaces),
            mass_at_decel_start_t, after_waypoint_time_min (level +
            descent + approach: decel waypoint -> touchdown),
            flags (str, "" when clean; names joined by ";")
        plus per-flag booleans level_nm_clamped / cruise_fl_clamped /
        level_gs_nonpositive / level_mass_outside_envelope /
        wind_fl_clamped / descent_wind_clamped, and `by_schedule`:
        {325: {...}, 350: {...}, 380: {...}}, each the full breakdown for
        that forced schedule.

    Flags rather than exceptions:
        level_nm_clamped      descent did not fit the available distance;
                              level_nm clamped to 0
        decel_starts_before_cruise
                              decel_nm is longer than the whole cruise
                              (top of climb -> decel waypoint); the decel is
                              backed out of the cruise only, not the climb
        cruise_fl_clamped     cruise_fl outside the decel table's 470-600
        level_gs_nonpositive  headwind >= M0.95 TAS; level_time_min AND
                              level_fuel_t (so fuel_t) are inf (B8 -- fuel
                              used to stay finite here even though time
                              didn't, since the old formula never divided
                              by ground speed at all; see mass docstring
                              above). A trip fuel of inf reaches fuel.py's
                              fixed point exactly like the NaN case below --
                              np.clip resolves it to MTOW_T same as inf
                              always clips, and it is flagged
                              tow_above_mtow_185 there, not silently
                              accepted as a real number.
        level_mass_outside_envelope
                              mass_at_level_t (mass_at_barix_t minus the
                              decel burn) fell outside conc_subsonic_cruise
                              .csv's published envelope at this level/ISA
                              AND the level segment actually had distance to
                              fly (level_nm > 0 -- B7 found this flag firing
                              on a zero-distance leftover that burns no fuel
                              regardless of the table); level_fuel_t (and so
                              fuel_t) is NaN. Should never fire at real
                              arrival masses.
        wind_fl_clamped       the decel midpoint or the level segment read
                              wind/temp outside wind_at_fl's own source span
                              and got a clamped value instead of a genuine
                              one (B6). A REAL number still comes back --
                              this flag is the only thing that says so isn't
                              a genuine reading; see search.
                              _build_arrival_wind_fn's stitched FL183-605
                              profile. The descent segment's own midpoint
                              clamps at the bottom on nearly every candidate
                              (down to FL163.5 on the 380 kt schedule) and
                              is deliberately NOT included here (B8) -- see
                              descent_wind_clamped.
        descent_wind_clamped  the descent segment's own midpoint clamped
                              below the stitched profile's ~FL183 floor.
                              Kept separate from wind_fl_clamped (B8) since
                              it fires on nearly every candidate and would
                              otherwise saturate that flag into meaning
                              "which schedule was picked" rather than
                              "rests on a clamped read"; accepted as
                              immaterial (~60 nm, well under 0.3 min -- see
                              search._build_arrival_wind_fn). NOT joined
                              into the `flags` string that reaches the
                              search CSV -- inspect this key directly.
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

    # The descent below the waypoint is the same airspace for every schedule
    # and level -- built once. The CAS limit it reads is weight-dependent;
    # mass_at_barix_t stands in for the (slightly lighter) descent weight.
    g = _descent_grid(descent_wind_at_fl or wind_at_fl, warm, n_cand)
    reverser_budget_min = REVERSER_MAX_MIN if thrust_reversers else 0.0

    by_schedule = {
        spd: _arrival_for_speed(
            cruise_fl, arrival_nm, wind_at_fl, warm, spd, n_cand, mass_at_barix_t, march_tail,
            g, isa_dev_at_cruise, reverser_budget_min, decel_wind_at_fl, descent_wind_at_fl,
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
                "wind_fl_clamped", "descent_wind_clamped", "decel_starts_before_cruise"):
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
