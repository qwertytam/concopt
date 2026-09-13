"""Fuel-driven take-off weight: TOW from ZFW by fixed-point iteration.

Trip fuel depends on weight -- a heavier aircraft climbs longer, burns more
getting to top of climb, and cruises heavier -- so TOW cannot be read off ZFW
directly:

    uplift -> TOW = ZFW + uplift -> fly the model -> trip fuel
       ^                                                 |
       +---------- uplift = trip fuel + reserve ---------+

The marginal fuel-to-carry-fuel is well under 1, so the loop is contractive
and settles in a handful of passes. Damped at 0.5 for safety.

Vectorised across candidates: each pass runs the WHOLE existing march
(search.march_legs) for every candidate at once against the current TOW
*vector*, gets a trip-fuel vector back, and updates the whole TOW vector.
Never a per-candidate loop -- that would be five orders of magnitude worse
over the ~31,000 candidate departures search.py scans.

The reserve is fuel remaining AT TOUCHDOWN, not a percentage and not a
landing-weight limit, so landing weight is ZFW + reserve by construction.
"""

import numpy as np

from concopt.data.conc_data import CLIMB_TOW_MAX_T, CLIMB_TOW_MIN_T

# PLACEHOLDER -- Phase B replaces this. Descent burn from the decel point to
# touchdown, in tonnes. data/conc_descent.csv tabulates the real per-level
# decel+descent fuel/time/distance but nothing imports it yet; until it does,
# every trip-fuel figure carries this flat allowance, and anything that
# displays a fuel plan must say so rather than passing it off as computed.
DESCENT_FUEL_T = 2.0

# Structural max take-off weight. Coincides with the top of conc_climb.csv's
# TOW axis (CLIMB_TOW_MAX_T), but it is a different kind of limit: above this
# the candidate is infeasible, not merely off the end of a table.
MTOW_T = 185.0

# Fuel remaining at touchdown, tonnes. --min-landing-fuel on the CLI.
MIN_LANDING_FUEL_T = 10.0

DEFAULT_DAMPING = 0.5
DEFAULT_TOLERANCE_T = 0.05
DEFAULT_MAX_ITERATIONS = 30

# Where the loop starts before it has seen any trip fuel. Only the iteration
# count depends on this -- a contractive map lands in the same place from any
# start inside the table -- so it is deliberately a plain constant rather than
# a tuned guess.
INITIAL_TOW_T = 150.0


def trip_fuel_split(climb, legs_out):
    """(climb_fuel_t, cruise_fuel_t, descent_fuel_t), each (n_cand,) tonnes.

    The split, not just the total, because it is what makes it visible when
    the climb is eating the flight -- at 185 t on a warm day the climb alone
    is 35.7 t of a ~65 t trip.

    climb is search._climb_profile's dict (fuel_used_kg is brake release to
    top of climb, straight off conc_climb.csv at that TOW and band; mass_t is
    the top-of-climb mass). legs_out is search.march_legs' dict, whose
    weight_at_barix is the weight after the last cruise leg's burn -- so the
    cruise burn is just the drop between those two. Descent is the flat
    DESCENT_FUEL_T placeholder."""
    climb_fuel_t = np.asarray(climb["fuel_used_kg"], dtype=float) / 1000.0
    cruise_fuel_t = (np.asarray(climb["mass_t"], dtype=float)
                     - np.asarray(legs_out["weight_at_barix"], dtype=float))
    descent_fuel_t = np.full_like(climb_fuel_t, DESCENT_FUEL_T)
    return climb_fuel_t, cruise_fuel_t, descent_fuel_t


def calculate_trip_fuel(climb, legs_out):
    """Total trip fuel (n_cand,) in tonnes -- climb + cruise + descent, the
    sum of trip_fuel_split."""
    return sum(trip_fuel_split(climb, legs_out))


def fuel_plan(climb, legs_out, zfw_t=None, min_landing_fuel_t=MIN_LANDING_FUEL_T,
              tow_t=None, n_iterations=None, flags=None):
    """The whole fuel plan for one march, as a dict of (n_cand,) arrays --
    what `concopt report` prints and what the fixed point is solving for.

    Keys: climb_fuel_t, cruise_fuel_t, descent_fuel_t, trip_fuel_t, uplift_t,
    zfw_t, tow_t, tow_required_t, landing_weight_t, min_landing_fuel_t,
    n_iterations, flags.

    Two ways in, differing only in which of ZFW/TOW is the given:

    - zfw_t given (the fixed point's own output): landing weight is
      ZFW + reserve by construction, and TOW is ZFW + uplift -- exactly,
      except where the fixed point hit a bound. Pass its converged tow_t
      through as well and tow_t is the weight actually flown while
      tow_required_t is the ZFW + uplift the plan really wanted; the two
      part company precisely when a boundary flag fired.
    - zfw_t None (a plain --tow run, no fixed point): TOW is the given, so
      ZFW is what falls out the other end, ZFW = TOW - uplift. Nothing was
      iterated, so n_iterations stays None."""
    climb_fuel_t, cruise_fuel_t, descent_fuel_t = trip_fuel_split(climb, legs_out)
    trip_fuel_t = climb_fuel_t + cruise_fuel_t + descent_fuel_t
    uplift_t = trip_fuel_t + min_landing_fuel_t

    def _as_vector(value):
        return np.broadcast_to(np.asarray(value, dtype=float),
                               climb_fuel_t.shape).copy()

    if zfw_t is None:
        if tow_t is None:
            raise ValueError("fuel_plan needs one of zfw_t or tow_t")
        tow_out = _as_vector(tow_t)
        zfw_out = tow_out - uplift_t
        tow_required_t = tow_out
    else:
        zfw_out = _as_vector(zfw_t)
        tow_required_t = zfw_out + uplift_t
        tow_out = tow_required_t if tow_t is None else _as_vector(tow_t)

    if flags is None:
        flags = np.array([""] * len(climb_fuel_t), dtype=object)

    return dict(
        climb_fuel_t=climb_fuel_t, cruise_fuel_t=cruise_fuel_t,
        descent_fuel_t=descent_fuel_t, trip_fuel_t=trip_fuel_t,
        uplift_t=uplift_t, zfw_t=zfw_out, tow_t=tow_out,
        tow_required_t=tow_required_t,
        landing_weight_t=zfw_out + min_landing_fuel_t,
        min_landing_fuel_t=min_landing_fuel_t,
        n_iterations=n_iterations, flags=flags,
    )


def _boundary_flags(tow_calc):
    """(n_cand,) flag strings for the FINAL iterate's unclamped TOW.

    Evaluated once, on the converged value, never accumulated across
    iterations: the loop starts every candidate at the same INITIAL_TOW_T and
    walks towards its own answer, so a candidate can pass through (say)
    186 t on the way down to 170 t. A flag set on that transit would be a
    plain false positive."""
    flags = np.array([""] * len(tow_calc), dtype=object)
    flags[tow_calc > MTOW_T] = f"tow_above_mtow_{MTOW_T:.0f}"
    flags[tow_calc < CLIMB_TOW_MIN_T] = f"tow_below_climb_table_{CLIMB_TOW_MIN_T:.0f}"
    return flags


def fixed_point_fuel_iteration(
    cc_legs, cc_idx, data, dep_i8, zfw_t,
    min_landing_fuel_t=MIN_LANDING_FUEL_T, march_legs_fn=None,
    cruise_mach=None, damping=DEFAULT_DAMPING,
    tolerance_t=DEFAULT_TOLERANCE_T, max_iterations=DEFAULT_MAX_ITERATIONS,
):
    """Solve TOW = ZFW + trip_fuel(TOW) + reserve, for every candidate at once.

    cc_legs/cc_idx/data/dep_i8 are march_legs' own arguments (the climb+cruise
    span and the candidate departure timestamps); zfw_t is (n_cand,) tonnes.
    march_legs_fn defaults to search.march_legs and exists to be substituted
    in tests. cruise_mach None means limits.CRUISE_MACH.

    Each pass marches every candidate at the current TOW vector, reads the
    trip fuel back out, and steps

        TOW <- damping * (ZFW + trip + reserve) + (1 - damping) * TOW

    Convergence is on the UNDAMPED residual |ZFW + uplift - TOW|, which is
    what "the fixed point is solved" actually means; the damped step is
    smaller than that residual by exactly `damping` and would declare victory
    early. On convergence the returned TOW is the residual's own right-hand
    side, so TOW == ZFW + uplift holds exactly for the march that is returned
    alongside it (that march having been flown within tolerance_t of it).

    Returns (tow_t, n_iterations, flags, legs_out, weight_per_leg, climb),
    the last three being the final march's own output at the converged
    weight. flags is (n_cand,) of "" / tow_above_mtow_185 /
    tow_below_climb_table_130 / fuel_not_converged."""
    if march_legs_fn is None:
        from concopt.search import march_legs as march_legs_fn
    if cruise_mach is None:
        from concopt.limits import CRUISE_MACH as cruise_mach

    n_cand = len(dep_i8)
    zfw_t = np.asarray(zfw_t, dtype=float)
    if zfw_t.shape == ():
        zfw_t = np.full(n_cand, float(zfw_t))
    if zfw_t.shape != (n_cand,):
        raise ValueError(f"zfw_t shape {zfw_t.shape} does not match n_cand {n_cand}")

    tow_t = np.full(n_cand, INITIAL_TOW_T, dtype=float)

    for iteration in range(1, max_iterations + 1):
        legs_out, weight_per_leg, climb = march_legs_fn(
            cc_legs, cc_idx, data, dep_i8, tow_t=tow_t, cruise_mach=cruise_mach
        )
        tow_calc = zfw_t + calculate_trip_fuel(climb, legs_out) + min_landing_fuel_t

        # Clamped, not extrapolated: conc_climb.csv has no pages outside
        # [CLIMB_TOW_MIN_T, MTOW_T] and above MTOW the aircraft can't go
        # anyway. _boundary_flags records which candidates hit a bound, so a
        # clamp is never silent -- and clamping is also what lets those
        # candidates converge at all (a pinned value has zero residual).
        tow_next = np.clip(tow_calc, CLIMB_TOW_MIN_T, MTOW_T)

        if np.abs(tow_next - tow_t).max() < tolerance_t:
            return (tow_next, iteration, _boundary_flags(tow_calc),
                    legs_out, weight_per_leg, climb)

        tow_t = damping * tow_next + (1.0 - damping) * tow_t

    flags = _boundary_flags(tow_calc)
    flags[flags == ""] = "fuel_not_converged"
    # tow_next, not tow_t: tow_t was already overwritten by the damped blend
    # at the bottom of the loop above, computed from the SAME tow_calc/
    # tow_next as this flag check but never itself flown -- returning it
    # here would hand back a weight that disagrees with the returned
    # legs_out/climb (which were marched at the tow_t from BEFORE that
    # overwrite) by (1 - damping) * residual. tow_next is what the returned
    # march's own trip fuel actually implies, matching the converged
    # branch's return above.
    return tow_next, max_iterations, flags, legs_out, weight_per_leg, climb
