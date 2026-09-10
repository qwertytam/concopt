"""Speed/altitude limits and level-selection for the Concorde envelope.
Vectorised numpy, SI units, no pint, no classes.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from concopt.atmos import KT_TO_MS, isa, mach_from_cas, mach_from_total_temp, speed_of_sound
from concopt.data.conc_data import TOTAL_TEMP_MAX_C, cas_limit_kt, ceiling_ft_table

TOTAL_TEMP_MAX_K = TOTAL_TEMP_MAX_C + 273.15

# The manual's cruise is flown at M2.00, not Mmo 2.04 -- its ceiling table
# (conc_data.ceiling_ft_table) is "the altitude attainable at M2.00", so
# pairing those ceilings with Mmo would let max_mach claim speeds the
# ceiling was never validated at. MMO (2.04) stays in conc_data.py as the
# aircraft's structural limit; CRUISE_MACH is what max_mach actually uses,
# overridable from the CLI (--cruise-mach) to try both.
CRUISE_MACH = 2.00

# CAS-limit Mach depends only on altitude and weight, so it is precomputed
# once on this grid (455 brentq calls, still instant at import) and
# interpolated in the hot loop instead of root-finding per candidate. This
# has to be a fixed weight grid, not keyed on the caller's weight_t, because
# weight_t is itself an array once burn-off along the route (and later,
# live SimConnect weight) is in play.
_FL_GRID = np.arange(280.0, 601.0, 5.0)  # FL280..FL600, 500 ft steps
_ALT_GRID_FT = _FL_GRID * 100.0
_ALT_GRID_M = _ALT_GRID_FT * 0.3048
_WEIGHT_GRID = np.arange(105.0, 165.1, 10.0)  # 105..165 t, 10 t steps


def _build_mach_cas_limit_grid():
    """(len(_FL_GRID), len(_WEIGHT_GRID)) grid of CAS-limit Mach, built once
    via brentq per node."""
    alt_grid, wgt_grid = np.meshgrid(_ALT_GRID_FT, _WEIGHT_GRID, indexing="ij")
    cas_ms = cas_limit_kt(alt_grid, wgt_grid) * KT_TO_MS
    _, p_Pa = isa(_ALT_GRID_M)
    p_Pa_grid = np.broadcast_to(p_Pa[:, None], cas_ms.shape)
    mach_flat = [
        mach_from_cas(cas, p) for cas, p in zip(cas_ms.ravel(), p_Pa_grid.ravel())
    ]
    return np.array(mach_flat).reshape(cas_ms.shape)


_MACH_CAS_LIMIT_GRID = _build_mach_cas_limit_grid()
_mach_cas_interp = RegularGridInterpolator(
    (_FL_GRID, _WEIGHT_GRID), _MACH_CAS_LIMIT_GRID, bounds_error=False, fill_value=None
)


def mach_components(fl, T_K, weight_t=135, cruise_mach=CRUISE_MACH):
    """The three Mach limits max_mach takes the elementwise min of:
    cruise_mach (constant -- CRUISE_MACH by default, the manual's M2.00,
    not Mmo), the interpolated CAS-limit Mach (altitude + weight), and the
    total-temperature-limit Mach (temperature). Returns (cruise_mach,
    cas_mach, tt_mach), each broadcast to the common shape of
    fl/T_K/weight_t -- max_mach reduces these to one number,
    binding_mach_limit reports which one is smallest."""
    fl = np.asarray(fl, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    weight_t = np.asarray(weight_t, dtype=float)

    fl_b, weight_t_b = np.broadcast_arrays(fl, weight_t)
    pts = np.stack([fl_b.ravel(), weight_t_b.ravel()], axis=-1)
    cas_mach = _mach_cas_interp(pts).reshape(fl_b.shape)
    tt_mach = mach_from_total_temp(T_K, TOTAL_TEMP_MAX_K)
    cruise = np.full_like(cas_mach, cruise_mach)

    return np.broadcast_arrays(cruise, cas_mach, tt_mach)


def max_mach(fl, T_K, weight_t=135, cruise_mach=CRUISE_MACH):
    """Elementwise min of cruise_mach (CRUISE_MACH by default), the
    interpolated CAS-limit Mach, and the total-temperature-limit Mach.
    weight_t may be a scalar or an array broadcastable with fl."""
    cruise, cas_mach, tt_mach = mach_components(fl, T_K, weight_t, cruise_mach)
    return np.minimum(np.minimum(cruise, cas_mach), tt_mach)


_MACH_LIMIT_NAMES = np.array(["cruise_mach", "CAS", "total_temp"])


def binding_mach_limit(fl, T_K, weight_t=135, cruise_mach=CRUISE_MACH):
    """Which of cruise_mach/CAS/total_temp is smallest -- i.e. actually
    constrains max_mach -- at each point. String array, same shape as
    max_mach's output. Doesn't know about ceiling_ft: that's a separate,
    altitude-side constraint on which levels are even in play, not a speed
    limit at a given level; callers combine the two (see
    search.march_legs)."""
    stacked = np.stack(mach_components(fl, T_K, weight_t, cruise_mach), axis=-1)
    idx = np.argmin(stacked, axis=-1)
    return _MACH_LIMIT_NAMES[idx]


def max_tas(fl, T_K, weight_t=135, cruise_mach=CRUISE_MACH):
    """Max true airspeed (m/s) at flight level fl, static temperature T_K."""
    T_K = np.asarray(T_K, dtype=float)
    return max_mach(fl, T_K, weight_t, cruise_mach) * speed_of_sound(T_K)


def ceiling_ft(weight_t, isa_dev_c):
    """Ceiling (ft) from the Air France performance table
    (conc_data.ceiling_ft_table), bilinear over (weight_t, isa_dev_c).
    Clamped to the table's bounds, no extrapolation. Replaces the old
    weight-only placeholder, which was wrong by up to 4,200 ft and ignored
    temperature entirely -- the real spread at 165 t is 43,494 ft at
    ISA+15 to 52,269 ft at ISA-20."""
    return ceiling_ft_table(weight_t, isa_dev_c)


def ground_speed(tas_ms, track_deg, u_ms, v_ms):
    """Ground speed (m/s) holding track_deg (deg clockwise from true north)
    against wind (u_ms eastward, v_ms northward), ERA5 convention."""
    tas_ms = np.asarray(tas_ms, dtype=float)
    u_ms = np.asarray(u_ms, dtype=float)
    v_ms = np.asarray(v_ms, dtype=float)

    s = np.sin(np.radians(track_deg))
    c = np.cos(np.radians(track_deg))
    along = u_ms * s + v_ms * c
    cross = u_ms * c - v_ms * s
    radicand = tas_ms ** 2 - cross ** 2
    return np.where(radicand <= 0, -np.inf, along + np.sqrt(np.maximum(radicand, 0)))


def best_level(fls, T_K, u_ms, v_ms, track_deg, weight_t, cruise_mach=CRUISE_MACH):
    """Ground speed at every level (last axis), masking levels above
    ceiling_ft(weight_t, isa_dev_c) -- isa_dev_c is derived here from fls/
    T_K (the ceiling table is temperature-dependent; every level can have a
    different ISA deviation). Returns (best_fl, best_gs_ms, best_idx,
    gs_per_level). best_idx is the last-axis index of the winning level --
    callers that need to pull other per-level quantities (wind, ISA
    deviation, ...) at the chosen level should use it with
    np.take_along_axis rather than recovering it via gs_per_level ==
    best_gs_ms, which breaks (silently picks index 0) when best_gs_ms is
    NaN."""
    fls, T_K, u_ms, v_ms, track_deg = np.broadcast_arrays(
        np.asarray(fls, dtype=float),
        np.asarray(T_K, dtype=float),
        np.asarray(u_ms, dtype=float),
        np.asarray(v_ms, dtype=float),
        np.asarray(track_deg, dtype=float),
    )

    tas_ms = max_tas(fls, T_K, weight_t, cruise_mach)
    gs_per_level = ground_speed(tas_ms, track_deg, u_ms, v_ms)

    isa_t_k, _ = isa(fls * 100.0 * 0.3048)
    isa_dev_c = T_K - isa_t_k
    ceiling = ceiling_ft(weight_t, isa_dev_c)
    above_ceiling = fls * 100.0 > ceiling
    gs_masked = np.where(above_ceiling, -np.inf, gs_per_level)

    best_idx = np.argmax(gs_masked, axis=-1, keepdims=True)
    best_fl = np.take_along_axis(fls, best_idx, axis=-1).squeeze(-1)
    best_gs_ms = np.take_along_axis(gs_masked, best_idx, axis=-1).squeeze(-1)
    best_idx = best_idx.squeeze(-1)

    # Every level masked (all above ceiling): -inf is not a real answer.
    all_above_ceiling = np.all(above_ceiling, axis=-1)
    best_fl = np.where(all_above_ceiling, np.nan, best_fl)
    best_gs_ms = np.where(all_above_ceiling, np.nan, best_gs_ms)
    return best_fl, best_gs_ms, best_idx, gs_per_level
