"""Speed/altitude limits and level-selection for the Concorde envelope.
Vectorised numpy, SI units, no pint, no classes.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from concopt.atmos import KT_TO_MS, isa, mach_from_cas, mach_from_total_temp, speed_of_sound
from concopt.data.conc_data import MMO, TOTAL_TEMP_MAX_C, cas_limit_kt

TOTAL_TEMP_MAX_K = TOTAL_TEMP_MAX_C + 273.15

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


def mach_components(fl, T_K, weight_t=135):
    """The three Mach limits max_mach takes the elementwise min of: Mmo
    (constant), the interpolated CAS-limit Mach (altitude + weight), and the
    total-temperature-limit Mach (temperature). Returns (mmo, cas_mach,
    tt_mach), each broadcast to the common shape of fl/T_K/weight_t --
    max_mach reduces these to one number, binding_mach_limit reports which
    one is smallest."""
    fl = np.asarray(fl, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    weight_t = np.asarray(weight_t, dtype=float)

    fl_b, weight_t_b = np.broadcast_arrays(fl, weight_t)
    pts = np.stack([fl_b.ravel(), weight_t_b.ravel()], axis=-1)
    cas_mach = _mach_cas_interp(pts).reshape(fl_b.shape)
    tt_mach = mach_from_total_temp(T_K, TOTAL_TEMP_MAX_K)
    mmo = np.full_like(cas_mach, MMO)

    return np.broadcast_arrays(mmo, cas_mach, tt_mach)


def max_mach(fl, T_K, weight_t=135):
    """Elementwise min of Mmo, the interpolated CAS-limit Mach, and the
    total-temperature-limit Mach. weight_t may be a scalar or an array
    broadcastable with fl."""
    mmo, cas_mach, tt_mach = mach_components(fl, T_K, weight_t)
    return np.minimum(np.minimum(mmo, cas_mach), tt_mach)


_MACH_LIMIT_NAMES = np.array(["Mmo", "CAS", "total_temp"])


def binding_mach_limit(fl, T_K, weight_t=135):
    """Which of Mmo/CAS/total_temp is smallest -- i.e. actually constrains
    max_mach -- at each point. String array, same shape as max_mach's
    output. Doesn't know about ceiling_ft: that's a separate, altitude-side
    constraint on which levels are even in play, not a speed limit at a
    given level; callers combine the two (see search.march_legs)."""
    stacked = np.stack(mach_components(fl, T_K, weight_t), axis=-1)
    idx = np.argmin(stacked, axis=-1)
    return _MACH_LIMIT_NAMES[idx]


def max_tas(fl, T_K, weight_t=135):
    """Max true airspeed (m/s) at flight level fl, static temperature T_K."""
    T_K = np.asarray(T_K, dtype=float)
    return max_mach(fl, T_K, weight_t) * speed_of_sound(T_K)


def ceiling_ft(weight_t):
    """Piecewise-linear ceiling (ft) vs weight (t), placeholder values to be
    replaced from the FS Labs manual. Clamped both ends."""
    weight_t = np.asarray(weight_t, dtype=float)
    return np.interp(weight_t, [120.0, 135.0, 165.0], [60000.0, 57000.0, 50000.0])


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


def best_level(fls, T_K, u_ms, v_ms, track_deg, weight_t):
    """Ground speed at every level (last axis), masking levels above
    ceiling_ft(weight_t). Returns (best_fl, best_gs_ms, best_idx,
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

    tas_ms = max_tas(fls, T_K, weight_t)
    gs_per_level = ground_speed(tas_ms, track_deg, u_ms, v_ms)

    ceiling = ceiling_ft(weight_t)
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
