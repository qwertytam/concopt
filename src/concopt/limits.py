"""Speed/altitude limits and level-selection for the Concorde envelope.
Vectorised numpy, SI units, no pint, no classes.
"""
from functools import lru_cache

import numpy as np

from concopt.atmos import KT_TO_MS, isa, mach_from_cas, mach_from_total_temp, speed_of_sound
from concopt.data.conc_data import MMO, TOTAL_TEMP_MAX_C, cas_limit_kt

TOTAL_TEMP_MAX_K = TOTAL_TEMP_MAX_C + 273.15

# CAS-limit Mach depends only on altitude and weight, so it is precomputed
# once on this grid and interpolated in the hot loop instead of root-finding
# per candidate.
_FL_GRID = np.arange(280.0, 601.0, 5.0)  # FL280..FL600, 500 ft steps
_ALT_GRID_FT = _FL_GRID * 100.0
_ALT_GRID_M = _ALT_GRID_FT * 0.3048


@lru_cache(maxsize=None)
def mach_cas_limit_table(weight_t):
    """Cached (fl_grid, mach_grid) of CAS-limit Mach vs flight level for
    weight_t (t). Built once per weight_t via brentq; interpolate with
    np.interp for everything after."""
    cas_kt = cas_limit_kt(_ALT_GRID_FT, weight_t)
    cas_ms = cas_kt * KT_TO_MS
    _, p_Pa = isa(_ALT_GRID_M)
    mach_grid = np.array([
        mach_from_cas(cas, p) for cas, p in zip(cas_ms, p_Pa)
    ])
    return _FL_GRID, mach_grid


def max_mach(fl, T_K, weight_t=135):
    """Elementwise min of Mmo, the interpolated CAS-limit Mach, and the
    total-temperature-limit Mach."""
    fl = np.asarray(fl, dtype=float)
    T_K = np.asarray(T_K, dtype=float)

    fl_grid, mach_grid = mach_cas_limit_table(weight_t)
    cas_mach = np.interp(fl, fl_grid, mach_grid)
    tt_mach = mach_from_total_temp(T_K, TOTAL_TEMP_MAX_K)

    return np.minimum(np.minimum(MMO, cas_mach), tt_mach)


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
    return along + np.sqrt(tas_ms ** 2 - cross ** 2)


def best_level(fls, T_K, u_ms, v_ms, track_deg, weight_t):
    """Ground speed at every level (last axis), masking levels above
    ceiling_ft(weight_t). Returns (best_fl, best_gs_ms, gs_per_level)."""
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
    gs_masked = np.where(fls * 100.0 > ceiling, -np.inf, gs_per_level)

    best_idx = np.argmax(gs_masked, axis=-1, keepdims=True)
    best_fl = np.take_along_axis(fls, best_idx, axis=-1).squeeze(-1)
    best_gs_ms = np.take_along_axis(gs_masked, best_idx, axis=-1).squeeze(-1)
    return best_fl, best_gs_ms, gs_per_level
