from importlib.resources import files
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from concopt import atmos
from concopt.params import FT_TO_M, LEVEL_MACH, MMO, TOTAL_TEMP_MAX_C  # noqa: F401 (MMO/TOTAL_TEMP_MAX_C re-exported)


_fp = files("concopt").joinpath("data/conc_cas_limit.csv")
_tbl = pd.read_csv(_fp, encoding="utf-8-sig")
_ALT_FT = _tbl["alt"].to_numpy(float)              # 0 .. 60000
_WGT_T = _tbl.columns[1:].to_numpy(float)          # 105, 135, 165
_CAS_KT = _tbl.iloc[:, 1:].to_numpy(float)
_interp = RegularGridInterpolator(
    (_ALT_FT, _WGT_T), _CAS_KT, bounds_error=False, fill_value=None
)

def cas_limit_kt(alt_ft, weight_t):
    """CAS limit (kt) for altitude(s) in ft and weight in tonnes.
    Clamped to the table's bounds; broadcasts over alt_ft."""
    alt = np.clip(np.asarray(alt_ft, float), _ALT_FT[0], _ALT_FT[-1])
    wgt = np.clip(np.asarray(weight_t, float), _WGT_T[0], _WGT_T[-1])
    alt, wgt = np.broadcast_arrays(alt, wgt)
    return _interp(np.stack([alt.ravel(), wgt.ravel()], axis=-1)).reshape(alt.shape)


_perf_fp = files("concopt").joinpath("data/conc_supersonic_cruise.csv")
_perf_tbl = pd.read_csv(_perf_fp, encoding="utf-8-sig")
# 165..100 t descending in the file; sorted ascending here so the
# RegularGridInterpolator grid axes are increasing, as scipy requires.
_perf_tbl = _perf_tbl.sort_values(["weight_t", "isa_dev_c"]).reset_index(drop=True)
_PERF_WGT_T = np.sort(_perf_tbl["weight_t"].unique())
_PERF_ISA_DEV_C = np.sort(_perf_tbl["isa_dev_c"].unique())
_CEILING_FT_GRID = (
    _perf_tbl.pivot(index="weight_t", columns="isa_dev_c", values="ceiling_ft")
    .loc[_PERF_WGT_T, _PERF_ISA_DEV_C]
    .to_numpy(float)
)
_FUEL_TOTAL_KGH_GRID = (
    _perf_tbl.pivot(index="weight_t", columns="isa_dev_c", values="fuel_total_kgh")
    .loc[_PERF_WGT_T, _PERF_ISA_DEV_C]
    .to_numpy(float)
)
_ceiling_interp = RegularGridInterpolator(
    (_PERF_WGT_T, _PERF_ISA_DEV_C), _CEILING_FT_GRID, bounds_error=False, fill_value=None
)
_fuel_total_interp = RegularGridInterpolator(
    (_PERF_WGT_T, _PERF_ISA_DEV_C), _FUEL_TOTAL_KGH_GRID, bounds_error=False, fill_value=None
)


def _clamp_perf_inputs(weight_t, isa_dev_c):
    """weight_t/isa_dev_c clamped to the performance table's grid -- no
    extrapolation, per RegularGridInterpolator's fill_value=None default
    combined with bounds_error=False would otherwise extrapolate."""
    weight_t = np.clip(np.asarray(weight_t, float), _PERF_WGT_T[0], _PERF_WGT_T[-1])
    isa_dev_c = np.clip(np.asarray(isa_dev_c, float), _PERF_ISA_DEV_C[0], _PERF_ISA_DEV_C[-1])
    return np.broadcast_arrays(weight_t, isa_dev_c)


def ceiling_ft_table(weight_t, isa_dev_c):
    """Ceiling (ft) from the AF performance table, (weight_t, isa_dev_c)
    grid. Clamped to the table's bounds; broadcasts."""
    weight_t, isa_dev_c = _clamp_perf_inputs(weight_t, isa_dev_c)
    return _ceiling_interp(
        np.stack([weight_t.ravel(), isa_dev_c.ravel()], axis=-1)
    ).reshape(weight_t.shape)


def fuel_total_kgh_table(weight_t, isa_dev_c):
    """Total fuel flow, all 4 engines (kg/h), from the AF performance
    table. Clamped to the table's bounds; broadcasts."""
    weight_t, isa_dev_c = _clamp_perf_inputs(weight_t, isa_dev_c)
    return _fuel_total_interp(
        np.stack([weight_t.ravel(), isa_dev_c.ravel()], axis=-1)
    ).reshape(weight_t.shape)


CLIMB_BANDS = ("isa_minus_20_to_minus_10", "isa_minus_10_to_isa", "isa_to_isa_plus_10")

_climb_fp = files("concopt").joinpath("data/conc_climb.csv")
_climb_tbl = pd.read_csv(_climb_fp, encoding="utf-8-sig")
_CLIMB_TOW_T = np.sort(_climb_tbl["tow_t"].unique())
_CLIMB_LEVEL_FL = np.sort(_climb_tbl["level_fl"].unique())
_CLIMB_COLS = ("mass_t", "fuel_used_kg", "dist_nm", "time_min")

# The climb table's own TOW span, read off the file rather than written out
# as a literal -- fuel.py clamps its fixed point to this, and the table has
# already been extended downwards once (160 -> 130 t) since that clamp was
# written. Anything that needs the bound should import it from here.
CLIMB_TOW_MIN_T = float(_CLIMB_TOW_T[0])
CLIMB_TOW_MAX_T = float(_CLIMB_TOW_T[-1])


def _build_climb_interps():
    """{band: {col: RegularGridInterpolator}}, one (tow_t, level_fl) grid per
    band per column -- built once at import, same pattern as the cruise
    table's _ceiling_interp/_fuel_total_interp above."""
    interps = {}
    for band in CLIMB_BANDS:
        band_tbl = (_climb_tbl[_climb_tbl["temp_band"] == band]
                    .sort_values(["tow_t", "level_fl"]))
        interps[band] = {}
        for col in _CLIMB_COLS:
            grid = (band_tbl.pivot(index="tow_t", columns="level_fl", values=col)
                    .loc[_CLIMB_TOW_T, _CLIMB_LEVEL_FL].to_numpy(float))
            interps[band][col] = RegularGridInterpolator(
                (_CLIMB_TOW_T, _CLIMB_LEVEL_FL), grid, bounds_error=False, fill_value=None
            )
    return interps


_CLIMB_INTERPS = _build_climb_interps()


def climb_to(level_fl, tow_t, temp_band):
    """(mass_t, fuel_used_kg, dist_nm, time_min) at level_fl, from
    conc_climb.csv, bilinear over (tow_t, level_fl) within temp_band (one
    of CLIMB_BANDS -- discrete bands, never interpolated between: a day's
    temperature regime picks exactly one). Clamped to the table's
    (tow_t, level_fl) bounds, no extrapolation; tow_t/level_fl broadcast
    against each other.

    dist_nm is AIR distance -- the climb tables carry no wind columns,
    unlike the descent tables (conc_descent.csv). Ground distance =
    dist_nm + wind_component_kt * time_min / 60, the same relation the
    descent tables tabulate explicitly; applying it is the caller's job
    (search.py), since the along-track wind isn't available here."""
    if temp_band not in _CLIMB_INTERPS:
        raise ValueError(f"temp_band {temp_band!r} not one of {CLIMB_BANDS}")

    level_fl = np.clip(np.asarray(level_fl, float), _CLIMB_LEVEL_FL[0], _CLIMB_LEVEL_FL[-1])
    tow_t = np.clip(np.asarray(tow_t, float), _CLIMB_TOW_T[0], _CLIMB_TOW_T[-1])
    tow_t, level_fl = np.broadcast_arrays(tow_t, level_fl)
    pts = np.stack([tow_t.ravel(), level_fl.ravel()], axis=-1)

    interps = _CLIMB_INTERPS[temp_band]
    return tuple(
        interps[col](pts).reshape(tow_t.shape) for col in _CLIMB_COLS
    )


_descent_fp = files("concopt").joinpath("data/conc_descent.csv")
_descent_tbl = pd.read_csv(_descent_fp, encoding="utf-8-sig")


def _build_descent_interps():
    """{(table, from_supersonic_cruise, speed_kt, temp_band):
        {col: RegularGridInterpolator, "decel_end_fl": scalar, "level_correction_nm_per_2000ft": scalar,
         "level_fl_min": scalar, "level_fl_max": scalar}}
    one interpolator per (level_fl,) grid per group. Each group stores its own level bounds for clamping."""
    interps = {}

    for table in _descent_tbl["table"].unique():
        for from_super in _descent_tbl["from_supersonic_cruise"].unique():
            for speed_kt in _descent_tbl["descent_speed_kt"].unique():
                for temp_band in _descent_tbl["temp_band"].unique():
                    mask = (
                        (_descent_tbl["table"] == table) &
                        (_descent_tbl["from_supersonic_cruise"] == from_super) &
                        (_descent_tbl["descent_speed_kt"] == speed_kt) &
                        (_descent_tbl["temp_band"] == temp_band)
                    )
                    if not mask.any():
                        continue

                    row_data = _descent_tbl[mask].sort_values("level_fl")
                    levels = row_data["level_fl"].to_numpy(float)

                    key = (table, from_super, speed_kt, temp_band)
                    interps[key] = {
                        "decel_end_fl": float(row_data["decel_end_fl"].iloc[0]),
                        "level_correction_nm_per_2000ft": float(row_data["level_correction_nm_per_2000ft"].iloc[0]),
                        "level_fl_min": float(levels[0]),
                        "level_fl_max": float(levels[-1]),
                    }

                    for col in ("fuel_t", "time_min", "dist_zero_wind_nm"):
                        vals = row_data[col].to_numpy(float)
                        interps[key][col] = RegularGridInterpolator(
                            (levels,), vals, bounds_error=False, fill_value=None
                        )
    return interps


_DESCENT_INTERPS = _build_descent_interps()


def decel_to_mach1(level_fl, speed_kt, temp_band):
    """(fuel_t, time_min, dist_zero_wind_nm, decel_end_fl, level_correction_nm_per_2000ft)
    from conc_descent.csv, linear interpolation over level_fl for the given
    speed_kt and temp_band (one of "above_isa_minus_10" or "isa_minus_10_and_below").
    Valid level_fl 470-600. decel_end_fl and level_correction_nm_per_2000ft are
    scalar constants within each speed schedule (not interpolated).
    Clamped to the table's level_fl bounds, no extrapolation; broadcasts over level_fl."""
    key = ("decel_to_mach1", True, float(speed_kt), temp_band)
    if key not in _DESCENT_INTERPS:
        raise ValueError(f"decel_to_mach1: no data for speed_kt={speed_kt}, temp_band={temp_band}")

    group = _DESCENT_INTERPS[key]
    level_fl = np.clip(np.asarray(level_fl, float), group["level_fl_min"], group["level_fl_max"])
    original_shape = level_fl.shape
    level_fl_flat = level_fl.ravel()

    result = {}
    for col in ("fuel_t", "time_min", "dist_zero_wind_nm"):
        result[col] = group[col](level_fl_flat).reshape(original_shape)
    result["decel_end_fl"] = group["decel_end_fl"]
    result["level_correction_nm_per_2000ft"] = group["level_correction_nm_per_2000ft"]
    return result


def descent_to_1500ft(level_fl, speed_kt, temp_band):
    """(fuel_t, time_min, dist_zero_wind_nm) from conc_descent.csv for subsonic
    descent from a level-off altitude (from_supersonic_cruise=False), linear
    interpolation over level_fl for the given speed_kt and temp_band (one of
    "above_isa_minus_10" or "isa_minus_10_and_below"). Valid level_fl 30-550.
    Clamped to the table's level_fl bounds, no extrapolation; broadcasts over level_fl."""
    key = ("descent_to_1500ft", False, float(speed_kt), temp_band)
    if key not in _DESCENT_INTERPS:
        raise ValueError(f"descent_to_1500ft: no data for speed_kt={speed_kt}, temp_band={temp_band}")

    group = _DESCENT_INTERPS[key]
    level_fl = np.clip(np.asarray(level_fl, float), group["level_fl_min"], group["level_fl_max"])
    original_shape = level_fl.shape
    level_fl_flat = level_fl.ravel()

    result = {}
    for col in ("fuel_t", "time_min", "dist_zero_wind_nm"):
        result[col] = group[col](level_fl_flat).reshape(original_shape)
    return result


def descent_direct_from_cruise(level_fl, speed_kt, temp_band):
    """(fuel_t, time_min, dist_zero_wind_nm) from conc_descent.csv for combined
    decel-AND-descent directly from cruise (from_supersonic_cruise=True), linear
    interpolation over level_fl for the given speed_kt and temp_band (one of
    "above_isa_minus_10" or "isa_minus_10_and_below"). Valid level_fl 470-600.
    Only 8 rows per group.

    WARNING: This is an ALTERNATIVE to decel_to_mach1 + descent_to_1500ft,
    never a sequential addition -- summing them double-counts the deceleration."""
    key = ("descent_to_1500ft", True, float(speed_kt), temp_band)
    if key not in _DESCENT_INTERPS:
        raise ValueError(f"descent_direct_from_cruise: no data for speed_kt={speed_kt}, temp_band={temp_band}")

    group = _DESCENT_INTERPS[key]
    level_fl = np.clip(np.asarray(level_fl, float), group["level_fl_min"], group["level_fl_max"])
    original_shape = level_fl.shape
    level_fl_flat = level_fl.ravel()

    result = {}
    for col in ("fuel_t", "time_min", "dist_zero_wind_nm"):
        result[col] = group[col](level_fl_flat).reshape(original_shape)
    return result


def dist_with_wind(dist_zero_wind_nm, time_min, wind_kt):
    """Ground distance (nm) from zero-wind distance, time, and along-track wind
    component. wind_kt is positive for tailwind. Verified against all 582
    printed endpoints to within 1 nm."""
    return dist_zero_wind_nm + wind_kt * time_min / 60.0


_subsonic_fp = files("concopt").joinpath("data/conc_subsonic_cruise.csv")
_subsonic_tbl = pd.read_csv(_subsonic_fp, encoding="utf-8-sig")
_SUBSONIC_LEVEL_FL = np.sort(_subsonic_tbl["level_fl"].unique()).astype(float)
_SUBSONIC_MASS_T = np.sort(_subsonic_tbl["mass_t"].unique()).astype(float)
_SUBSONIC_ISA_DEV_C = np.sort(_subsonic_tbl["isa_dev_c"].unique()).astype(float)


def _build_subsonic_grid(col):
    """Dense (level_fl, mass_t, isa_dev_c) grid for one conc_subsonic_cruise.csv
    column, NaN wherever the CSV has no row. The grid is genuinely ragged --
    higher levels are only published down to a lower maximum mass (FL410
    tops out at 125 t, FL290 at 180 t), plus a handful of cells missing at
    the hot/heavy corner of an otherwise-published row. Gaps are left as
    NaN, never filled -- _trilinear's job is to return NaN for a query that
    needs one, not to guess."""
    grid = np.full(
        (len(_SUBSONIC_LEVEL_FL), len(_SUBSONIC_MASS_T), len(_SUBSONIC_ISA_DEV_C)),
        np.nan,
    )
    li = {v: i for i, v in enumerate(_SUBSONIC_LEVEL_FL)}
    mi = {v: i for i, v in enumerate(_SUBSONIC_MASS_T)}
    ii = {v: i for i, v in enumerate(_SUBSONIC_ISA_DEV_C)}
    for row in _subsonic_tbl.itertuples(index=False):
        grid[li[float(row.level_fl)], mi[float(row.mass_t)], ii[float(row.isa_dev_c)]] = (
            getattr(row, col)
        )
    return grid


_SUBSONIC_FUEL_TOTAL_KGH_GRID = _build_subsonic_grid("fuel_total_kgh")
_SUBSONIC_SR_NM_PER_T_GRID = _build_subsonic_grid("specific_range_nm_per_t")


def _frac_index(x, grid):
    """(idx, frac) bracketing x into a 1-D sorted grid: x is clamped to the
    grid's own bounds first (no extrapolation), idx is the lower bracket
    index (clipped to len(grid)-2 so idx+1 is always valid), and frac is in
    [0, 1] -- exactly 0.0 or 1.0, in bit-exact float arithmetic, whenever x
    lands exactly on a grid value. _trilinear leans on that exactness: a
    corner reached with exactly zero weight never contributes, even if that
    corner itself is NaN, so an exact grid hit or an axis-bound clamp can't
    be poisoned by a ragged neighbour it doesn't actually need."""
    x = np.clip(np.asarray(x, float), grid[0], grid[-1])
    idx = np.searchsorted(grid, x, side="right") - 1
    idx = np.clip(idx, 0, len(grid) - 2)
    x0, x1 = grid[idx], grid[idx + 1]
    frac = (x - x0) / (x1 - x0)
    return idx, frac


def _trilinear(grid, level_fl, mass_t, isa_dev_c):
    """Trilinear interpolation into a dense (level_fl, mass_t, isa_dev_c)
    grid that may hold NaN gaps (see _build_subsonic_grid). Clamped to the
    grid's outer bounds on every axis; a query that needs a NaN corner with
    nonzero weight returns NaN -- outside the published envelope, per
    subsonic_cruise's contract. Broadcasts level_fl/mass_t/isa_dev_c."""
    level_fl, mass_t, isa_dev_c = np.broadcast_arrays(
        np.asarray(level_fl, float), np.asarray(mass_t, float),
        np.asarray(isa_dev_c, float),
    )
    li, lf = _frac_index(level_fl, _SUBSONIC_LEVEL_FL)
    mi, mf = _frac_index(mass_t, _SUBSONIC_MASS_T)
    ii, iff = _frac_index(isa_dev_c, _SUBSONIC_ISA_DEV_C)

    total = np.zeros(level_fl.shape)
    for dl, wl in ((0, 1.0 - lf), (1, lf)):
        for dm, wm in ((0, 1.0 - mf), (1, mf)):
            for di, wi in ((0, 1.0 - iff), (1, iff)):
                w = wl * wm * wi
                vals = grid[li + dl, mi + dm, ii + di]
                # w * vals is NaN wherever w==0 and vals is NaN too -- np.where
                # discards that NaN rather than letting a zero-weight corner
                # poison a genuine exact hit or boundary clamp.
                total = total + np.where(w == 0.0, 0.0, w * vals)
    return total


def subsonic_cruise(level_fl, mass_t, isa_dev_c):
    """dict(tas_kt, fuel_total_kgh, specific_range_nm_per_t) for the subsonic
    (M0.95) arrival cruise, from conc_subsonic_cruise.csv. Trilinear over
    (level_fl, mass_t, isa_dev_c), clamped at each axis's own bounds; NaN
    wherever the query needs a mass the aircraft cannot hold at that
    level/ISA deviation -- the table is ragged (FL410 is only published up
    to 125 t, FL290 up to 180 t, plus a few cells missing at the hot/heavy
    corner of an otherwise-published row) and gaps are never filled.
    Broadcasts level_fl/mass_t/isa_dev_c against each other.

    mach is a constant 0.95 throughout the table, so it is never
    interpolated. tas_kt is likewise not read off the (ragged, mass-
    independent) table column -- it does not actually depend on mass_t at
    all, and computing it directly from atmos.py matches the transcribed
    column to within 0.5 kt (all 27 printed values checked) without an
    envelope NaN that TAS was never subject to in the first place."""
    level_fl = np.asarray(level_fl, float)
    isa_dev_c = np.asarray(isa_dev_c, float)
    level_fl_clamped = np.clip(level_fl, _SUBSONIC_LEVEL_FL[0], _SUBSONIC_LEVEL_FL[-1])
    isa_dev_c_clamped = np.clip(isa_dev_c, _SUBSONIC_ISA_DEV_C[0], _SUBSONIC_ISA_DEV_C[-1])
    isa_t_k, _ = atmos.isa(level_fl_clamped * 100.0 * FT_TO_M)
    tas_ms = atmos.speed_of_sound(isa_t_k + isa_dev_c_clamped) * LEVEL_MACH
    tas_kt, _ = np.broadcast_arrays(tas_ms / atmos.KT_TO_MS, np.asarray(mass_t, float))

    return {
        "tas_kt": tas_kt,
        "fuel_total_kgh": _trilinear(_SUBSONIC_FUEL_TOTAL_KGH_GRID, level_fl, mass_t, isa_dev_c),
        "specific_range_nm_per_t": _trilinear(
            _SUBSONIC_SR_NM_PER_T_GRID, level_fl, mass_t, isa_dev_c
        ),
    }
