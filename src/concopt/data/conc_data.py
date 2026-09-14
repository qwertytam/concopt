from importlib.resources import files
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

MMO = 2.04                  # max operating Mach, all altitudes
TOTAL_TEMP_MAX_C = 127.0    # max stagnation temperature, all altitudes

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
_DESCENT_LEVEL_FL = np.sort(_descent_tbl["level_fl"].unique())


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
