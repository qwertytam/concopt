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


_desc_fp = files("concopt").joinpath("data/conc_desc_time.csv")
_desc_tbl = pd.read_csv(_desc_fp, encoding="utf-8-sig")
# Table is FL600 (60,000 ft) down to 3,000 ft; reversed to ascending altitude
# for np.interp, which needs increasing x.
_DESC_ALT_FT = _desc_tbl["altitude"].to_numpy(float)[::-1]
_DESC_MIN = _desc_tbl["mins"].to_numpy(float)[::-1]


def desc_time_min(alt_ft):
    """Descent time (minutes) from altitude (ft) to landing, linear
    interpolation on conc_desc_time.csv (60,000 ft -> 17.1 min). Clamped to
    the table's bounds (3,000-60,000 ft); broadcasts over alt_ft."""
    alt = np.clip(np.asarray(alt_ft, float), _DESC_ALT_FT[0], _DESC_ALT_FT[-1])
    return np.interp(alt, _DESC_ALT_FT, _DESC_MIN)
