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
