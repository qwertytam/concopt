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
