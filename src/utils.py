from constants import T_ice
from constants import mins_per_deg
from constants import kts_per_ms

def convert_dms_to_dd(tude):
    multiplier = 1 if tude[-1] in ['N', 'E'] else -1
    dd = sum(
        float(x) / mins_per_deg ** n for n, x in enumerate(
            tude[:-1].split('-')))
    return multiplier * dd

def convert_degc_to_degK(degC):
    return degC + T_ice

def convert_hpa_to_pa(hpa):
    return hpa * 100

def convert_kts_to_ms(kts):
    return kts * kts_per_ms