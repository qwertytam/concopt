"""ISA atmosphere, speed of sound, flight-level/pressure inversion, and CAS/
Mach conversions. Vectorised numpy, SI units, no classes.
"""
import numpy as np
from scipy.optimize import brentq

from concopt.params import (A0, G0, GAMMA, ISA_H1_M as _H1, ISA_H2_M as _H2,  # noqa: F401
                            ISA_L1_K_PER_M as _L1, ISA_L3_K_PER_M as _L3,
                            ISA_T_ISO_K as _T_ISO, KT_TO_MS, M_PER_FL as _M_PER_FL, P0, R, T0)




_P1 = P0 * (_T_ISO / T0) ** (-G0 / (R * _L1))          # pressure at 11 km
_P2 = _P1 * np.exp(-G0 * (_H2 - _H1) / (R * _T_ISO))   # pressure at 20 km



def isa(h_m):
    """ISA temperature (K) and pressure (Pa) up to 32 km, vectorised."""
    h = np.asarray(h_m, dtype=float)

    t1 = T0 + _L1 * h
    p1 = P0 * (t1 / T0) ** (-G0 / (R * _L1))

    t2 = np.full_like(h, _T_ISO)
    p2 = _P1 * np.exp(-G0 * (h - _H1) / (R * _T_ISO))

    t3 = _T_ISO + _L3 * (h - _H2)
    p3 = _P2 * (t3 / _T_ISO) ** (-G0 / (R * _L3))

    conds = [h < _H1, (h >= _H1) & (h < _H2), h >= _H2]
    T_K = np.select(conds, [t1, t2, t3])
    p_Pa = np.select(conds, [p1, p2, p3])
    return T_K, p_Pa


def speed_of_sound(T_K):
    """Local speed of sound (m/s) for static temperature T_K."""
    return np.sqrt(GAMMA * R * np.asarray(T_K, dtype=float))


def fl_to_pressure(fl):
    """Pressure (Pa) at the given flight level (FL = pressure altitude / 100 ft)."""
    h_m = np.asarray(fl, dtype=float) * _M_PER_FL
    _, p_Pa = isa(h_m)
    return p_Pa


def pressure_to_fl(p_Pa):
    """Invert the ISA pressure relation exactly to get flight level from pressure."""
    p = np.asarray(p_Pa, dtype=float)

    t1 = T0 * (p / P0) ** (-R * _L1 / G0)
    h1 = (t1 - T0) / _L1

    h2 = _H1 - (R * _T_ISO / G0) * np.log(p / _P1)

    t3 = _T_ISO * (p / _P2) ** (-R * _L3 / G0)
    h3 = _H2 + (t3 - _T_ISO) / _L3

    conds = [p >= _P1, (p < _P1) & (p >= _P2), p < _P2]
    h_m = np.select(conds, [h1, h2, h3])
    return h_m / _M_PER_FL


def qc_over_p(M):
    """Impact pressure ratio qc/p, per cas_formula.md eq. 15 (subsonic) and
    eq. 18 (supersonic)."""
    M = np.asarray(M, dtype=float)
    sub = (1 + 0.2 * M ** 2) ** 3.5 - 1
    # Guard the supersonic branch against the 7M^2-1 singularity below
    # M ~ 0.38 so np.where doesn't emit warnings evaluating the unused branch.
    M_safe = np.maximum(M, 0.4)
    sup = 166.92158 * M_safe ** 7 / (7 * M_safe ** 2 - 1) ** 2.5 - 1
    return np.where(sub <= 0.893, sub, sup)


def mach_from_cas(cas_ms, p_Pa):
    """Mach number for a scalar CAS (m/s) at ambient pressure p_Pa (Pa)."""
    qc = qc_over_p(cas_ms / A0) * P0
    target = qc / p_Pa

    def f(M):
        return qc_over_p(M) - target

    return brentq(f, 0.05, 6.0)


def cas_from_mach(M, p_Pa):
    """CAS (m/s) at Mach M and ambient pressure p_Pa (Pa), vectorised -- the
    inverse of mach_from_cas. Subsonic-CAS formula (valid below A0, ~661 kt;
    Concorde's CAS never gets near it)."""
    qc = qc_over_p(M) * np.asarray(p_Pa, dtype=float)
    return A0 * np.sqrt(2.0 / (GAMMA - 1.0) * ((qc / P0 + 1.0) ** ((GAMMA - 1.0) / GAMMA) - 1.0))


def mach_from_total_temp(T_K, total_temp_max_K):
    """Mach number at which static temperature T_K reaches total_temp_max_K
    (inverse of Tt = T*(1 + 0.2*M^2))."""
    T_K = np.asarray(T_K, dtype=float)
    return np.sqrt(5.0 * (total_temp_max_K / T_K - 1.0))
