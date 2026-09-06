"""Phase 1 acceptance checks for concopt.atmos / concopt.limits.
Prints PASS/FAIL per item. No test framework.
"""
import time

import numpy as np

from concopt.atmos import (
    A0,
    KT_TO_MS,
    P0,
    fl_to_pressure,
    isa,
    mach_from_cas,
    pressure_to_fl,
    qc_over_p,
)
from concopt.limits import max_tas

results = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append(ok)
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


# --- 1. cas_formula.md Example 2, h=11 km -----------------------------------
h = 11000.0
_, p = isa(h)
check("isa(11000) pressure ~= 22632 Pa", abs(p - 22632.0) / 22632.0 < 1e-3,
      f"p={p:.1f}")

for cas_kt, exp_ratio, exp_M in [(200.0, 0.2931, 0.6173), (400.0, 1.2546, 1.1458)]:
    cas_ms = cas_kt * KT_TO_MS
    qc = qc_over_p(cas_ms / A0) * P0
    ratio = qc / p
    M = mach_from_cas(cas_ms, p)
    ok = (abs(ratio - exp_ratio) / exp_ratio < 1e-3
          and abs(M - exp_M) / exp_M < 1e-3)
    check(f"Example 2: CAS {cas_kt:.0f} kt -> qc/p={ratio:.4f}, M={M:.4f}", ok,
          f"expected qc/p={exp_ratio}, M={exp_M}")

# --- 2. cas_formula.md Example 1, impact pressures at sea level -------------
expected_qc = {200.0: 6634.0, 400.0: 28394.0, 600.0: 71367.0, 800.0: 145402.0,
               1000.0: 249050.0}
for cas_kt, exp_qc in expected_qc.items():
    cas_ms = cas_kt * KT_TO_MS
    qc = qc_over_p(cas_ms / A0) * P0
    ok = abs(qc - exp_qc) / exp_qc < 1e-4
    check(f"Example 1: CAS {cas_kt:.0f} kt -> qc={qc:.1f} N/m2", ok,
          f"expected {exp_qc}")

# --- 3. ISA + limits table, max_tas at weight 135 t, still air --------------
expected_tas_kt = {430: 983.0, 470: 1070.5, 500: 1142.3, 510: 1167.5,
                    530: 1170.1, 600: 1170.1}
for fl, exp_kt in expected_tas_kt.items():
    T_K, _ = isa(fl * 30.48)
    tas_ms = max_tas(fl, T_K, weight_t=135)
    tas_kt = tas_ms / KT_TO_MS
    ok = abs(tas_kt - exp_kt) < 0.5
    check(f"FL{fl}: max_tas={tas_kt:.1f} kt", ok, f"expected {exp_kt} kt")

# --- 4. Optimum-temperature identity ----------------------------------------
T_star = 400.15 / (1 + 0.2 * 2.04 ** 2)
T_sweep = np.linspace(190.0, 250.0, 600001)
tas_sweep = max_tas(600, T_sweep, weight_t=135)
T_peak = T_sweep[np.argmax(tas_sweep)]
ok = abs(T_peak - T_star) < 0.5
check(f"Optimum-temperature identity: T_peak={T_peak:.2f} K", ok,
      f"expected T*={T_star:.2f} K ({T_star - 273.15:.2f} C)")

# --- 5. Round-trip pressure_to_fl(fl_to_pressure(fl)) == fl -----------------
fl_in = np.arange(0, 601, dtype=float)
fl_out = pressure_to_fl(fl_to_pressure(fl_in))
ok = np.allclose(fl_in, fl_out, atol=1e-6)
check("Round-trip pressure_to_fl(fl_to_pressure(fl)) == fl for FL0..FL600", ok,
      f"max abs error={np.max(np.abs(fl_in - fl_out)):.2e}")

# --- 6. Vectorisation -------------------------------------------------------
rng = np.random.default_rng(0)
shape = (25000, 20, 8)
fl_big = rng.uniform(280.0, 600.0, size=shape)
T_big = rng.uniform(210.0, 290.0, size=shape)
max_tas(600, np.array([250.0]), weight_t=135)  # warm the cached CAS table

t0 = time.perf_counter()
tas_big = max_tas(fl_big, T_big, weight_t=135)
elapsed = time.perf_counter() - t0

ok = elapsed < 2.0 and not np.isnan(tas_big).any()
check(f"Vectorisation: max_tas over {shape} in {elapsed:.3f}s, no NaN", ok)

# --- summary -----------------------------------------------------------------
print()
if all(results):
    print(f"ALL {len(results)} CHECKS PASSED")
else:
    print(f"{results.count(False)} of {len(results)} CHECKS FAILED")
