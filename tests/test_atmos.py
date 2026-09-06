"""Acceptance tests for concopt.atmos, derived from cas_formula.md's worked
examples plus the flight-level/pressure round-trip.
"""
import numpy as np
import pytest

from concopt.atmos import A0, KT_TO_MS, P0, fl_to_pressure, isa, mach_from_cas, pressure_to_fl, qc_over_p


def test_isa_pressure_at_11km():
    """cas_formula.md Example 2: h=11 km -> p=22632 Pa."""
    _, p = isa(11000.0)
    assert p == pytest.approx(22632.0, rel=1e-3)


@pytest.mark.parametrize(
    "cas_kt, exp_ratio, exp_mach",
    [
        (200.0, 0.2931, 0.6173),
        (400.0, 1.2546, 1.1458),
    ],
)
def test_cas_formula_example_2(cas_kt, exp_ratio, exp_mach):
    """cas_formula.md Example 2, h=11 km (p=22632 Pa): CAS -> qc/p -> M."""
    _, p = isa(11000.0)
    cas_ms = cas_kt * KT_TO_MS

    qc = qc_over_p(cas_ms / A0) * P0
    ratio = qc / p
    M = mach_from_cas(cas_ms, p)

    assert ratio == pytest.approx(exp_ratio, rel=1e-3)
    assert M == pytest.approx(exp_mach, rel=1e-3)


@pytest.mark.parametrize(
    "cas_kt, exp_qc",
    [
        (200.0, 6634.0),
        (400.0, 28394.0),
        (600.0, 71367.0),
        (800.0, 145402.0),
        (1000.0, 249050.0),
    ],
)
def test_cas_formula_example_1_impact_pressure(cas_kt, exp_qc):
    """cas_formula.md Example 1: impact pressure at sea level for a range of
    CAS values."""
    cas_ms = cas_kt * KT_TO_MS
    qc = qc_over_p(cas_ms / A0) * P0
    assert qc == pytest.approx(exp_qc, rel=1e-4)


def test_pressure_fl_roundtrip():
    """pressure_to_fl(fl_to_pressure(fl)) == fl for FL0..FL600."""
    fl_in = np.arange(0, 601, dtype=float)
    fl_out = pressure_to_fl(fl_to_pressure(fl_in))
    assert np.allclose(fl_in, fl_out, atol=1e-6)
