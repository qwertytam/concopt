"""Acceptance tests for concopt.limits: the ISA-derived speed limits table,
the Mmo/total-temperature crossover identity, and the vectorised-scan
performance/NaN guarantees.
"""
import time

import numpy as np
import pytest

from concopt.atmos import KT_TO_MS, isa
from concopt.limits import max_tas


@pytest.mark.parametrize(
    "fl, exp_tas_kt",
    [
        (430, 983.0),
        (470, 1070.5),
        (500, 1142.3),
        (510, 1167.5),
        (530, 1170.1),
        (600, 1170.1),
    ],
)
def test_max_tas_table(fl, exp_tas_kt):
    """ISA + limits table at weight 135 t, still air. Note the flat region at
    FL530/FL600: Mmo binds and the stratosphere is isothermal, so TAS stops
    rising."""
    T_K, _ = isa(fl * 30.48)
    tas_kt = max_tas(fl, T_K, weight_t=135) / KT_TO_MS
    assert tas_kt == pytest.approx(exp_tas_kt, abs=0.5)


def test_optimum_temperature_identity():
    """Max TAS over static temperature occurs where the Mmo and
    total-temperature limits cross. Warmer AND colder are both slower."""
    T_star = 400.15 / (1 + 0.2 * 2.04 ** 2)

    T_sweep = np.linspace(190.0, 250.0, 600_001)
    tas_sweep = max_tas(600, T_sweep, weight_t=135)
    T_peak = T_sweep[np.argmax(tas_sweep)]

    assert T_peak == pytest.approx(T_star, abs=0.5)


def test_max_tas_vectorisation_performance():
    """max_tas over a (25000, 20, 8) array completes in under 2 s and
    contains no NaN."""
    rng = np.random.default_rng(0)
    shape = (25000, 20, 8)
    fl_big = rng.uniform(280.0, 600.0, size=shape)
    T_big = rng.uniform(210.0, 290.0, size=shape)

    max_tas(600, np.array([250.0]), weight_t=135)  # warm the cached CAS table

    t0 = time.perf_counter()
    tas_big = max_tas(fl_big, T_big, weight_t=135)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0
    assert not np.isnan(tas_big).any()
