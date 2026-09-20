"""params.py is the one home for the project's constants. These tests keep it
that way: it must stay a leaf (so any module can import it without a cycle),
its unit factors must agree with each other, and the handful of facts that
used to be repeated inline across modules must not creep back in.
"""
import ast
import pathlib

import pytest

from concopt import params

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "concopt"

# The vendored flightcondition fork -- deliberately left alone.
VENDORED = {"condition.py", "atmosphere.py", "common.py", "airframeflows.py",
            "nondimensional.py", "constants.py", "units.py", "utils.py"}


def _project_modules():
    return [p for p in SRC.rglob("*.py") if p.name not in VENDORED and p.name != "params.py"]


def test_params_is_a_leaf_module():
    """No concopt imports: replay.py once had to duplicate a constant to dodge
    an import cycle, and atmos.py's hot path must be free to import this."""
    tree = ast.parse((SRC / "params.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "concopt" not in imported and "" not in imported, imported


def test_unit_factors_agree_with_each_other():
    assert params.M_PER_FL == pytest.approx(100.0 * params.FT_TO_M)
    assert params.KT_TO_MS == pytest.approx(params.NM_TO_M / params.S_PER_HOUR)
    assert params.TOTAL_TEMP_MAX_K == pytest.approx(params.TOTAL_TEMP_MAX_C + params.C_TO_K)
    assert params.DESCENT_END_FL == pytest.approx(params.DESCENT_END_FT / 100.0)
    assert params.APPROACH_AGL_FT == params.DESCENT_END_FT
    assert params.DEFAULT_TOW_T == params.MTOW_T


def test_old_module_names_still_resolve_to_the_params_values():
    """Modules import their constants from params and keep the old names, so
    `search.DEFAULT_TOW_T`, `limits.CRUISE_MACH`, ... still work."""
    from concopt import arrival, atmos, era5, fuel, inflight, limits, runways, search, verify
    from concopt.data import conc_data

    assert search.DEFAULT_TOW_T is params.DEFAULT_TOW_T
    assert search.TARGET_FL is params.TARGET_FL
    assert limits.CRUISE_MACH is params.CRUISE_MACH
    assert fuel.MTOW_T is params.MTOW_T
    assert arrival.LEVEL_MACH is params.LEVEL_MACH
    assert atmos.KT_TO_MS is params.KT_TO_MS
    assert atmos.A0 is params.A0
    assert era5.ARCHIVE_START is params.ARCHIVE_START
    assert runways.RUNWAYS is params.RUNWAYS
    assert inflight.BRAKE_RELEASE_GS_KT is params.BRAKE_RELEASE_GS_KT
    assert verify.SNAPSHOT_CACHE_PATH is params.SNAPSHOT_CACHE_PATH
    assert conc_data.MMO is params.MMO


# Facts that used to be written inline in several modules. Each now has one
# definition in params.py; a stray literal elsewhere is a second definition.
_BANNED_LITERALS = {
    0.3048: "params.FT_TO_M",
    273.15: "params.C_TO_K",
    1852.0: "params.NM_TO_M",
    1852: "params.NM_TO_M",
    19285: "params.ACTIVE_SKY_PORT",
    86400.0: "params.S_PER_DAY",
}


@pytest.mark.parametrize("path", _project_modules(), ids=lambda p: p.name)
def test_no_stray_unit_or_port_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [
        (node.lineno, node.value, _BANNED_LITERALS[node.value])
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value in _BANNED_LITERALS
    ]
    assert not offenders, f"{path.name}: use params instead: {offenders}"
