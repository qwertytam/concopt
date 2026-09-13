"""Acceptance test for the fuel plan block `concopt report` prints above its
waypoint table (A3 feature) -- an end-to-end run through run_report with
--zfw, on a synthetic still-air .npz (same construction as test_verify's
_still_air_data), asserting the printed block's shape rather than pinning
specific fuel numbers to the synthetic atmosphere.
"""
import datetime as dt

import numpy as np

from concopt.atmos import isa, pressure_to_fl
from concopt.route import build_legs, parse_pln
from concopt.search import DEFAULT_TOW_T
from concopt.report import run_report
from tests.test_route import SAMPLE_PLN


def _still_air_npz(tmp_path):
    """A minimal era5-shaped .npz, ISA+0 and zero wind everywhere -- enough
    for march_legs (and the climb+fuel fixed point on top of it) to run
    end to end without a real ERA5 download."""
    plan = parse_pln(SAMPLE_PLN)
    legs = build_legs(plan["waypoints"])

    levels_hpa = np.array([150.0, 125.0, 100.0, 70.0])
    fl_at_level = pressure_to_fl(levels_hpa * 100.0)
    temp_at_level, _ = isa(fl_at_level * 30.48)

    n_time = 2
    n_legs_total = len(legs)
    times = np.array(["2016-01-01T00:00:00", "2026-12-31T00:00:00"], dtype="datetime64[ns]")
    u = np.zeros((n_time, 4, n_legs_total))
    v = np.zeros((n_time, 4, n_legs_total))
    t = np.broadcast_to(temp_at_level[None, :, None], (n_time, 4, n_legs_total)).copy()

    npz_path = tmp_path / "route_legs.npz"
    np.savez(npz_path, time=times, level=levels_hpa, u=u, v=v, t=t,
              cum_nm=np.array([leg.cum_nm for leg in legs]),
              track_deg=np.array([leg.track_deg for leg in legs]))
    return npz_path


def test_run_report_with_zfw_prints_fuel_plan_block(tmp_path, capsys):
    npz_path = _still_air_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, zfw_t=92.0, min_landing_fuel_t=10.0)

    printed = capsys.readouterr().out
    fuel_block = printed.split("\n\n")[0]

    assert fuel_block.startswith("Fuel plan (ZFW 92.0 t, reserve 10.0 t at touchdown)")
    assert "uplift" in fuel_block
    assert "trip fuel" in fuel_block
    assert "climb" in fuel_block and "cruise" in fuel_block and "descent" in fuel_block
    assert "take-off weight" in fuel_block
    assert "landing weight" in fuel_block
    assert "converged in" in fuel_block
    assert "descent fuel is a placeholder" in fuel_block
    # No boundary flag expected for a plain ZFW 92 t day.
    assert "!!" not in fuel_block


def test_run_report_tow_override_skips_fixed_point(tmp_path, capsys):
    """--tow still works as a what-if override: no fixed point, and the
    printed block says so instead of an iteration count."""
    npz_path = _still_air_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, tow_t=DEFAULT_TOW_T)

    printed = capsys.readouterr().out
    fuel_block = printed.split("\n\n")[0]

    assert f"take-off weight {DEFAULT_TOW_T:.1f} t" in fuel_block
    assert "no fixed point" in fuel_block
    assert "converged in" not in fuel_block


def test_run_report_flags_boundary_clamp(tmp_path, capsys):
    """A ZFW low enough that the fixed point clamps against the climb
    table's floor must say so in the fuel plan block, not just the search
    CSV's flags column."""
    npz_path = _still_air_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, zfw_t=50.0, min_landing_fuel_t=10.0)

    printed = capsys.readouterr().out
    fuel_block = printed.split("\n\n")[0]

    assert "!!" in fuel_block
    assert "tow_below_climb_table" in fuel_block
