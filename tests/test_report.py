"""Acceptance test for the fuel plan block `concopt report` prints above its
waypoint table (A3 feature) -- an end-to-end run through run_report with
--zfw, on a synthetic still-air .npz (same construction as test_verify's
_still_air_data), asserting the printed block's shape rather than pinning
specific fuel numbers to the synthetic atmosphere.

B3 extends this with the Arrival breakdown block (arrival.py wired in),
--subsonic-npz end-to-end, and the arrival-fuel-inside-the-fixed-point
regression the wiring exists for.
"""
import datetime as dt

import numpy as np
import pytest

from concopt.arrival import LEGACY_FLAT_FUEL_T
from concopt.atmos import isa, pressure_to_fl
from concopt.era5 import load_legs_npz
from concopt.route import build_legs, climb_cruise_segment, parse_pln
from concopt.search import DEFAULT_TOW_T, resolve_tow_and_arrival
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


def _still_air_subsonic_npz(tmp_path):
    """era5.reduce_to_legs-shaped .npz over the post-BARIX (arrival) legs
    only, at the 7 subsonic levels (era5.SUBSONIC_LEVELS, 175-500 hPa),
    ISA+0 and zero wind everywhere -- enough for arrival.arrival()'s
    wind_at_fl to run end to end without a real ERA5 download."""
    plan = parse_pln(SAMPLE_PLN)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs)
    arrival_legs = [legs[i] for i, m in enumerate(mask) if not m]

    levels_hpa = np.array([175.0, 200.0, 225.0, 250.0, 300.0, 400.0, 500.0])
    fl_at_level = pressure_to_fl(levels_hpa * 100.0)
    temp_at_level, _ = isa(fl_at_level * 30.48)

    n_time = 2
    n_legs = len(arrival_legs)
    times = np.array(["2016-01-01T00:00:00", "2026-12-31T00:00:00"], dtype="datetime64[ns]")
    u = np.zeros((n_time, 7, n_legs))
    v = np.zeros((n_time, 7, n_legs))
    t = np.broadcast_to(temp_at_level[None, :, None], (n_time, 7, n_legs)).copy()

    npz_path = tmp_path / "route_legs_subsonic.npz"
    np.savez(npz_path, time=times, level=levels_hpa, u=u, v=v, t=t,
              cum_nm=np.array([leg.cum_nm for leg in arrival_legs]),
              track_deg=np.array([leg.track_deg for leg in arrival_legs]))
    return npz_path


def _still_air_arrival_upper_npz(tmp_path):
    """era5.reduce_to_legs-shaped .npz over the post-BARIX (arrival) legs,
    at the 4 upper-air levels (era5.UPPER_AIR_LEVELS, 70-150 hPa,
    FL447-FL605), ISA+0 and zero wind everywhere -- B6, the same netCDFs
    already reduced onto the cruise legs, reduced a second time onto the
    arrival legs to extend _still_air_subsonic_npz's FL183-FL414 coverage
    up to FL605. The time axis MUST match _still_air_subsonic_npz's own
    (both stand in for era5.UPPER_AIR_TIMES-built requests against the
    same route) -- _build_arrival_wind_fn asserts this rather than
    assuming it."""
    plan = parse_pln(SAMPLE_PLN)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs)
    arrival_legs = [legs[i] for i, m in enumerate(mask) if not m]

    levels_hpa = np.array([70.0, 100.0, 125.0, 150.0])
    fl_at_level = pressure_to_fl(levels_hpa * 100.0)
    temp_at_level, _ = isa(fl_at_level * 30.48)

    n_time = 2
    n_legs = len(arrival_legs)
    times = np.array(["2016-01-01T00:00:00", "2026-12-31T00:00:00"], dtype="datetime64[ns]")
    u = np.zeros((n_time, 4, n_legs))
    v = np.zeros((n_time, 4, n_legs))
    t = np.broadcast_to(temp_at_level[None, :, None], (n_time, 4, n_legs)).copy()

    npz_path = tmp_path / "route_legs_arrival_upper.npz"
    np.savez(npz_path, time=times, level=levels_hpa, u=u, v=v, t=t,
              cum_nm=np.array([leg.cum_nm for leg in arrival_legs]),
              track_deg=np.array([leg.track_deg for leg in arrival_legs]))
    return npz_path


def test_run_report_with_zfw_prints_fuel_plan_block(tmp_path, capsys):
    npz_path = _still_air_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, zfw_t=92.0, min_landing_fuel_t=10.0,
                decel_descent_min=35.0)

    printed = capsys.readouterr().out
    fuel_block = printed.split("\n\n")[0]

    assert fuel_block.startswith("Fuel plan (ZFW 92.0 t, reserve 10.0 t at touchdown)")
    assert "uplift" in fuel_block
    assert "trip fuel" in fuel_block
    assert "climb" in fuel_block and "cruise" in fuel_block and "arrival" in fuel_block
    assert "take-off weight" in fuel_block
    assert "landing weight" in fuel_block
    assert "converged in" in fuel_block
    assert "Arrival breakdown below" in fuel_block
    # No boundary flag expected for a plain ZFW 92 t day.
    assert "!!" not in fuel_block


def test_run_report_prints_arrival_block(tmp_path, capsys):
    """The Arrival (BARIX -> touchdown) block: per-segment nm/min/t plus a
    total row, printed as its own block right after the fuel plan."""
    npz_path = _still_air_npz(tmp_path)
    subsonic_npz_path = _still_air_subsonic_npz(tmp_path)
    arrival_upper_npz_path = _still_air_arrival_upper_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, zfw_t=92.0, min_landing_fuel_t=10.0,
                subsonic_npz_path=subsonic_npz_path,
                arrival_upper_npz_path=arrival_upper_npz_path)

    printed = capsys.readouterr().out
    arrival_block = printed.split("\n\n")[1]

    assert arrival_block.startswith("Arrival (BARIX -> touchdown,")
    assert "decel to M1.0" in arrival_block
    assert "level at M0.95" in arrival_block
    assert "descent" in arrival_block
    assert "approach" in arrival_block
    assert "total" in arrival_block
    assert "kt schedule" in arrival_block
    assert "conc_subsonic_cruise.csv" in arrival_block


def test_run_report_prints_flat_override_arrival_block(tmp_path, capsys):
    """--decel-descent-min forces a short flat block, not a segment
    breakdown whose rows wouldn't sum to the flat total."""
    npz_path = _still_air_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, zfw_t=92.0, min_landing_fuel_t=10.0,
                decel_descent_min=35.0)

    printed = capsys.readouterr().out
    arrival_block = printed.split("\n\n")[1]

    assert arrival_block.startswith("Arrival (BARIX -> touchdown,")
    assert "FLAT OVERRIDE" in arrival_block
    assert "decel to M1.0" not in arrival_block
    assert "total" in arrival_block


def test_run_report_tow_override_skips_fixed_point(tmp_path, capsys):
    """--tow is an optional override on top of --zfw: no fixed point, ZFW
    stays as given, and the printed block reports fuel loaded against the
    fuel the trip needs instead of an iteration count."""
    npz_path = _still_air_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, zfw_t=92.0, tow_t=DEFAULT_TOW_T, decel_descent_min=35.0)

    printed = capsys.readouterr().out
    fuel_block = printed.split("\n\n")[0]

    assert f"take-off weight {DEFAULT_TOW_T:.1f} t" in fuel_block
    assert "ZFW 92.0 t" in fuel_block
    assert "TOW override, no fixed point" in fuel_block
    assert "fuel loaded" in fuel_block
    assert "converged in" not in fuel_block


def test_run_report_flags_boundary_clamp(tmp_path, capsys):
    """A ZFW low enough that the fixed point clamps against the climb
    table's floor must say so in the fuel plan block, not just the search
    CSV's flags column."""
    npz_path = _still_air_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, zfw_t=50.0, min_landing_fuel_t=10.0,
                decel_descent_min=35.0)

    printed = capsys.readouterr().out
    fuel_block = printed.split("\n\n")[0]

    assert "!!" in fuel_block
    assert "tow_below_climb_table" in fuel_block


def test_run_report_missing_subsonic_npz_and_no_override_raises(tmp_path):
    """Neither --subsonic-npz nor --decel-descent-min given -- the real
    arrival model has nothing to sample wind from, so this must fail loud
    rather than silently falling back to something flat."""
    npz_path = _still_air_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    with pytest.raises(ValueError, match="subsonic_data"):
        run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                    out_path=out_path, zfw_t=92.0)


def test_run_report_end_to_end_with_zfw_and_subsonic_npz(tmp_path, capsys):
    """search/report's real path: --zfw + --subsonic-npz, no legacy
    override -- the real per-day arrival.arrival() model runs inside the
    fixed point."""
    npz_path = _still_air_npz(tmp_path)
    subsonic_npz_path = _still_air_subsonic_npz(tmp_path)
    arrival_upper_npz_path = _still_air_arrival_upper_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, zfw_t=92.0, min_landing_fuel_t=10.0,
                subsonic_npz_path=subsonic_npz_path,
                arrival_upper_npz_path=arrival_upper_npz_path)

    printed = capsys.readouterr().out
    fuel_block, arrival_block = printed.split("\n\n")[:2]

    assert "converged in" in fuel_block
    assert "flat override" not in arrival_block
    assert "kt schedule" in arrival_block


def test_arrival_fuel_inside_fixed_point_converges_tow_higher(tmp_path):
    """The whole point of wiring arrival.py into fuel.py's fixed point: at a
    fixed ZFW, the real arrival model's fuel (several tonnes, vs the old
    flat 2.0 t DESCENT_FUEL_T it replaces) feeds back into TOW, converging
    meaningfully higher than the flat legacy override does -- amplified
    further by the climb feedback (more TOW -> more climb burn -> more
    TOW), so the final gap is bigger than the raw arrival-fuel delta alone."""
    npz_path = _still_air_npz(tmp_path)
    subsonic_npz_path = _still_air_subsonic_npz(tmp_path)
    arrival_upper_npz_path = _still_air_arrival_upper_npz(tmp_path)

    plan = parse_pln(SAMPLE_PLN)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs)
    cc_idx = np.flatnonzero(mask)
    cc_legs = [legs[i] for i in cc_idx]
    arrival_idx = np.flatnonzero(~mask)
    arrival_legs = [legs[i] for i in arrival_idx]
    arrival_nm = legs[-1].cum_nm - cc_legs[-1].cum_nm

    data = load_legs_npz(npz_path)
    subsonic_data = load_legs_npz(subsonic_npz_path)
    arrival_upper_data = load_legs_npz(arrival_upper_npz_path)
    dep_i8 = np.array([1455289200000000000], dtype="int64")  # 2016-02-12 15:00Z
    zfw_t = np.array([92.0])

    tow_flat, *_ = resolve_tow_and_arrival(
        cc_legs, cc_idx, arrival_legs, arrival_nm, data, None, dep_i8,
        zfw_t=zfw_t, min_landing_fuel_t=10.0, decel_descent_min=35.0,
    )
    tow_real, *_ = resolve_tow_and_arrival(
        cc_legs, cc_idx, arrival_legs, arrival_nm, data, subsonic_data, dep_i8,
        zfw_t=zfw_t, min_landing_fuel_t=10.0,
        arrival_upper_data=arrival_upper_data,
    )

    diff_t = float(tow_real[0] - tow_flat[0])
    assert 4.0 < diff_t < 15.0, f"expected several t higher TOW with real arrival, got {diff_t:.2f} t"


def test_decel_descent_min_reproduces_old_flat_behaviour(tmp_path, capsys):
    """--decel-descent-min forces the exact pre-B3 numbers: the given
    minutes and arrival.LEGACY_FLAT_FUEL_T (2.0 t), not the real model."""
    npz_path = _still_air_npz(tmp_path)
    out_path = tmp_path / "report.csv"

    run_report(SAMPLE_PLN, npz_path, dt.date(2016, 2, 12), 10,
                out_path=out_path, zfw_t=92.0, tow_t=DEFAULT_TOW_T, decel_descent_min=35.0)

    printed = capsys.readouterr().out
    fuel_block = printed.split("\n\n")[0]
    arrival_block = printed.split("\n\n")[1]

    assert f"arrival {LEGACY_FLAT_FUEL_T:.1f}" in fuel_block
    assert "FLAT OVERRIDE" in arrival_block
    assert "35.0 min" in arrival_block
    assert f"{LEGACY_FLAT_FUEL_T:.2f} t" in arrival_block
