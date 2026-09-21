"""report.flight_profile -- the full brake-release -> touchdown profile the
day-search notebook plots. Still-air synthetic .npz (test_report's fixtures), so
these pin the profile's *structure* (it joins up, telescopes to the fuel plan,
carries the right limits and NaNs), not any particular weather's numbers."""
import datetime as dt

import numpy as np
import pytest

from concopt.atmos import cas_from_mach, isa, mach_from_cas
from concopt.params import FT_TO_M, KT_TO_MS, SUBSONIC_LIMIT_MACH
from concopt.report import flight_profile
from concopt.route import build_legs, parse_pln, position_at_cum_nm
from tests.test_report import (SAMPLE_PLN, _still_air_arrival_upper_npz, _still_air_npz,
                               _still_air_subsonic_npz)

ZFW_T = 92.0


@pytest.fixture(scope="module")
def prof(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("profile")
    return flight_profile(
        SAMPLE_PLN, _still_air_npz(tmp), dt.date(2016, 2, 12), 10, ZFW_T,
        subsonic_npz_path=_still_air_subsonic_npz(tmp),
        arrival_upper_npz_path=_still_air_arrival_upper_npz(tmp),
        runway_penalties_s=(30.0, 60.0))


def test_cas_from_mach_round_trips_mach_from_cas():
    _, p_pa = isa(45_000.0 * FT_TO_M)
    for cas_kt in (250.0, 350.0, 380.0, 530.0):
        mach = mach_from_cas(cas_kt * KT_TO_MS, p_pa)
        assert cas_from_mach(mach, p_pa) / KT_TO_MS == pytest.approx(cas_kt, rel=1e-6)


def test_position_at_cum_nm_hits_route_ends():
    legs = build_legs(parse_pln(SAMPLE_PLN)["waypoints"])
    first, last = parse_pln(SAMPLE_PLN)["waypoints"][0], parse_pln(SAMPLE_PLN)["waypoints"][-1]
    lat0, lon0, _ = position_at_cum_nm(legs, 0.0)
    lat1, lon1, _ = position_at_cum_nm(legs, legs[-1].cum_nm)
    assert (lat0, lon0) == pytest.approx(first[1:], abs=1e-3)
    assert (lat1, lon1) == pytest.approx(last[1:], abs=1e-3)


def test_profile_segments_join_up(prof):
    """Every segment starts where the last one ended, in distance, time and
    weight -- the climb table, the cruise march and the arrival ramps are three
    separate computations, so this is what proves they stitch."""
    p = prof["profile"]
    starts, ends = p.iloc[0::2].reset_index(drop=True), p.iloc[1::2].reset_index(drop=True)
    for col in ("cum_nm", "elapsed_s", "weight_t"):
        assert ends[col].iloc[:-1].to_numpy() == pytest.approx(starts[col].iloc[1:].to_numpy(), abs=1e-6)
    assert np.all(np.diff(p["cum_nm"]) >= -1e-9)
    assert np.all(np.diff(p["elapsed_s"]) >= -1e-9)
    assert p["alt_ft"].iloc[0] == 0.0 and p["alt_ft"].iloc[-1] == 0.0


def test_profile_telescopes_to_the_fuel_plan(prof):
    p, s = prof["profile"], prof["summary"]
    tol = 0.06  # fuel.DEFAULT_TOLERANCE_T -- the march flew the iterate before the last update
    assert p["weight_t"].iloc[0] == pytest.approx(s["tow_t"], abs=tol)
    assert p["weight_t"].iloc[-1] == pytest.approx(s["landing_weight_t"], abs=tol)
    assert s["fuel_loaded_t"] == pytest.approx(s["tow_t"] - ZFW_T)
    assert s["landing_fuel_t"] == pytest.approx(s["reserve_t"], abs=tol)  # fixed point: lands on the reserve
    assert (s["climb_fuel_t"] + s["cruise_fuel_t"] + s["arrival_fuel_t"]) == pytest.approx(s["trip_fuel_t"])
    assert p["fuel_remaining_t"].iloc[-1] == pytest.approx(s["landing_fuel_t"], abs=tol)


def test_phase_table_reconciles(prof):
    ph = prof["phases"].set_index("phase")
    flown = ["climb", "acceleration", "cruise", "deceleration", "subsonic cruise", "descent", "approach"]
    assert list(ph.index[1:-2]) == flown
    total = ph.loc["TOTAL"]
    assert total["duration_s"] == pytest.approx(
        ph.loc[flown, "duration_s"].sum() + 30.0 + 60.0)  # runway penalties ride on top
    assert total["fuel_t"] == pytest.approx(ph.loc[flown, "fuel_t"].sum())
    assert total["duration_s"] == pytest.approx(prof["summary"]["total_time_s"])
    assert ph.loc["climb", "end_nm"] == pytest.approx(prof["summary"]["linnd_nm"])
    assert ph.loc["acceleration", "end_nm"] == pytest.approx(prof["summary"]["top_of_climb_nm"], abs=1e-6)
    assert ph.loc["runway penalty (KJFK)", "duration_s"] == 30.0


def test_climb_has_no_speeds_and_arrival_does(prof):
    """conc_climb.csv carries no speeds -- climb Mach/TAS/CAS stay NaN, only the
    whole-climb mean GS is filled; the arrival's are derived."""
    p = prof["profile"]
    climb = p[p["phase"].isin(["climb", "acceleration"])]
    assert climb[["mach", "tas_kt", "cas_kt"]].isna().all().all()
    assert (climb["speed_basis"] == "mean").all() and climb["gs_kt"].notna().all()
    for phase in ("cruise", "deceleration", "subsonic cruise", "descent"):
        g = p[p["phase"] == phase]
        assert g[["mach", "tas_kt", "cas_kt", "gs_kt"]].notna().all().all(), phase


def test_mach_limit_is_subsonic_before_linnd_and_after_decel(prof):
    p = prof["profile"]
    for phase in ("climb", "subsonic cruise", "descent", "approach"):
        assert (p.loc[p["phase"] == phase, "mach_limit"] == SUBSONIC_LIMIT_MACH).all(), phase
    for phase in ("acceleration", "cruise", "deceleration"):
        assert (p.loc[p["phase"] == phase, "mach_limit"] > SUBSONIC_LIMIT_MACH).all(), phase


def test_arrival_speeds_follow_the_model(prof):
    p = prof["profile"]
    sched = prof["summary"]["schedule_kt"]
    assert p.loc[p["phase"] == "descent", "cas_kt"].to_numpy() == pytest.approx(sched, abs=0.01)  # CAS schedule
    assert p.loc[p["phase"] == "subsonic cruise", "mach"].to_numpy() == pytest.approx(0.95)
    decel = p[p["phase"] == "deceleration"]
    assert decel["mach"].iloc[-1] == pytest.approx(SUBSONIC_LIMIT_MACH)
    assert decel["mach"].is_monotonic_decreasing


def test_ceiling_and_cas_limit_columns(prof):
    p = prof["profile"]
    assert p.loc[p["phase"] == "cruise", "ceiling_ft"].notna().all()
    assert p.loc[p["phase"] != "cruise", "ceiling_ft"].isna().all()
    assert p["cas_limit_kt"].notna().all()
    assert p.loc[p["alt_ft"] > 43_000.0, "cas_limit_kt"].to_numpy() == pytest.approx(530.0)
