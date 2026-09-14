"""Tests for the arrival model (decel waypoint -> touchdown)."""

import time

import numpy as np
import pytest

from concopt import arrival, atmos


def _isa_temp_k(level_fl, isa_dev_c):
    """Static temperature (K) at a flight level, ISA + deviation."""
    isa_t_k, _ = atmos.isa(np.asarray(level_fl, float) * 100.0 * 0.3048)
    return isa_t_k + isa_dev_c


class Wind:
    """A wind_at_fl that is genuinely PER CANDIDATE.

    Takes one wind and one ISA deviation per candidate and returns arrays
    aligned with the candidate axis, which is what any real implementation
    reading ERA5 must do. A mock that ignores the candidate axis and simply
    broadcasts a constant hides shape bugs -- an earlier band-masked
    implementation passed such a mock and then crashed on the first real
    candidate set that straddled the ISA-10 band boundary.
    """

    def __init__(self, wind_kt, isa_dev_c):
        self.wind_kt = np.atleast_1d(np.asarray(wind_kt, float))
        self.isa_dev_c = np.atleast_1d(np.asarray(isa_dev_c, float))
        self.seen_shapes = []

    def __call__(self, level_fl):
        level_fl = np.asarray(level_fl, float)
        self.seen_shapes.append(level_fl.shape)
        n = level_fl.shape[0]
        wind = np.broadcast_to(self.wind_kt, (n,))
        dev = np.broadcast_to(self.isa_dev_c, (n,))
        return {"wind_kt": wind, "temp_k": _isa_temp_k(level_fl, dev)}


def _still_air(n=1, isa_dev_c=0.0):
    return Wind(np.zeros(n), np.full(n, isa_dev_c))


ROUTE_NM = 307.0  # JFK->LHR post-decel distance, for reference cases


class TestSegmentsSum:
    """Segments reconstruct the totals they were split from."""

    @pytest.mark.parametrize("speed", arrival.SCHEDULES_KT)
    def test_distances_sum_to_arrival_less_approach(self, speed):
        """decel_nm + level_nm + descent_nm == arrival_nm - APPROACH_NM."""
        arrival_nm = np.array([260.0, 307.0, 350.0])
        isa_dev = np.array([5.0, -15.0, 0.0])
        r = arrival.arrival(
            550.0, arrival_nm, Wind(np.zeros(3), isa_dev), isa_dev, speed=speed
        )
        total = r["decel_nm"] + r["level_nm"] + r["descent_nm"]
        assert np.allclose(total, arrival_nm - arrival.APPROACH_NM, atol=0.1)

    @pytest.mark.parametrize("speed", arrival.SCHEDULES_KT)
    def test_times_and_fuels_sum_to_totals(self, speed):
        arrival_nm = np.array([260.0, 307.0, 350.0])
        isa_dev = np.array([5.0, -15.0, 0.0])
        r = arrival.arrival(
            550.0, arrival_nm, Wind(np.zeros(3), isa_dev), isa_dev, speed=speed
        )
        t = (r["decel_time_min"] + r["level_time_min"] + r["descent_time_min"]
             + arrival.APPROACH_MIN)
        f = (r["decel_fuel_t"] + r["level_fuel_t"] + r["descent_fuel_t"]
             + arrival.APPROACH_FUEL_T)
        assert np.allclose(t, r["time_min"], atol=1e-9)
        assert np.allclose(f, r["fuel_t"], atol=1e-9)


class TestZeroWindReference:
    """Absolute sanity anchors -- these catch a missing whole segment."""

    def test_fl600_warm_380kt(self):
        """~29.1 min and ~6.5 t. ~2 t means the level segment vanished;
        ~17 min means the decel did."""
        w = _still_air(1, isa_dev_c=15.0)
        r = arrival.arrival(600.0, np.array([ROUTE_NM]), w, np.array([15.0]), speed=380)
        assert 29.1 * 0.9 < r["time_min"][0] < 29.1 * 1.1
        assert 6.5 * 0.9 < r["fuel_t"][0] < 6.5 * 1.1

    def test_beats_the_placeholder_it_replaces(self):
        """The whole point: the old flat 2.0 t was ~4.8 t light."""
        w = _still_air(1, isa_dev_c=0.0)
        r = arrival.arrival(580.0, np.array([ROUTE_NM]), w, np.array([0.0]))
        assert r["fuel_t"][0] > 4.0
        assert 25.0 < r["time_min"][0] < 40.0


class TestBands:
    """Two bands, and candidates may straddle them in one call."""

    def test_band_boundary_is_isa_minus_10(self):
        assert _band(-9.9) == "warm"
        assert _band(-10.0) == "cold"
        assert _band(-10.1) == "cold"

    def test_candidates_split_across_bands_in_one_call(self):
        """REGRESSION: a band-masked implementation crashed here, because it
        handed wind_at_fl a candidate subset it could not identify."""
        isa_dev = np.array([5.0, 0.0, -15.0, -25.0])  # warm, warm, cold, cold
        w = Wind(np.array([-80.0, -20.0, 20.0, 80.0]), isa_dev)
        r = arrival.arrival(550.0, np.full(4, ROUTE_NM), w, isa_dev, speed=350)
        assert np.all(np.isfinite(r["time_min"]))
        # Winds ascend across the candidates, so times must strictly descend.
        assert np.all(np.diff(r["time_min"]) < 0)

    def test_split_matches_per_band_evaluated_alone(self):
        """Each candidate gets the same answer whether or not it shares the
        call with candidates from the other band."""
        isa_dev = np.array([5.0, -15.0])
        winds = np.array([30.0, -30.0])
        together = arrival.arrival(
            550.0, np.full(2, ROUTE_NM), Wind(winds, isa_dev), isa_dev, speed=350
        )
        for i in range(2):
            alone = arrival.arrival(
                550.0, np.array([ROUTE_NM]),
                Wind(winds[i:i + 1], isa_dev[i:i + 1]),
                isa_dev[i:i + 1], speed=350,
            )
            assert np.isclose(together["time_min"][i], alone["time_min"][0])
            assert np.isclose(together["fuel_t"][i], alone["fuel_t"][0])

    def test_wind_callable_always_sees_full_candidate_axis(self):
        """wind_at_fl must never be asked about a subset it cannot identify."""
        isa_dev = np.array([5.0, -15.0, 0.0])
        w = Wind(np.zeros(3), isa_dev)
        arrival.arrival(550.0, np.full(3, ROUTE_NM), w, isa_dev)
        assert w.seen_shapes, "wind_at_fl was never called"
        assert all(s == (3,) for s in w.seen_shapes), w.seen_shapes


def _band(isa_dev_c):
    return "warm" if arrival._band_is_warm(isa_dev_c) else "cold"


class TestWind:
    """Tailwind helps, headwind hurts, monotonically."""

    def test_monotonic_in_wind(self):
        winds = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        isa_dev = np.zeros(5)
        r = arrival.arrival(
            550.0, np.full(5, ROUTE_NM), Wind(winds, isa_dev), isa_dev, speed=350
        )
        assert np.all(np.diff(r["time_min"]) < 0)

    def test_tailwind_faster_headwind_slower_than_still_air(self):
        isa_dev = np.zeros(3)
        r = arrival.arrival(
            550.0, np.full(3, ROUTE_NM),
            Wind(np.array([-60.0, 0.0, 60.0]), isa_dev), isa_dev, speed=350,
        )
        assert r["time_min"][0] > r["time_min"][1] > r["time_min"][2]

    def test_wind_shifts_distance_split_not_just_time(self):
        """A tailwind stretches the decel/descent ground distance, which eats
        into the level segment."""
        isa_dev = np.zeros(2)
        r = arrival.arrival(
            550.0, np.full(2, ROUTE_NM),
            Wind(np.array([0.0, 80.0]), isa_dev), isa_dev, speed=350,
        )
        assert r["decel_nm"][1] > r["decel_nm"][0]
        assert r["descent_nm"][1] > r["descent_nm"][0]
        assert r["level_nm"][1] < r["level_nm"][0]


class TestAutoSchedule:
    def test_auto_never_worse_than_best_forced(self):
        arrival_nm = np.full(6, ROUTE_NM)
        isa_dev = np.array([10.0, 5.0, 0.0, -5.0, -15.0, -25.0])
        winds = np.array([-60.0, -20.0, 0.0, 20.0, 60.0, 100.0])
        kw = dict(arrival_nm=arrival_nm, isa_dev_at_cruise=isa_dev)
        auto = arrival.arrival(550.0, wind_at_fl=Wind(winds, isa_dev), **kw)
        forced = [
            arrival.arrival(550.0, wind_at_fl=Wind(winds, isa_dev), speed=s, **kw)["time_min"]
            for s in arrival.SCHEDULES_KT
        ]
        assert np.all(auto["time_min"] <= np.min(np.stack(forced), axis=0) + 1e-9)

    def test_auto_reports_the_schedule_it_picked(self):
        isa_dev = np.zeros(3)
        w = Wind(np.zeros(3), isa_dev)
        r = arrival.arrival(550.0, np.full(3, ROUTE_NM), w, isa_dev)
        assert set(np.unique(r["schedule_kt"])) <= set(arrival.SCHEDULES_KT)
        for i, spd in enumerate(r["schedule_kt"]):
            assert np.isclose(r["time_min"][i], r["by_schedule"][spd]["time_min"][i])

    def test_by_schedule_exposes_all_three_for_total_time_choice(self):
        """The caller must be able to re-decide on total time once fuel feeds
        back through the climb, so all three must survive the call."""
        isa_dev = np.zeros(2)
        r = arrival.arrival(550.0, np.full(2, ROUTE_NM), Wind(np.zeros(2), isa_dev), isa_dev)
        assert set(r["by_schedule"]) == set(arrival.SCHEDULES_KT)
        for spd in arrival.SCHEDULES_KT:
            assert r["by_schedule"][spd]["time_min"].shape == (2,)

    def test_faster_schedule_costs_fuel(self):
        """380 kt buys time and pays for it in fuel -- the trade the caller
        has to resolve against the climb."""
        isa_dev = np.zeros(1)
        got = {
            s: arrival.arrival(
                580.0, np.array([ROUTE_NM]), _still_air(1), isa_dev, speed=s
            )
            for s in arrival.SCHEDULES_KT
        }
        assert got[380]["time_min"][0] < got[325]["time_min"][0]
        assert got[380]["fuel_t"][0] > got[325]["fuel_t"][0]

    def test_rejects_an_unknown_schedule(self):
        with pytest.raises(ValueError, match="speed must be"):
            arrival.arrival(550.0, np.array([ROUTE_NM]), _still_air(1),
                            np.zeros(1), speed=300)


class TestFlags:
    """Flags, never exceptions."""

    def test_short_arrival_clamps_level_nm_and_flags(self):
        r = arrival.arrival(550.0, np.array([150.0]), _still_air(1),
                            np.zeros(1), speed=325)
        assert r["level_nm"][0] == 0.0
        assert r["level_nm_clamped"][0]
        assert "level_nm_clamped" in r["flags"][0]

    def test_ample_arrival_is_unflagged(self):
        r = arrival.arrival(550.0, np.array([ROUTE_NM]), _still_air(1),
                            np.zeros(1), speed=325)
        assert not r["level_nm_clamped"][0]
        assert r["flags"][0] == ""

    @pytest.mark.parametrize("fl,expect", [(450.0, True), (610.0, True),
                                           (470.0, False), (600.0, False)])
    def test_cruise_fl_outside_table_clamps_and_flags(self, fl, expect):
        r = arrival.arrival(fl, np.array([ROUTE_NM]), _still_air(1),
                            np.zeros(1), speed=350)
        assert bool(r["cruise_fl_clamped"][0]) is expect
        assert ("cruise_fl_clamped" in r["flags"][0]) is expect
        assert np.isfinite(r["time_min"][0])

    def test_clamped_cruise_fl_matches_the_bound_it_clamped_to(self):
        out = arrival.arrival(610.0, np.array([ROUTE_NM]), _still_air(1),
                              np.zeros(1), speed=350)
        at_max = arrival.arrival(600.0, np.array([ROUTE_NM]), _still_air(1),
                                 np.zeros(1), speed=350)
        assert np.isclose(out["time_min"][0], at_max["time_min"][0])

    def test_headwind_beyond_tas_flags_rather_than_dividing_by_zero(self):
        isa_dev = np.zeros(1)
        r = arrival.arrival(550.0, np.array([ROUTE_NM]),
                            Wind(np.array([-700.0]), isa_dev), isa_dev, speed=350)
        assert r["level_gs_nonpositive"][0]
        assert "level_gs_nonpositive" in r["flags"][0]
        assert np.isinf(r["time_min"][0])

    def test_flags_can_combine(self):
        r = arrival.arrival(650.0, np.array([150.0]), _still_air(1),
                            np.zeros(1), speed=325)
        assert "cruise_fl_clamped" in r["flags"][0]
        assert "level_nm_clamped" in r["flags"][0]


class TestShapes:
    def test_scalars_broadcast_to_one_candidate(self):
        r = arrival.arrival(550.0, ROUTE_NM, _still_air(1), 0.0)
        for k in ("time_min", "fuel_t", "schedule_kt", "level_nm", "flags"):
            assert np.asarray(r[k]).shape == (1,), k

    def test_every_returned_array_has_the_candidate_shape(self):
        n = 7
        isa_dev = np.linspace(-25.0, 10.0, n)
        r = arrival.arrival(550.0, np.full(n, ROUTE_NM),
                            Wind(np.zeros(n), isa_dev), isa_dev)
        for k, v in r.items():
            if k == "by_schedule":
                continue
            assert np.asarray(v).shape == (n,), f"{k} has shape {np.shape(v)}"


class TestPerformance:
    def test_5000_candidates_under_a_second(self):
        n = 5000
        rng = np.random.default_rng(0)
        isa_dev = rng.uniform(-25.0, 12.0, n)
        winds = rng.uniform(-90.0, 90.0, n)
        cruise_fl = rng.uniform(470.0, 600.0, n)
        w = Wind(winds, isa_dev)
        t0 = time.perf_counter()
        r = arrival.arrival(cruise_fl, np.full(n, ROUTE_NM), w, isa_dev)
        elapsed = time.perf_counter() - t0
        assert r["time_min"].shape == (n,)
        assert np.all(np.isfinite(r["time_min"]))
        assert elapsed < 1.0, f"{n} candidates took {elapsed:.3f} s"


class TestLevelWindKt:
    """level_wind_kt -- exposed for report.py's Arrival block ("FL312, +18
    kt"), B3."""

    def test_present_and_shaped_like_every_other_array(self):
        isa_dev = np.zeros(3)
        r = arrival.arrival(550.0, np.full(3, ROUTE_NM), Wind(np.zeros(3), isa_dev), isa_dev)
        assert r["level_wind_kt"].shape == (3,)

    def test_matches_the_wind_at_the_level_segment(self):
        """A constant wind everywhere is exactly what shows up as
        level_wind_kt (the level segment is flown at decel_end_fl)."""
        isa_dev = np.zeros(1)
        r = arrival.arrival(550.0, np.array([ROUTE_NM]),
                            Wind(np.array([40.0]), isa_dev), isa_dev, speed=350)
        assert r["level_wind_kt"][0] == pytest.approx(40.0)


class TestFlatArrival:
    """flat_arrival -- the --decel-descent-min legacy override, B3."""

    def test_matches_the_old_flat_constants(self):
        r = arrival.flat_arrival(550.0, ROUTE_NM, _still_air(1), 0.0,
                                 decel_descent_min=35.0)
        assert r["time_min"][0] == pytest.approx(35.0)
        assert r["fuel_t"][0] == pytest.approx(arrival.LEGACY_FLAT_FUEL_T)

    def test_schedule_kt_is_zero_a_sentinel_for_no_real_schedule(self):
        r = arrival.flat_arrival(550.0, ROUTE_NM, _still_air(1), 0.0,
                                 decel_descent_min=35.0)
        assert r["schedule_kt"][0] == 0

    def test_ignores_wind_and_speed(self):
        headwind = arrival.flat_arrival(
            550.0, ROUTE_NM, Wind(np.array([-500.0]), np.zeros(1)), 0.0,
            speed=325, decel_descent_min=35.0,
        )
        tailwind = arrival.flat_arrival(
            550.0, ROUTE_NM, Wind(np.array([500.0]), np.zeros(1)), 0.0,
            speed=380, decel_descent_min=35.0,
        )
        assert headwind["time_min"][0] == tailwind["time_min"][0] == pytest.approx(35.0)

    def test_broadcasts_scalars_like_arrival_does(self):
        r = arrival.flat_arrival(550.0, ROUTE_NM, _still_air(1), 0.0,
                                 decel_descent_min=35.0)
        for k in ("time_min", "fuel_t", "schedule_kt", "level_nm", "flags"):
            assert np.asarray(r[k]).shape == (1,), k

    def test_same_call_signature_as_arrival_drop_in_for_fuel_py(self):
        """fuel.fixed_point_fuel_iteration calls whichever arrival_fn it was
        given the same way regardless of which one it is -- this is the
        contract that makes that substitution safe."""
        import inspect
        arrival_params = list(inspect.signature(arrival.arrival).parameters)
        flat_params = list(inspect.signature(arrival.flat_arrival).parameters)
        assert flat_params[:len(arrival_params)] == arrival_params
