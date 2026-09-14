"""Day/time scan for the JFK-LHR route. For every candidate departure,
climb (TOW-based, conc_data.climb_to) then march across the cruise legs
picking the best-wind flight level at each one (limits.best_level), and
rank candidates by total block time. Reads only the .npz written by
era5.reduce_to_legs; never touches the netCDF.

march_legs is the shared per-leg march: run_search calls it for ~31,000
candidates at once and collapses the result to means, report.run_report
calls it for a single candidate and keeps every per-leg quantity.
"""
import datetime as dt
import functools
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from concopt import arrival, fuel, limits, runways
from concopt.atmos import KT_TO_MS, fl_to_pressure, isa, pressure_to_fl, speed_of_sound
from concopt.data.conc_data import CLIMB_BANDS, climb_to, fuel_total_kgh_table
from concopt.era5 import ARCHIVE_START, load_legs_npz, load_surface_npz
from concopt.route import build_legs, climb_cruise_segment, parse_pln

NY_TZ = ZoneInfo("America/New_York")
DEPARTURE_LOCAL_HOURS = range(8, 15)  # 08:00..14:00 local, inclusive

NM_TO_M = 1852.0

# Take-off weight. From here, weight is a state variable: the climb burns
# it down to a top-of-climb mass (conc_data.climb_to), then it's integrated
# leg by leg from conc_data.fuel_total_kgh_table (Air France performance
# table) across the cruise, not a linear schedule against cum_nm -- see
# march_legs. It drives ceiling_ft, which is what actually keeps the
# optimiser off levels the aircraft can't hold; it does not change max_tas
# above FL430 (530 kt CAS at every weight there). --tow on the CLI.
DEFAULT_TOW_T = 185.0

# The climb table's top-of-climb level (max level_fl in conc_climb.csv) --
# climb_to(TOP_OF_CLIMB_FL, tow_t, temp_band) gives the brake-release-to-
# top-of-climb time/distance/mass the march starts the cruise from.
TOP_OF_CLIMB_FL = 502.0

# cum_nm within which _climb_conditions samples ISA deviation/wind to pick
# conc_climb.csv's temp_band and correct the climb's ground distance for
# wind -- conc_climb.csv's shortest climb span is ~210 nm, so this stays
# inside every band's actual climb, cold or warm.
CLIMB_BAND_SAMPLE_NM = 300.0

# best_level picks from this 16-level grid (1000 ft / FL10 steps) rather than
# the 4 raw ERA5 pressure levels -- interpolated per leg below, not stored in
# the npz (16 levels there would be ~350 MB vs ~32 MB for 4). FL450-FL600
# sits inside the ERA5 mandatory-level span (150-70 hPa = FL446-FL605), so
# every target is bracketed -- see the assert in march_legs.
TARGET_FL = np.arange(450.0, 601.0, 10.0)

# LEGACY -- no longer this module's own default. arrival.arrival() (below)
# replaced this flat 35-minute placeholder for search.py/report.py's own
# brakes-release-to-touchdown estimate; the constant is kept only because
# inflight.py's flight recorder still compares a measured decel-to-touchdown
# time against it (compare_to_report) and that wiring is a separate, later
# task. --decel-descent-min still exists on the CLI, but now as an explicit
# override that forces arrival.flat_arrival's flat (time, fuel) pair instead
# of the real per-day model, for comparing old and new numbers -- it is no
# longer given a default value, see cli.py.
DECEL_DESCENT_S = 35.0 * 60.0


def candidate_departures():
    """Every date from era5.ARCHIVE_START to today, at 08:00-14:00
    America/New_York (7 candidates/day) -- about 4,419 x 7 = 30,933 rows.
    Built tz-aware with zoneinfo and converted to UTC: the offset changes
    with DST (EDT/EST), so naive hour arithmetic would silently shift
    every winter candidate by an hour. Returns a DataFrame with one row
    per candidate: local_date, local_hour, departure_utc (naive, UTC)."""
    rows = []
    d = ARCHIVE_START
    today = dt.date.today()
    while d <= today:
        for h in DEPARTURE_LOCAL_HOURS:
            local = dt.datetime(d.year, d.month, d.day, h, tzinfo=NY_TZ)
            departure_utc = local.astimezone(dt.timezone.utc).replace(tzinfo=None)
            rows.append((d, h, np.datetime64(departure_utc)))
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows, columns=["local_date", "local_hour", "departure_utc"])


def local_to_departure_utc(local_date, local_hour):
    """One local_date/local_hour (America/New_York) -> naive UTC datetime,
    the same tz-aware conversion candidate_departures uses per row. Lets
    report.py build a single candidate the same way search builds ~31,000."""
    local = dt.datetime(local_date.year, local_date.month, local_date.day,
                         local_hour, tzinfo=NY_TZ)
    return local.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _format_hmm(seconds):
    """Seconds -> 'h:mm' string, for display only."""
    total_min = int(round(seconds / 60.0))
    h, m = divmod(total_min, 60)
    return f"{h}:{m:02d}"


def _climb_conditions(data, dep_i8, cc_legs, cc_idx):
    """Per-candidate (temp_band, warm_flag, wind_component_kt) describing
    the early part of the route -- everything conc_climb.csv's climb_to
    needs beyond TOW, plus the wind correction its dist_nm (air distance)
    needs to become a ground distance.

    Sampled from the raw ERA5 data at the lowest stored pressure level
    (data["level"][0], ~FL450-ish) over the legs within CLIMB_BAND_SAMPLE_NM
    of the route's start, at departure clock time -- the upper-air .npz only
    carries the 4 mandatory levels used for the supersonic cruise scan,
    nothing below, so this is the best available proxy for climb-altitude
    conditions, not a real climb-profile temperature/wind trace. Good
    enough to bucket the day into one of 3 discrete bands and to apply one
    wind correction to the climb's tabulated air distance -- nothing more.

    Returns temp_band ((n_cand,) string array, one of
    conc_data.CLIMB_BANDS), warm_flag ((n_cand,) bool -- True where the raw
    ISA deviation exceeded the warmest band (+10 C) and was clamped down
    into it; clamping warm is optimistic, so run_search flags these
    candidates rather than silently trusting the climb numbers), and
    wind_component_kt ((n_cand,), along each sampled leg's own track,
    averaged)."""
    sample_local = [i for i, leg in enumerate(cc_legs) if leg.cum_nm <= CLIMB_BAND_SAMPLE_NM]
    if not sample_local:
        sample_local = [0]
    sample_legs = [cc_legs[i] for i in sample_local]
    sample_cols = cc_idx[sample_local]

    times_i8 = data["time"].astype("datetime64[ns]").astype("int64")
    idx1 = np.searchsorted(times_i8, dep_i8, side="right") - 1
    idx1 = np.clip(idx1, 0, len(times_i8) - 2)
    idx2 = idx1 + 1
    t0, t1 = times_i8[idx1], times_i8[idx2]
    frac = np.clip((dep_i8 - t0) / (t1 - t0), 0.0, 1.0)[:, None]

    u_low = data["u"][:, 0, :][:, sample_cols]  # (n_time, n_sample)
    v_low = data["v"][:, 0, :][:, sample_cols]
    t_low = data["t"][:, 0, :][:, sample_cols]

    u_at = u_low[idx1] + frac * (u_low[idx2] - u_low[idx1])  # (n_cand, n_sample)
    v_at = v_low[idx1] + frac * (v_low[idx2] - v_low[idx1])
    t_at = t_low[idx1] + frac * (t_low[idx2] - t_low[idx1])

    fl_low = float(pressure_to_fl(data["level"][0] * 100.0))
    isa_t_low, _ = isa(fl_low * 100.0 * 0.3048)
    isa_dev = t_at.mean(axis=1) - isa_t_low

    temp_band = np.where(
        isa_dev < -10.0, "isa_minus_20_to_minus_10",
        np.where(isa_dev < 0.0, "isa_minus_10_to_isa", "isa_to_isa_plus_10")
    )
    warm_flag = isa_dev > 10.0

    tracks = np.radians(np.array([leg.track_deg for leg in sample_legs]))
    along = u_at * np.sin(tracks)[None, :] + v_at * np.cos(tracks)[None, :]
    wind_component_kt = along.mean(axis=1) / KT_TO_MS

    return temp_band, warm_flag, wind_component_kt


def _climb_profile(data, dep_i8, cc_legs, cc_idx, tow_t):
    """Per-candidate brake-release-to-top-of-climb profile: mass_t,
    fuel_used_kg, dist_nm (air), ground_dist_nm (wind-corrected, what the
    march actually starts the cruise at), time_min, temp_band, warm_flag.
    tow_t is either a scalar (one --tow for the whole run) or a (n_cand,)
    array -- fuel.py's fixed point marches every candidate at its own
    current TOW estimate, so this has to accept a per-candidate vector, not
    just one shared weight. temp_band varies per candidate, so
    conc_data.climb_to (one temp_band at a time) is called once per band
    (CLIMB_BANDS has only 3) rather than per candidate."""
    n_cand = len(dep_i8)
    temp_band, warm_flag, wind_component_kt = _climb_conditions(data, dep_i8, cc_legs, cc_idx)

    # Scalar or (n_cand,) in, always (n_cand,) here, so the per-band masks
    # below select each candidate's own TOW rather than broadcasting a
    # single weight into a band-sized slot (which raised for any run whose
    # candidates spanned more than one band).
    tow_per_cand = np.broadcast_to(np.asarray(tow_t, dtype=float), (n_cand,))

    fuel_used_kg = np.empty(n_cand)
    dist_nm = np.empty(n_cand)
    time_min = np.empty(n_cand)
    for band in CLIMB_BANDS:
        band_mask = temp_band == band
        if not band_mask.any():
            continue
        _m, f, d, t = climb_to(TOP_OF_CLIMB_FL, tow_per_cand[band_mask], band)
        fuel_used_kg[band_mask] = f
        dist_nm[band_mask] = d
        time_min[band_mask] = t

    # mass_t derived from fuel burned, not read off the table's own mass_t
    # column: conc_climb.csv's mass_t and fuel_used_kg are rounded
    # independently in the source manual and disagree by up to 0.5 t at
    # FL502 (tests/test_climb.py's own tolerance on this). fuel.py's trip
    # fuel must telescope exactly to tow - weight_at_touchdown - descent, and
    # the reserve constrains fuel, not the table's mass column, so fuel is
    # the one to trust here.
    mass_t = tow_per_cand - fuel_used_kg / 1000.0

    ground_dist_nm = dist_nm + wind_component_kt * time_min / 60.0

    return dict(mass_t=mass_t, fuel_used_kg=fuel_used_kg, dist_nm=dist_nm,
                ground_dist_nm=ground_dist_nm, time_min=time_min,
                temp_band=temp_band, warm_flag=warm_flag)


def march_legs(cc_legs, cc_idx, data, dep_i8, tow_t=DEFAULT_TOW_T,
                cruise_mach=limits.CRUISE_MACH):
    """The whole per-leg march, vectorised across candidates -- dep_i8 (int64
    ns UTC timestamps) may be length 1 (report.py, one candidate) or ~31,000
    (run_search), every operation inside the per-leg loop treats it the
    same way. cc_legs/cc_idx are the climb+cruise Leg objects (brake release
    through the decel point) and their positions in data's leg axis (from
    route.climb_cruise_segment).

    The march starts with a TOW-based climb (conc_data.climb_to): brake
    release to top of climb (TOP_OF_CLIMB_FL) burns tow_t down to a
    top-of-climb mass over some ground distance and time, both used as the
    cruise's starting state instead of the old fixed WEIGHT_ACCEL_T/
    DEPARTURE_TO_ACCEL_S. Legs wholly inside that climb distance contribute
    nothing further (effective distance 0 -- see eff_dist_nm below); a leg
    straddling the top of climb contributes only its portion past it. Since
    the climb's ground distance varies per candidate (different TOW-fixed,
    weather-varying temp_band and wind correction), so does which legs are
    "wholly inside" -- eff_dist_nm is computed per (candidate, leg) rather
    than by literally splitting Leg objects per candidate.

    At each leg, u/v/t are interpolated from the 4 raw ERA5 pressure levels
    onto the 16-level TARGET_FL grid (linear in log(pressure) -- the
    standard treatment for u/v/t between mandatory levels) before
    limits.best_level picks a level. data["level"] (hPa) runs *decreasing*
    along the level axis (it's stored high-FL-last -> low pressure last),
    so log(pressure) decreases along that axis; np.interp needs increasing
    x, hence the reversal to ascending-pressure order below. FL450-FL600
    sits inside the ERA5 mandatory-level span (150-70 hPa = FL446-FL605),
    so every target is bracketed -- asserted, not extrapolated.

    Weight is a state variable, not a schedule: it starts at the
    top-of-climb mass (see above) and is integrated leg by leg from
    conc_data.fuel_total_kgh_table (the Air France performance table),
    using the weight and ISA deviation at the *start* of each leg -- weight
    feeds both ceiling_ft (so which levels are even reachable changes as
    fuel burns off) and max_mach's CAS component.

    Returns (legs, weight_per_leg, climb): legs is a dict of (n_cand,
    n_legs) arrays -- chosen_fl, best_idx, mach, tas_kt, gs_kt, wind_kt,
    temp_c, isa_dev_k, leg_time_s, eff_dist_nm (the actual, possibly
    partial or zero, ground distance this candidate flew this leg --
    0 for a leg wholly inside the climb, leg.dist_nm for one wholly past
    it), elapsed_s (cumulative seconds since brake release, at the end of
    each leg), and binding (str array: which of cruise_mach/CAS/
    total_temp/ceiling constrains the chosen level -- "ceiling" when the
    chosen level is the highest one ceiling_ft allows, i.e. altitude-capped
    rather than speed-capped) -- plus accumulated_s (n_cand,), the final
    total elapsed time since brake release, and weight_at_barix (n_cand,),
    the weight after the last leg's burn. weight_per_leg is (n_cand,
    n_legs): the weight at the *start* of each leg (pre-burn), i.e. what
    that leg's level/mach selection actually used -- constant at the
    top-of-climb mass through any leading climb-only legs. climb is
    _climb_profile's dict, one row per candidate."""
    n_legs = len(cc_legs)
    n_cand = len(dep_i8)

    climb = _climb_profile(data, dep_i8, cc_legs, cc_idx, tow_t)
    climb_ground_nm = climb["ground_dist_nm"]

    # Per (candidate, leg) ground distance actually flown this leg: 0 for a
    # leg wholly inside the climb, leg.dist_nm for one wholly past it,
    # something in between for the one leg straddling top of climb --
    # this is what lets the climb's per-candidate-varying end point split
    # (or fully consume) legs without literally rebuilding Leg objects per
    # candidate. leg.cum_nm/leg.dist_nm are scalars (one leg at a time,
    # below); climb_ground_nm is (n_cand,).
    leg_end_nm = np.array([leg.cum_nm for leg in cc_legs])
    leg_start_nm = leg_end_nm - np.array([leg.dist_nm for leg in cc_legs])

    times_i8 = data["time"].astype("datetime64[ns]").astype("int64")
    isa_t_per_level, _ = isa(TARGET_FL * 100.0 * 0.3048)  # ISA temp on the 16-level grid, fixed per level

    target_p = fl_to_pressure(TARGET_FL)          # (16,) Pa, decreasing as FL increases
    target_log_p = np.log(target_p)
    src_p = data["level"].astype(float)[::-1] * 100.0  # hPa -> Pa, reversed to ascending
    src_log_p = np.log(src_p)
    assert np.all(np.diff(src_log_p) > 0), \
        "ERA5 levels must be strictly monotonic in pressure"
    assert target_log_p.min() >= src_log_p.min() and target_log_p.max() <= src_log_p.max(), \
        "TARGET_FL must be bracketed by the ERA5 levels -- no extrapolation"

    v_idx0 = np.searchsorted(src_log_p, target_log_p, side="right") - 1
    v_idx0 = np.clip(v_idx0, 0, len(src_log_p) - 2)
    v_idx1 = v_idx0 + 1
    v_frac = (target_log_p - src_log_p[v_idx0]) / (src_log_p[v_idx1] - src_log_p[v_idx0])

    def _interp_vertical(values):
        """(n_cand, n_raw_level) at data["level"] order -> (n_cand, 16) on
        TARGET_FL, linear in log(pressure)."""
        values_asc = values[:, ::-1]  # match src_p's ascending-pressure order
        lo = values_asc[:, v_idx0]
        hi = values_asc[:, v_idx1]
        return lo + v_frac[None, :] * (hi - lo)

    u_ss = data["u"][:, :, cc_idx]      # (n_time, n_level, n_legs)
    v_ss = data["v"][:, :, cc_idx]
    temp_ss = data["t"][:, :, cc_idx]

    weight = climb["mass_t"].copy()  # state, burned off leg by leg below

    accumulated_s = climb["time_min"] * 60.0
    chosen_fl = np.empty((n_cand, n_legs))
    best_idx_out = np.empty((n_cand, n_legs), dtype=int)
    mach = np.empty((n_cand, n_legs))
    tas_kt = np.empty((n_cand, n_legs))
    gs_kt = np.empty((n_cand, n_legs))
    wind_kt = np.empty((n_cand, n_legs))
    temp_c = np.empty((n_cand, n_legs))
    isa_dev_k = np.empty((n_cand, n_legs))
    leg_time_s = np.empty((n_cand, n_legs))
    eff_dist_nm = np.empty((n_cand, n_legs))
    elapsed_s = np.empty((n_cand, n_legs))
    binding = np.empty((n_cand, n_legs), dtype=object)
    weight_per_leg = np.empty((n_cand, n_legs))

    for i, leg in enumerate(cc_legs):
        weight_per_leg[:, i] = weight  # weight at the start of this leg
        this_eff_dist_nm = np.clip(
            leg_end_nm[i] - np.maximum(leg_start_nm[i], climb_ground_nm), 0.0, leg.dist_nm
        )

        leg_time_i8 = dep_i8 + (accumulated_s * 1e9).astype("int64")

        idx1 = np.searchsorted(times_i8, leg_time_i8, side="right") - 1
        idx1 = np.clip(idx1, 0, len(times_i8) - 2)
        idx2 = idx1 + 1
        t0, t1 = times_i8[idx1], times_i8[idx2]
        frac = np.clip((leg_time_i8 - t0) / (t1 - t0), 0.0, 1.0)[:, None]

        u_leg_raw = u_ss[idx1, :, i] + frac * (u_ss[idx2, :, i] - u_ss[idx1, :, i])
        v_leg_raw = v_ss[idx1, :, i] + frac * (v_ss[idx2, :, i] - v_ss[idx1, :, i])
        temp_leg_raw = temp_ss[idx1, :, i] + frac * (temp_ss[idx2, :, i] - temp_ss[idx1, :, i])

        # (n_cand, 4) raw ERA5 levels -> (n_cand, 16) on TARGET_FL.
        u_leg = _interp_vertical(u_leg_raw)
        v_leg = _interp_vertical(v_leg_raw)
        temp_leg = _interp_vertical(temp_leg_raw)

        # weight[:, None]: best_level broadcasts weight_t against the
        # (n_cand, 16) TARGET_FL grid, not just the (n_cand,) reduced
        # quantities used further down.
        best_fl, best_gs_ms, best_idx, _ = limits.best_level(
            TARGET_FL[None, :], temp_leg, u_leg, v_leg, leg.track_deg, weight[:, None], cruise_mach
        )

        along_per_level = (u_leg * np.sin(np.radians(leg.track_deg))
                            + v_leg * np.cos(np.radians(leg.track_deg)))
        isa_dev_per_level = temp_leg - isa_t_per_level[None, :]

        wind_at_best = np.take_along_axis(along_per_level, best_idx[:, None], axis=-1).squeeze(-1)
        isa_dev_at_best = np.take_along_axis(isa_dev_per_level, best_idx[:, None], axis=-1).squeeze(-1)
        temp_k_at_best = np.take_along_axis(temp_leg, best_idx[:, None], axis=-1).squeeze(-1)

        # Achieved Mach/TAS at the chosen level, and which of cruise_mach/
        # CAS/total-temp constrains it there. "ceiling" overrides that when
        # the chosen level is the highest one ceiling_ft(weight, isa_dev)
        # allows for this leg -- altitude-capped, not speed-capped (ceiling
        # never appears in mach_components/max_mach, which only cap speed
        # at a given level).
        mach_at_best = limits.max_mach(best_fl, temp_k_at_best, weight, cruise_mach)
        tas_ms_at_best = mach_at_best * speed_of_sound(temp_k_at_best)
        mach_limit_label = limits.binding_mach_limit(best_fl, temp_k_at_best, weight, cruise_mach)

        # ceiling is now (n_cand, 16) -- it depends on isa_dev, which varies
        # per level, not just on weight -- so "the top available level" is
        # found per candidate rather than read off a single shared array.
        ceiling = limits.ceiling_ft(weight[:, None], isa_dev_per_level)
        not_above_ceiling = TARGET_FL[None, :] * 100.0 <= ceiling
        has_any_level = not_above_ceiling.any(axis=-1)
        from_right = np.argmax(not_above_ceiling[:, ::-1], axis=-1)
        top_available_idx = np.where(
            has_any_level, not_above_ceiling.shape[-1] - 1 - from_right, -1
        )
        ceiling_bound = best_idx == top_available_idx
        binding_at_best = np.where(ceiling_bound, "ceiling", mach_limit_label)

        leg_s = (this_eff_dist_nm * NM_TO_M) / best_gs_ms
        accumulated_s = accumulated_s + leg_s

        # Fuel burn over this leg, at the weight/ISA-deviation used to fly
        # it -- weight is now a state variable, not a schedule keyed on
        # cum_nm (see _climb_profile).
        fuel_total_kgh = fuel_total_kgh_table(weight, isa_dev_at_best)
        weight = weight - fuel_total_kgh * (leg_s / 3600.0) / 1000.0

        chosen_fl[:, i] = best_fl
        best_idx_out[:, i] = best_idx
        mach[:, i] = mach_at_best
        tas_kt[:, i] = tas_ms_at_best / KT_TO_MS
        gs_kt[:, i] = best_gs_ms / KT_TO_MS
        wind_kt[:, i] = wind_at_best / KT_TO_MS
        temp_c[:, i] = temp_k_at_best - 273.15
        isa_dev_k[:, i] = isa_dev_at_best
        leg_time_s[:, i] = leg_s
        eff_dist_nm[:, i] = this_eff_dist_nm
        elapsed_s[:, i] = accumulated_s
        binding[:, i] = binding_at_best

    legs_out = dict(
        chosen_fl=chosen_fl, best_idx=best_idx_out, mach=mach, tas_kt=tas_kt,
        gs_kt=gs_kt, wind_kt=wind_kt, temp_c=temp_c, isa_dev_k=isa_dev_k,
        leg_time_s=leg_time_s, eff_dist_nm=eff_dist_nm, elapsed_s=elapsed_s,
        binding=binding, accumulated_s=accumulated_s, weight_at_barix=weight,
    )
    return legs_out, weight_per_leg, climb


def _build_arrival_wind_fn(subsonic_data, arrival_upper_data, dep_i8, arrival_legs):
    """A wind_at_fl_builder for arrival.arrival(): a function of accumulated_s
    (n_cand, seconds since brake release, i.e. march_legs' own elapsed time
    at the decel point) returning the wind_at_fl callable arrival() needs.

    Sampled at a single representative arrival leg -- the one nearest the
    midpoint of the post-BARIX span, positionally indexed (both subsonic_data
    and arrival_upper_data's leg axes are arrival_legs itself -- era5.
    reduce_to_legs run against exactly that list, once per level set -- see
    run_search) -- the same single-point proxy convention _climb_conditions
    uses for the climb, at BARIX clock time (dep_i8 + accumulated_s);
    arrival.py's own segment times determine how much later touchdown
    actually is, so this is a coarse snapshot, not a per-segment march,
    consistent with the climb model's own accuracy.

    STITCHED PROFILE (B6): subsonic_data alone (era5.SUBSONIC_LEVELS,
    175-500 hPa, FL183-FL414) does not reach the decel segment's own
    midpoint -- (cruise_fl + decel_end_fl) / 2, roughly FL446-491 from a
    realistic FL580-600 cruise -- so that segment, the largest single piece
    of the ~307 nm arrival, used to be silently corrected with FL414 wind
    every time. arrival_upper_data (era5.UPPER_AIR_LEVELS, 70-150 hPa,
    FL447-FL605 -- the SAME netCDFs already downloaded and reduced onto the
    cruise legs, just reduced a second time onto arrival_legs; no new CDS
    download) is concatenated onto subsonic_data's level axis before the
    log(pressure) interpolation below -- 150 and 175 hPa are adjacent, so
    together they form one continuous ~FL183-FL605 profile with no gap.
    Both npz's must share the same time axis (era5.UPPER_AIR_TIMES, since
    both are reduced from requests built from it) -- asserted, not assumed,
    since a silent misalignment there would be exactly this bug's twin. Both
    must also have been reduced onto arrival_legs itself, not the cruise
    legs -- also asserted (B8), via each npz's own stored cum_nm.

    Vertical interpolation is still linear in log(pressure), clamped to the
    stitched span (no extrapolation, same convention as every other table
    lookup here) -- fl_min/fl_max are now about FL183/FL605. The clamp
    still bites at the BOTTOM: the descent segment's own midpoint,
    (decel_end_fl + 15) / 2, reaches FL163.5 on the fastest (380 kt)
    schedule, FL182.5 on 350 kt (marginal), and FL199 on 325 kt (clear) --
    faster schedules descend from a lower decel_end_fl, so THEY are the
    ones that clamp, not the slower ones. Accepted, not chased further (see
    arrival.py's B6 task notes): that segment is ~60 nm of ~7.7 min, and
    low-level wind is weak, so a clamped read there is worth well under
    0.3 min -- adding 600/700 hPa would cost another CDS download for less
    than wind_at_fl's own fl_clamped flag (below) now already tells us.
    Every wind_at_fl call reports, per candidate, whether ITS OWN request
    needed that clamp -- arrival.py ORs the three per-segment reads into
    wind_fl_clamped, so a clamped number is flagged rather than silently
    plausible, which is the point of this whole task."""
    if not np.array_equal(subsonic_data["time"], arrival_upper_data["time"]):
        raise ValueError(
            "subsonic_data and arrival_upper_data have different time axes -- "
            "both must come from era5.reduce_to_legs runs built from "
            "era5.UPPER_AIR_TIMES against the SAME arrival_legs, or the two "
            "stitched level sets would silently misalign in time"
        )

    # B8: the time axes above were checked, but the LEG axis never was --
    # mid_local_idx below is a bare positional index into subsonic_data's/
    # arrival_upper_data's own "leg" dimension, valid as long as (and only
    # as long as) each npz was built by era5.reduce_to_legs run against
    # exactly arrival_legs. Both npz's are cut from the same era5_upper_*.nc
    # files as the cruise --npz, so passing a cruise-leg npz here (e.g. the
    # same file already used for --npz) is the natural mistake -- it passes
    # the time check above (both share UPPER_AIR_TIMES) and then silently
    # reads mid-Atlantic wind for the arrival segment, since a small index
    # like 2 is a perfectly valid position on a ~32-leg cruise axis too. An
    # operator error like this belongs at startup, not as a per-candidate
    # flag, so raise rather than proceed.
    expected_cum_nm = np.array([leg.cum_nm for leg in arrival_legs])
    for name, d in (("subsonic_data", subsonic_data),
                    ("arrival_upper_data", arrival_upper_data)):
        leg_cum_nm = np.asarray(d["cum_nm"])
        if (leg_cum_nm.shape != expected_cum_nm.shape
                or not np.allclose(leg_cum_nm, expected_cum_nm)):
            raise ValueError(
                f"{name} was not reduced onto arrival_legs -- it has "
                f"{leg_cum_nm.shape[0]} leg(s), but arrival_legs has "
                f"{len(arrival_legs)}; era5.reduce_to_legs must be run "
                "against the SAME post-BARIX arrival_legs list used here, "
                "not the cruise legs (--npz)"
            )

    mid_local_idx = len(arrival_legs) // 2
    mid_leg = arrival_legs[mid_local_idx]
    track_rad = np.radians(mid_leg.track_deg)

    times_i8 = subsonic_data["time"].astype("datetime64[ns]").astype("int64")

    u_col = np.concatenate(
        [subsonic_data["u"][:, :, mid_local_idx], arrival_upper_data["u"][:, :, mid_local_idx]],
        axis=1,
    )  # (n_time, n_level)
    v_col = np.concatenate(
        [subsonic_data["v"][:, :, mid_local_idx], arrival_upper_data["v"][:, :, mid_local_idx]],
        axis=1,
    )
    t_col = np.concatenate(
        [subsonic_data["t"][:, :, mid_local_idx], arrival_upper_data["t"][:, :, mid_local_idx]],
        axis=1,
    )

    src_p_hpa = np.concatenate(
        [subsonic_data["level"], arrival_upper_data["level"]]
    ).astype(float)
    order = np.argsort(src_p_hpa)  # ascending pressure -- descending FL
    src_log_p = np.log(src_p_hpa[order] * 100.0)
    fl_min = float(pressure_to_fl(src_p_hpa[order][-1] * 100.0))
    fl_max = float(pressure_to_fl(src_p_hpa[order][0] * 100.0))

    def wind_at_fl_builder(accumulated_s):
        at_i8 = dep_i8 + (np.asarray(accumulated_s, dtype=float) * 1e9).astype("int64")

        idx1 = np.searchsorted(times_i8, at_i8, side="right") - 1
        idx1 = np.clip(idx1, 0, len(times_i8) - 2)
        idx2 = idx1 + 1
        t0, t1 = times_i8[idx1], times_i8[idx2]
        frac = np.clip((at_i8 - t0) / (t1 - t0), 0.0, 1.0)[:, None]

        u_at = u_col[idx1] + frac * (u_col[idx2] - u_col[idx1])  # (n_cand, n_level)
        v_at = v_col[idx1] + frac * (v_col[idx2] - v_col[idx1])
        t_at = t_col[idx1] + frac * (t_col[idx2] - t_col[idx1])

        along_asc = (u_at * np.sin(track_rad) + v_at * np.cos(track_rad))[:, order]
        t_asc = t_at[:, order]

        def wind_at_fl(level_fl):
            level_fl = np.asarray(level_fl, dtype=float)
            fl_clamped = (level_fl < fl_min) | (level_fl > fl_max)
            level_fl = np.clip(level_fl, fl_min, fl_max)
            target_log_p = np.log(fl_to_pressure(level_fl))

            v_idx0 = np.searchsorted(src_log_p, target_log_p, side="right") - 1
            v_idx0 = np.clip(v_idx0, 0, len(src_log_p) - 2)
            v_idx1 = v_idx0 + 1
            v_frac = ((target_log_p - src_log_p[v_idx0])
                      / (src_log_p[v_idx1] - src_log_p[v_idx0]))

            def _at(col):
                lo = np.take_along_axis(col, v_idx0[:, None], axis=1).squeeze(1)
                hi = np.take_along_axis(col, v_idx1[:, None], axis=1).squeeze(1)
                return lo + v_frac * (hi - lo)

            return {"wind_kt": _at(along_asc) / KT_TO_MS, "temp_k": _at(t_asc),
                    "fl_clamped": fl_clamped}

        return wind_at_fl

    return wind_at_fl_builder


def resolve_tow_and_arrival(
    cc_legs, cc_idx, arrival_legs, arrival_nm, data, subsonic_data, dep_i8,
    tow_t=None, zfw_t=None, min_landing_fuel_t=fuel.MIN_LANDING_FUEL_T,
    decel_descent_min=None, cruise_mach=limits.CRUISE_MACH,
    arrival_upper_data=None,
):
    """Shared by run_search/run_report/verify.run_verify: solves TOW (fixed
    point on zfw_t, or the flat tow_t override) and the arrival segment
    together -- arrival fuel has to be inside the fixed point (see fuel.py),
    so the two can't be solved separately.

    decel_descent_min given forces arrival.flat_arrival -- the pre-B3 flat
    (DECEL_DESCENT_S, DESCENT_FUEL_T) pair -- instead of the real per-day
    arrival.arrival() model, for comparing old and new numbers; subsonic_data
    and arrival_upper_data are unused in that case and may both be None.
    decel_descent_min None (the default) requires BOTH subsonic_data (era5.
    reduce_to_legs run against arrival_legs, the post-BARIX legs -- see
    run_search) and arrival_upper_data (era5.reduce_to_legs run against the
    SAME arrival_legs, but from the UPPER_AIR_LEVELS netCDFs already
    downloaded for the cruise legs -- --arrival-upper-npz) -- B6:
    subsonic_data alone (FL183-FL414) doesn't reach the decel segment's own
    wind-sampling midpoint from a realistic cruise level, so both are
    stitched together in _build_arrival_wind_fn.

    Returns (tow_t, n_iterations, fuel_flags, legs_out, weight_per_leg,
    climb, arrival_out) -- n_iterations is None for a plain --tow run, same
    convention as fuel.fuel_plan."""
    n_cand = len(dep_i8)

    if decel_descent_min is not None:
        arrival_fn = functools.partial(arrival.flat_arrival,
                                        decel_descent_min=decel_descent_min)
        wind_fn_builder = lambda accumulated_s: None  # noqa: E731 -- never called by flat_arrival
    else:
        if subsonic_data is None or arrival_upper_data is None:
            raise ValueError(
                "subsonic_data (--subsonic-npz) AND arrival_upper_data "
                "(--arrival-upper-npz) are both required to compute the real "
                "arrival model -- subsonic_data alone doesn't reach the decel "
                "segment's own wind-sampling midpoint (B6); pass "
                "--decel-descent-min to force the flat legacy arrival instead"
            )
        wind_fn_builder = _build_arrival_wind_fn(
            subsonic_data, arrival_upper_data, dep_i8, arrival_legs
        )
        arrival_fn = arrival.arrival

    if tow_t is None and zfw_t is None:
        tow_t = DEFAULT_TOW_T  # neither given -- pre-fuel-plan flat default

    if tow_t is None:
        zfw_arr = np.full(n_cand, zfw_t, dtype=float)
        tow_out, n_iterations, fuel_flags, legs_out, weight_per_leg, climb, arrival_out = (
            fuel.fixed_point_fuel_iteration(
                cc_legs, cc_idx, data, dep_i8, zfw_arr, arrival_nm, wind_fn_builder,
                min_landing_fuel_t=min_landing_fuel_t,
                march_legs_fn=march_legs, arrival_fn=arrival_fn, cruise_mach=cruise_mach,
            )
        )
    else:
        legs_out, weight_per_leg, climb = march_legs(cc_legs, cc_idx, data, dep_i8,
                                                        tow_t, cruise_mach)
        arrival_out = fuel._arrival_from_march(legs_out, arrival_nm, wind_fn_builder, arrival_fn)
        tow_out = np.broadcast_to(np.asarray(tow_t, dtype=float), (n_cand,)).copy()
        n_iterations = None
        fuel_flags = np.array([""] * n_cand, dtype=object)

    return tow_out, n_iterations, fuel_flags, legs_out, weight_per_leg, climb, arrival_out


def run_search(pln_path, npz_path, surface_npz_path, decel_id="BARIX",
                top=50, out_path="results.csv", out_all_path=None,
                tow_t=None, zfw_t=None,
                min_landing_fuel_t=fuel.MIN_LANDING_FUEL_T,
                subsonic_npz_path=None, decel_descent_min=None,
                cruise_mach=limits.CRUISE_MACH,
                arrival_upper_npz_path=None):
    """Builds legs from pln_path (same max_leg_nm default as
    `route`/reduce_to_legs, so the leg axis lines up with npz_path's), takes
    the climb+cruise span (brake release through decel_id -- see
    route.climb_cruise_segment), then calls march_legs across every
    candidate departure and collapses the per-leg arrays to means (weighted
    by eff_dist_nm, so the climb's leading zero/partial-distance legs don't
    pollute the cruise-only means with climb-altitude noise). Screens each
    candidate's KJFK departure and EGLL arrival wind (surface_npz_path,
    from era5.reduce_surface_to_npz) against runways.py's runway geometry,
    and ranks on total block time (brakes-release to landing, including
    runway penalties) rather than supersonic-segment time alone. Prints
    the top 10 rows and eight sanity checks (computed on the same
    filtered/sorted rows as the output table), writes the top `top` rows
    to out_path, and returns the full ranked DataFrame (no NaN rows).

    With out_all_path given, also writes the full ranked DataFrame (every
    valid candidate, ~31,000 rows, raw numeric columns rather than --out's
    rounded/formatted display strings) there -- for nb/day-search-results.ipynb,
    which needs the raw distribution rather than just the top rows.

    zfw_t (tonnes) drives fuel.fixed_point_fuel_iteration (via
    resolve_tow_and_arrival) across the WHOLE candidate vector at once, same
    convention as report.run_report's single-candidate use of the same
    fixed point: TOW is solved per candidate rather than assumed, so a
    warm/heavy day's own extra climb AND arrival fuel feeds back into its
    own extra weight rather than every candidate being flown at one shared
    guess. tow_t overrides zfw_t and skips the solve entirely, applying that
    one weight to every candidate -- for "what if the whole fleet loads X"
    or for matching an old run. Neither given falls back to the flat
    DEFAULT_TOW_T, same pre-fuel-plan behaviour as before this was wired in.
    The fixed point's own boundary/convergence flags (fuel.py's
    tow_above_mtow_*/tow_below_climb_table_*/fuel_not_converged) join the
    jfk/lhr/climb/arrival flags already in the output's flags column.

    subsonic_npz_path (era5.reduce_to_legs run against the post-BARIX legs,
    the route's complement of climb_cruise_segment) and arrival_upper_npz_path
    (era5.reduce_to_legs run against those SAME legs, but from the
    UPPER_AIR_LEVELS netCDFs already downloaded for the cruise legs -- B6, no
    new CDS download) together drive arrival.arrival()'s per-day decel/level/
    descent model via _build_arrival_wind_fn's stitched FL183-FL605 wind
    profile, replacing the old flat DESCENT_FUEL_T placeholder; both required
    unless decel_descent_min forces the flat legacy arrival instead, for
    comparing old and new numbers directly."""
    plan = parse_pln(pln_path)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs, decel_id=decel_id)
    cc_idx = np.flatnonzero(mask)
    cc_legs = [legs[i] for i in cc_idx]
    arrival_idx = np.flatnonzero(~mask)
    arrival_legs = [legs[i] for i in arrival_idx]
    arrival_nm = legs[-1].cum_nm - cc_legs[-1].cum_nm

    data = load_legs_npz(npz_path)
    surface_data = load_surface_npz(surface_npz_path)
    subsonic_data = load_legs_npz(subsonic_npz_path) if subsonic_npz_path is not None else None
    arrival_upper_data = (
        load_legs_npz(arrival_upper_npz_path) if arrival_upper_npz_path is not None else None
    )
    candidates = candidate_departures()
    n_cand = len(candidates)
    dep_i8 = candidates["departure_utc"].values.astype("datetime64[ns]").astype("int64")

    tow_t, _n_iterations, fuel_flags, legs_out, _weight_per_leg, climb, arrival_out = (
        resolve_tow_and_arrival(
            cc_legs, cc_idx, arrival_legs, arrival_nm, data, subsonic_data, dep_i8,
            tow_t=tow_t, zfw_t=zfw_t, min_landing_fuel_t=min_landing_fuel_t,
            decel_descent_min=decel_descent_min, cruise_mach=cruise_mach,
            arrival_upper_data=arrival_upper_data,
        )
    )
    chosen_fl = legs_out["chosen_fl"]
    wind_kt = legs_out["wind_kt"]
    isa_dev_k = legs_out["isa_dev_k"]
    gs_kt = legs_out["gs_kt"]
    eff_dist_nm = legs_out["eff_dist_nm"]
    cruise_weight = eff_dist_nm.sum(axis=1)

    candidates["supersonic_time_s"] = legs_out["accumulated_s"]
    candidates["tow_t"] = tow_t
    candidates["climb_time_s"] = climb["time_min"] * 60.0
    candidates["climb_temp_band"] = climb["temp_band"]
    candidates["climb_warm_clamped"] = climb["warm_flag"]
    candidates["mean_fl"] = (chosen_fl * eff_dist_nm).sum(axis=1) / cruise_weight
    candidates["mean_wind_kt"] = (wind_kt * eff_dist_nm).sum(axis=1) / cruise_weight
    candidates["mean_isa_dev_k"] = (isa_dev_k * eff_dist_nm).sum(axis=1) / cruise_weight
    candidates["weight_at_barix_t"] = legs_out["weight_at_barix"]
    candidates["arrival_time_s"] = arrival_out["time_min"] * 60.0
    candidates["arrival_fuel_t"] = arrival_out["fuel_t"]

    # Touchdown clock time = departure + accumulated_s (already climb +
    # cruise, see march_legs) + arrival_time_s (arrival.arrival()'s own
    # per-candidate decel+level+descent+approach time, or the flat
    # decel_descent_min override -- see resolve_tow_and_arrival). NaN
    # accumulated_s (a candidate the march couldn't complete) produces a
    # garbage touchdown_i8 here -- harmless, since that row is dropped by
    # the dropna below same as everywhere else.
    touchdown_i8 = dep_i8 + (
        (legs_out["accumulated_s"] + candidates["arrival_time_s"].to_numpy()) * 1e9
    ).astype("int64")

    jfk = runways.runway_screen(surface_data, "KJFK", dep_i8)
    lhr = runways.runway_screen(surface_data, "EGLL", touchdown_i8)

    candidates["jfk_runway"] = jfk["runway"]
    candidates["lhr_runway"] = lhr["runway"]
    candidates["jfk_xwind_gust_kt"] = jfk["xwind_gust_kt"]
    candidates["lhr_xwind_gust_kt"] = lhr["xwind_gust_kt"]
    candidates["jfk_flag"] = jfk["flag"]
    candidates["lhr_flag"] = lhr["flag"]
    candidates["jfk_penalty_s"] = jfk["penalty_s"]
    candidates["lhr_penalty_s"] = lhr["penalty_s"]
    candidates["flags"] = [
        ",".join(part for part in (
            (f"jfk_{jfk_flag}" if jfk_flag else ""),
            (f"lhr_{lhr_flag}" if lhr_flag else ""),
            ("climb_warm_clamped" if warm else ""),
            fuel_flag,  # already self-describing (fuel.py's own flag strings), no prefix needed
            arrival_flag,  # already self-describing (arrival.py's own flag strings)
        ) if part)
        for jfk_flag, lhr_flag, warm, fuel_flag, arrival_flag in
        zip(jfk["flag"], lhr["flag"], climb["warm_flag"], fuel_flags, arrival_out["flags"])
    ]
    candidates["total_time_s"] = (legs_out["accumulated_s"] + candidates["arrival_time_s"]
                                    + jfk["penalty_s"] + lhr["penalty_s"])

    # Same filter the output table gets, captured here so the sanity checks
    # below are computed on exactly those rows, not the unfiltered arrays.
    # Also restricted to cruise-only leg-instances (eff_dist_nm > 0) --
    # climb-consumed legs still have a (meaningless, climb-altitude) chosen
    # FL/ground speed computed for them, since best_level runs regardless.
    valid_mask = candidates[["supersonic_time_s", "mean_fl"]].notna().all(axis=1).to_numpy()
    cruise_only = eff_dist_nm[valid_mask] > 0.0
    chosen_fl_valid = chosen_fl[valid_mask][cruise_only]
    gs_kt_valid = gs_kt[valid_mask][cruise_only]

    candidates = candidates.dropna(subset=["supersonic_time_s", "mean_fl"])
    candidates = candidates.sort_values("total_time_s", ascending=True).reset_index(drop=True)

    display = pd.DataFrame({
        "date": candidates["local_date"],
        "local_departure": candidates["local_hour"].map(lambda h: f"{h:02d}:00"),
        "supersonic_time": candidates["supersonic_time_s"].map(_format_hmm),
        "mean_fl": candidates["mean_fl"].round(0).astype(int),
        "mean_wind_kt": candidates["mean_wind_kt"].round(1),
        "mean_isa_dev_k": candidates["mean_isa_dev_k"].round(1),
        "weight_at_barix_t": candidates["weight_at_barix_t"].round(1),
        "jfk_runway": candidates["jfk_runway"],
        "lhr_runway": candidates["lhr_runway"],
        "jfk_xwind_gust_kt": candidates["jfk_xwind_gust_kt"].round(1),
        "lhr_xwind_gust_kt": candidates["lhr_xwind_gust_kt"].round(1),
        "flags": candidates["flags"],
        "total_time": candidates["total_time_s"].map(_format_hmm),
        "tow_t": candidates["tow_t"],
    })

    print(display.head(10).to_string(index=False))

    cc_total_nm = cc_legs[-1].cum_nm
    best_row = candidates.iloc[0]
    worst_row = candidates.iloc[-1]
    spread_min = (worst_row["total_time_s"] - best_row["total_time_s"]) / 60.0
    top_fl = TARGET_FL.max()
    frac_at_top = float(np.mean(chosen_fl_valid == top_fl))

    is_winter = best_row["local_date"].month in (11, 12, 1, 2)
    print("\nSanity checks:")
    print(f"1. Best day (total time): {best_row['local_date']} "
          f"({'winter' if is_winter else 'NOT WINTER -- check wind sign'})")
    print(f"2. Spread best-worst (total time): {spread_min:.1f} min over {cc_total_nm:.0f} nm "
          f"(brake release to {decel_id}; expect a wider spread than the old fixed-climb model, "
          f"since climb.csv adds up to ~18 min of temperature-driven variation)")
    print(f"3. Mean chosen FL overall: {chosen_fl_valid.mean():.0f} "
          f"(top available FL {top_fl:.0f}); "
          f"at top level {frac_at_top * 100:.0f}% of leg-instances")
    print(f"4. Chosen ground speed range: {gs_kt_valid.min():.0f}-{gs_kt_valid.max():.0f} kt "
          f"(expect 800-1400 kt); NaN rows dropped: {n_cand - len(candidates)}")

    n_valid = len(candidates)
    jfk_unflyable_n = int((candidates["jfk_flag"] == "unflyable").sum())
    lhr_unflyable_n = int((candidates["lhr_flag"] == "unflyable").sum())
    jfk_flagged_n = int((candidates["jfk_flag"] == "xwind_25_30").sum())
    lhr_flagged_n = int((candidates["lhr_flag"] == "xwind_25_30").sum())
    lhr_westerly_n = int((candidates["lhr_runway"] == "27R/27L").sum())
    lhr_easterly_n = int((candidates["lhr_runway"] == "09L/09R").sum())

    print(f"5. Unflyable: JFK {jfk_unflyable_n} ({jfk_unflyable_n / n_valid * 100:.1f}%), "
          f"LHR {lhr_unflyable_n} ({lhr_unflyable_n / n_valid * 100:.1f}%) of {n_valid} "
          f"(expect a small percentage; LHR can only reject above {runways.XWIND_FLAG_KT:.0f} kt "
          f"crosswind gust since its runways are parallel, while JFK's two candidates are 90 deg "
          f"apart and leave a gap at easterly winds)")
    print(f"6. LHR ops: westerly (27R/27L) {lhr_westerly_n} "
          f"({lhr_westerly_n / n_valid * 100:.0f}%), easterly (09L/09R) {lhr_easterly_n} "
          f"({lhr_easterly_n / n_valid * 100:.0f}%) (expect westerlies to dominate; the minority "
          f"easterly days skip LHR's 5 min penalty and float up the total_time ranking)")
    print(f"7. Flagged {runways.XWIND_OK_KT:.0f}-{runways.XWIND_FLAG_KT:.0f} kt crosswind-gust "
          f"band: JFK {jfk_flagged_n} ({jfk_flagged_n / n_valid * 100:.1f}%), "
          f"LHR {lhr_flagged_n} ({lhr_flagged_n / n_valid * 100:.1f}%) of {n_valid}")

    best_total_idx = candidates["total_time_s"].idxmin()
    best_ss_idx = candidates["supersonic_time_s"].idxmin()
    best_total_row = candidates.loc[best_total_idx]
    best_ss_row = candidates.loc[best_ss_idx]
    same_day = bool(best_total_idx == best_ss_idx)
    penalty_min = (best_total_row["jfk_penalty_s"] + best_total_row["lhr_penalty_s"]) / 60.0
    arrival_min = best_total_row["arrival_time_s"] / 60.0
    expect_min = arrival_min + penalty_min
    diff_min = (best_total_row["total_time_s"] - best_ss_row["supersonic_time_s"]) / 60.0
    print(f"8. Best total time {_format_hmm(best_total_row['total_time_s'])} vs best "
          f"supersonic time {_format_hmm(best_ss_row['supersonic_time_s'])} "
          f"({'same day' if same_day else 'DIFFERENT day -- penalty reshuffled the ranking'}): "
          f"diff {diff_min:.1f} min (expect arrival {arrival_min:.1f} + penalties "
          f"{penalty_min:.0f} = {expect_min:.1f} min when same day -- supersonic_time_s is "
          f"brake-release-to-{decel_id}, climb included, so no separate accel term here any more; "
          f"arrival is now this candidate's own model output, not a flat constant)")

    out_path = Path(out_path)
    display.head(top).to_csv(out_path, index=False)
    print(f"\nWrote top {min(top, len(display))} rows to {out_path}")

    if out_all_path is not None:
        out_all_path = Path(out_all_path)
        candidates.to_csv(out_all_path, index=False)
        print(f"Wrote all {len(candidates)} ranked candidates to {out_all_path}")

    return candidates


def run_shortlist(search_csv_path, pln_path, npz_path, top=10, decel_id="BARIX"):
    """The top `top` rows of a concopt search --out CSV, printed as a
    ready-to-run `concopt verify` command per day -- so working through a
    shortlist by hand (load the date in Active Sky, run verify, repeat)
    doesn't mean re-typing date/hour/tow into the CLI each time. tow_t
    comes from the CSV's own tow_t column (run_search stamps every row
    with the --tow it was actually run under), so the generated command
    always matches -- verify.py's whole comparison depends on that."""
    df = pd.read_csv(search_csv_path).head(top)

    print(f"Top {len(df)} shortlist from {search_csv_path}, ready to verify by hand:\n")
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        local_date = dt.date.fromisoformat(str(row["date"]))
        local_hour = int(str(row["local_departure"]).split(":")[0])
        tow_t = float(row["tow_t"])
        flags = row["flags"] if pd.notna(row["flags"]) and row["flags"] else "-"

        print(f"{rank:>2}. {local_date} {local_hour:02d}:00  total {row['total_time']}  "
              f"mean_fl {row['mean_fl']}  wind {row['mean_wind_kt']:+.1f} kt  flags: {flags}")
        print(f"    concopt verify --pln {pln_path} --npz {npz_path} "
              f"--date {local_date} --hour {local_hour} --tow {tow_t:.0f} "
              f"--decel {decel_id}\n")

    return df
