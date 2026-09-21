"""concopt report -- the full per-leg breakdown for ONE candidate departure,
as opposed to search.py's means-only scan across ~31,000 of them. Reuses
search.march_legs for the marching; this module only aggregates its output
into a waypoint table, a profile summary, and brakes-release-to-touchdown
totals, then prints and writes them.
"""
import datetime as dt
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from concopt import arrival, fuel, limits, runways
from concopt.atmos import (KT_TO_MS, cas_from_mach, isa, mach_from_cas, mach_from_total_temp,
                           speed_of_sound)
from concopt.data.conc_data import CLIMB_LEVELS_FL, cas_limit_kt, climb_to
from concopt.era5 import load_legs_npz, load_surface_npz
from concopt.route import build_legs, climb_cruise_segment, parse_pln, position_at_cum_nm
from concopt.params import (ACCEL_WAYPOINT_ID, APPROACH_FUEL_T, APPROACH_MIN, APPROACH_NM,
                            ARRIVAL_PROFILE_STEPS, C_TO_K, DECEL_WAYPOINT_ID, DESCENT_END_FT,
                            FT_TO_M, LEVEL_MACH, NS_PER_S, S_PER_HOUR, SUBSONIC_LIMIT_MACH,
                            TOTAL_TEMP_MAX_K)
from concopt.search import (NY_TZ, TOP_OF_CLIMB_FL,
                             _format_hmm, local_to_departure_utc, march_legs,
                             resolve_tow_and_arrival)

# What each fuel.py boundary flag means, spelled out rather than left as the
# bare token the search CSV carries -- `concopt report` is read by a person
# deciding whether to fly the day, and a clamped or unconverged fuel plan is
# not a plan.
_FUEL_FLAG_NOTES = {
    f"tow_above_mtow_{fuel.MTOW_T:.0f}":
        "the uplift this ZFW needs puts TOW over the {mtow:.0f} t structural "
        "limit -- INFEASIBLE; TOW below is clamped, the plan wanted {req:.1f} t",
    f"tow_below_climb_table_{fuel.CLIMB_TOW_MIN_T:.0f}":
        "TOW fell below conc_climb.csv's {floor:.0f} t floor and was clamped "
        "(the plan wanted {req:.1f} t) -- lower-weight climb pages needed",
    "tow_below_required":
        "the TOW override loads less fuel than this trip needs "
        "(ZFW + trip fuel + reserve = {req:.1f} t) -- lands below the reserve",
    "fuel_not_converged":
        "the fixed point was still moving after {iters} iterations -- the "
        "weights below are the last iterate, not a solution",
}


def _format_mmss(seconds):
    """Seconds -> 'm:ss' string, for the (short) per-leg times."""
    total_s = int(round(seconds))
    m, s = divmod(total_s, 60)
    return f"{m}:{s:02d}"


def best_candidate_from_csv(search_csv_path):
    """The fastest candidate (top row) of a concopt search --out CSV, as
    (local_date, local_hour) -- what --best feeds run_report."""
    df = pd.read_csv(search_csv_path)
    row = df.iloc[0]
    local_date = dt.date.fromisoformat(str(row["date"]))
    local_hour = int(str(row["local_departure"]).split(":")[0])
    return local_date, local_hour


def _weighted_mean(values, weights):
    """np.average, but 0.0 total weight (every sub-leg in the group was
    wholly inside the climb -- see march_legs' eff_dist_nm) returns NaN
    instead of raising."""
    total_w = weights.sum()
    return float(np.average(values, weights=weights)) if total_w > 0 else np.nan


def _waypoint_table(cc_legs, departure_utc_ts, leg):
    """One row per original .pln waypoint spanned by cc_legs (brake release
    through decel point), aggregating each parent leg's subdivided sub-legs
    back up (build_legs gives every sub-leg of a parent the same
    from_id/to_id and equal length, so contiguous runs of equal (from_id,
    to_id) are exactly one parent leg). Weighted by eff_dist_nm rather than
    a plain mean: for a leg wholly inside the climb every sub-leg's weight
    is 0 (chosen_fl/wind/etc there are climb-altitude noise -- best_level
    still runs, but nothing was actually flown at it), for one wholly past
    it every weight equals its own length (an ordinary distance-weighted
    mean), and for the one leg straddling top of climb only its
    post-climb portion counts. `leg` is the per-sub-leg dict march_legs
    returned, already squeezed to this one candidate (1-D arrays, length
    len(cc_legs)).

    A named waypoint that falls entirely inside the climb (there's no
    profile data below TOP_OF_CLIMB_FL to interpolate a real elapsed
    time/level at it -- conc_climb.csv only gives the top-of-climb state)
    shows elapsed_s pinned to the climb's own total time and NaN for
    everything cruise-specific, with binding "climb" -- see run_report's
    separate climb-segment printout for where top of climb actually falls."""
    origin_id = cc_legs[0].from_id
    origin_cum_nm = cc_legs[0].cum_nm - cc_legs[0].dist_nm

    rows = [{
        "waypoint": origin_id,
        "cum_nm": origin_cum_nm,
        "elapsed_s": 0.0,
        "clock_utc": departure_utc_ts,
        "leg_time_s": np.nan,
        "chosen_fl": np.nan,
        "mach": np.nan,
        "tas_kt": np.nan,
        "gs_kt": np.nan,
        "wind_kt": np.nan,
        "temp_c": np.nan,
        "isa_dev_k": np.nan,
        "binding": "",
    }]

    for (_, to_id), members in groupby(range(len(cc_legs)),
                                               key=lambda i: (cc_legs[i].from_id, cc_legs[i].to_id)):
        idx = np.array(list(members))
        last = idx[-1]
        w = leg["eff_dist_nm"][idx]
        rows.append({
            "waypoint": to_id,
            "cum_nm": cc_legs[last].cum_nm,
            "elapsed_s": leg["elapsed_s"][last],
            "clock_utc": departure_utc_ts + pd.Timedelta(seconds=float(leg["elapsed_s"][last])),
            "leg_time_s": leg["leg_time_s"][idx].sum(),
            "chosen_fl": _weighted_mean(leg["chosen_fl"][idx], w),
            "mach": _weighted_mean(leg["mach"][idx], w),
            "tas_kt": _weighted_mean(leg["tas_kt"][idx], w),
            "gs_kt": _weighted_mean(leg["gs_kt"][idx], w),
            "wind_kt": _weighted_mean(leg["wind_kt"][idx], w),
            "temp_c": _weighted_mean(leg["temp_c"][idx], w),
            "isa_dev_k": _weighted_mean(leg["isa_dev_k"][idx], w),
            "binding": (Counter(leg["binding"][idx]).most_common(1)[0][0]
                        if w.sum() > 0 else "climb"),
        })

    table = pd.DataFrame(rows)
    table["clock_local"] = table["clock_utc"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)
    return table


def _fuel_plan_lines(plan):
    """fuel.fuel_plan's dict (n_cand=1) -> the printed fuel-plan block, as a
    list of lines. Trip fuel is broken out into climb / cruise / arrival
    rather than shown as one number: the split is what makes it obvious when
    the climb is eating the flight, which is the whole reason the fixed point
    matters. arrival is arrival.arrival()'s own total -- see the Arrival
    block below this one for its own decel/level/descent/approach split, the
    reason the one placeholder number in this model (subsonic cruise fuel)
    stays footnoted here. Any boundary flag fuel.py raised is reported here
    too -- a clamped or unconverged result has to say so in the report, not
    only in the search CSV."""
    one = {k: (float(v[0]) if isinstance(v, np.ndarray) and k != "flags" else v)
           for k, v in plan.items()}
    flag = plan["flags"][0]

    lines = [
        f"Fuel plan (ZFW {one['zfw_t']:.1f} t, reserve "
        f"{one['min_landing_fuel_t']:.1f} t at touchdown)",
        f"  {'uplift':<15} {one['uplift_t']:.1f} t",
        f"  {'trip fuel':<15} {one['trip_fuel_t']:.1f} t      "
        f"climb {one['climb_fuel_t']:.1f} / cruise {one['cruise_fuel_t']:.1f} "
        f"/ arrival {one['arrival_fuel_t']:.1f}*",
        f"  {'take-off weight':<15} {one['tow_t']:.1f} t",
        f"  {'landing weight':<15} {one['landing_weight_t']:.1f} t",
    ]

    if plan["n_iterations"] is None:
        loaded = one["tow_t"] - one["zfw_t"]
        margin = one["tow_t"] - one["tow_required_t"]
        lines.append(f"  TOW override, no fixed point -- fuel loaded {loaded:.1f} t, "
                     f"trip needs {one['uplift_t']:.1f} t ({margin:+.1f} t margin)")
    else:
        lines.append(f"  converged in {plan['n_iterations']} iterations")

    if flag:
        note = _FUEL_FLAG_NOTES[flag].format(
            mtow=fuel.MTOW_T, floor=fuel.CLIMB_TOW_MIN_T,
            req=one["tow_required_t"], iters=plan["n_iterations"],
        )
        lines.append(f"  !! {flag}: {note}")

    lines.append("  * see the Arrival breakdown below for the level-cruise split")
    return lines


def _arrival_lines(arrival_out, arrival_nm, cruise_fl):
    """arrival.arrival()'s dict (n_cand=1),
    arrival_nm, and the cruise FL flown into the decel point -> the printed
    Arrival block, as a list of lines. The per-segment split is the point:
    it is what makes it obvious when the level segment is dominating."""
    a = {k: (float(v[0]) if isinstance(v, np.ndarray) and k != "flags" else v)
         for k, v in arrival_out.items() if k != "by_schedule"}
    a["flags"] = arrival_out["flags"][0]
    schedule_kt = int(a["schedule_kt"])
    header = f"Arrival (BARIX -> touchdown, {arrival_nm:.0f} nm)"

    total_nm = a["decel_nm"] + a["level_nm"] + a["descent_nm"] + arrival.APPROACH_NM
    lines = [
        header,
        f"  {'decel to M1.0':<16}{a['decel_nm']:6.0f} nm  {a['decel_time_min']:5.1f} min  "
        f"{a['decel_fuel_t']:5.2f} t   FL{cruise_fl:.0f} -> FL{a['level_fl']:.0f}",
        f"  {'level at M0.95':<16}{a['level_nm']:6.0f} nm  {a['level_time_min']:5.1f} min  "
        f"{a['level_fuel_t']:5.2f} t*  FL{a['level_fl']:.0f}, {a['level_wind_kt']:+.0f} kt",
        f"  {'descent':<16}{a['descent_nm']:6.0f} nm  {a['descent_time_min']:5.1f} min  "
        f"{a['descent_fuel_t']:5.2f} t   FL{a['level_fl']:.0f} -> 1500 ft",
        f"  {'approach':<16}{arrival.APPROACH_NM:6.0f} nm  {arrival.APPROACH_MIN:5.1f} min  "
        f"{arrival.APPROACH_FUEL_T:5.2f} t",
        f"  {'total':<16}{total_nm:6.0f} nm  {a['time_min']:5.1f} min  "
        f"{a['fuel_t']:5.2f} t   ({schedule_kt} kt schedule)",
    ]
    if a["flags"]:
        lines.append(f"  !! {a['flags']}")
    lines.append("  * level-cruise fuel from conc_subsonic_cruise.csv at the mass "
                  "actually flown at level-off")
    return lines


def _runway_lines(jfk, lhr, departure_utc_ts, touchdown_utc_ts):
    """runways.runway_screen's dicts (n_cand=1) for KJFK at departure and
    EGLL at touchdown -> the printed Runways block, as a list of lines. Same
    screen search.run_search applies, so the penalty below is the one
    search.total_time already includes."""
    lines = ["Runways (ERA5 surface wind, same screen as concopt search):"]
    for airport, r, when in (("KJFK", jfk, departure_utc_ts), ("EGLL", lhr, touchdown_utc_ts)):
        lines.append(
            f"  {airport}  {r['runway'][0]:<8} headwind {float(r['headwind_kt'][0]):+4.0f} kt, "
            f"crosswind gust {float(r['xwind_gust_kt'][0]):3.0f} kt   "
            f"penalty {_format_hmm(float(r['penalty_s'][0]))}   ({when:%H:%M} UTC)")
        if r["flag"][0]:
            lines.append(f"  !! {airport} {r['flag'][0]}")
    return lines


def _step_climb_schedule(legs, chosen_fl):
    """[(cum_nm, FL), ...] at every sub-leg where the chosen level differs
    from the one before it -- sub-leg granularity, not waypoint-aggregated,
    so it shows the actual step climbs/descents within a leg too. legs is
    the cruise-only leg list (climb-consumed legs excluded by the caller --
    their chosen_fl is climb-altitude noise, not a real step)."""
    start_cum_nm0 = legs[0].cum_nm - legs[0].dist_nm
    schedule = [(start_cum_nm0, float(chosen_fl[0]))]
    for i in range(1, len(chosen_fl)):
        if chosen_fl[i] != chosen_fl[i - 1]:
            start_cum_nm = legs[i].cum_nm - legs[i].dist_nm
            schedule.append((start_cum_nm, float(chosen_fl[i])))
    return schedule


def run_report(pln_path, npz_path, local_date, local_hour,
                decel_id=DECEL_WAYPOINT_ID, out_path="report.csv",
                tow_t=None, zfw_t=None, surface_npz_path=None,
                min_landing_fuel_t=fuel.MIN_LANDING_FUEL_T,
                subsonic_npz_path=None, cruise_mach=limits.CRUISE_MACH,
                arrival_upper_npz_path=None):
    """The full breakdown for one candidate departure (local_date,
    local_hour, America/New_York). Reruns march_legs -- the same march
    run_search uses -- for this single candidate, then prints the fuel plan,
    the arrival breakdown, waypoint table, climb segment, profile summary
    and brakes-release-to-touchdown totals, and writes the waypoint table to
    out_path.

    zfw_t (tonnes) is the primary weight input: TOW is solved (via
    resolve_tow_and_arrival) rather than given, so the report shows the
    weight the day actually needs, arrival fuel included. zfw_t is required.
    tow_t is an optional override ("what if I actually load X", e.g. the
    sim's trip-calculator figure): the fixed point is skipped, ZFW stays as
    given, and the report shows fuel loaded (tow_t - zfw_t) against the
    fuel the trip needs.

    surface_npz_path (era5.reduce_surface_to_npz) is required: KJFK at
    departure and EGLL at touchdown go through the same runway screen
    run_search applies, so the runway penalty in the totals is the one
    search's total_time already includes and the two agree.

    subsonic_npz_path (era5.reduce_to_legs run against the post-BARIX legs)
    and arrival_upper_npz_path (era5.reduce_to_legs run against those SAME
    legs, from the UPPER_AIR_LEVELS netCDFs already downloaded for the
    cruise legs -- B6, no new CDS download) together drive the real
    arrival.arrival() model's stitched FL183-FL605 wind profile; both
    required."""
    if surface_npz_path is None:
        raise ValueError("surface_npz_path (--surface-npz) is required -- the runway "
                         "penalties are part of total block time, as in concopt search")
    parsed_pln = parse_pln(pln_path)
    legs = build_legs(parsed_pln["waypoints"])
    mask = climb_cruise_segment(legs, decel_id=decel_id)
    cc_idx = np.flatnonzero(mask)
    cc_legs = [legs[i] for i in cc_idx]
    arrival_idx = np.flatnonzero(~mask)
    arrival_legs = [legs[i] for i in arrival_idx]
    arrival_nm = legs[-1].cum_nm - cc_legs[-1].cum_nm

    data = load_legs_npz(npz_path)
    subsonic_data = load_legs_npz(subsonic_npz_path) if subsonic_npz_path is not None else None
    arrival_upper_data = (
        load_legs_npz(arrival_upper_npz_path) if arrival_upper_npz_path is not None else None
    )
    departure_utc = local_to_departure_utc(local_date, local_hour)
    departure_utc_ts = pd.Timestamp(departure_utc)
    dep_i8 = np.array([departure_utc_ts.value], dtype="int64")

    tow_arr, n_iterations, plan_flags, legs_out, weight_per_leg, climb, arrival_out = (
        resolve_tow_and_arrival(
            cc_legs, cc_idx, arrival_legs, arrival_nm, data, subsonic_data, dep_i8,
            tow_t=tow_t, zfw_t=zfw_t, min_landing_fuel_t=min_landing_fuel_t,
            cruise_mach=cruise_mach,
            arrival_upper_data=arrival_upper_data,
        )
    )
    tow_t = float(tow_arr[0])
    # zfw_t is always the given; under a --tow override (n_iterations None)
    # fuel_plan then reports tow_required_t (zfw + uplift) against the TOW
    # actually loaded.
    zfw_arr = np.array([zfw_t], dtype=float)
    plan = fuel.fuel_plan(climb, legs_out, arrival_out,
                           zfw_t=zfw_arr,
                           min_landing_fuel_t=min_landing_fuel_t,
                           tow_t=tow_arr, n_iterations=n_iterations,
                           flags=plan_flags)

    for line in _fuel_plan_lines(plan):
        print(line)
    print()

    cruise_fl_at_barix = float(legs_out["chosen_fl"][0, -1])
    # Computed here (rather than only at the Totals printout below) so it
    # can also ride along in the waypoint CSV -- inflight.py's --compare
    # reads it back (arrival_s, on the decel waypoint's own row) instead of
    # re-adding a flat constant on top of a plan that already accounts for
    # its own real per-day arrival.
    arrival_time_s = float(arrival_out["time_min"][0]) * 60.0
    for line in _arrival_lines(arrival_out, arrival_nm, cruise_fl_at_barix):
        print(line)
    print()

    # Squeeze the n_cand=1 axis -- everything below is per sub-leg (1-D).
    scalar_keys = ("accumulated_s", "weight_at_barix")
    leg = {k: v[0] for k, v in legs_out.items() if k not in scalar_keys}
    total_elapsed_s = float(legs_out["accumulated_s"][0])
    weight_at_barix_t = float(legs_out["weight_at_barix"][0])
    weight_per_leg = weight_per_leg[0]  # (n_legs,), weight at the start of each sub-leg
    climb_row = {k: (v[0] if k in ("temp_band", "warm_flag") else float(v[0]))
                 for k, v in climb.items()}
    climb_time_s = climb_row["time_min"] * 60.0
    climb_ground_nm = climb_row["ground_dist_nm"]

    waypoint_table = _waypoint_table(cc_legs, departure_utc_ts, leg)

    display = pd.DataFrame({
        "waypoint": waypoint_table["waypoint"],
        "cum_nm": waypoint_table["cum_nm"].round(1),
        "elapsed": waypoint_table["elapsed_s"].map(_format_hmm),
        "clock_utc": waypoint_table["clock_utc"].dt.strftime("%Y-%m-%d %H:%M"),
        "clock_local": waypoint_table["clock_local"].dt.strftime("%Y-%m-%d %H:%M"),
        "leg_time": waypoint_table["leg_time_s"].map(lambda s: "" if pd.isna(s) else _format_mmss(s)),
        "chosen_fl": waypoint_table["chosen_fl"].round(0),
        "mach": waypoint_table["mach"].round(3),
        "tas_kt": waypoint_table["tas_kt"].round(1),
        "gs_kt": waypoint_table["gs_kt"].round(1),
        "wind_kt": waypoint_table["wind_kt"].round(1),
        "temp_c": waypoint_table["temp_c"].round(1),
        "isa_dev_k": waypoint_table["isa_dev_k"].round(1),
        "binding": waypoint_table["binding"],
        # NaN everywhere except the decel waypoint's own (last) row -- the
        # arrival segment starts there, not at any of the earlier waypoints.
        "arrival_s": [np.nan] * (len(waypoint_table) - 1) + [arrival_time_s],
    })
    print(display.to_string(index=False))

    # Which waypoint interval top of climb falls in, from the same
    # per-waypoint cum_nm's the table above already computed.
    wp_names = waypoint_table["waypoint"].tolist()
    wp_cum_nm = waypoint_table["cum_nm"].tolist()
    after_idx = next((i for i, c in enumerate(wp_cum_nm) if c > climb_ground_nm),
                      len(wp_cum_nm) - 1)
    before_idx = max(after_idx - 1, 0)
    climb_between = f"{wp_names[before_idx]} -> {wp_names[after_idx]}"

    print("\nClimb segment (brake release -> top of climb):")
    warm_note = " [WARM -- clamped to +10C band, optimistic]" if climb_row["warm_flag"] else ""
    print(f"  TOW {tow_t:.0f} t, temp band {climb_row['temp_band']}{warm_note}")
    print(f"  top of climb: FL{TOP_OF_CLIMB_FL:.0f} at {climb_ground_nm:.1f} nm "
          f"(between {climb_between}), {_format_hmm(climb_time_s)}, "
          f"{climb_row['fuel_used_kg']:.0f} kg fuel, mass {climb_row['mass_t']:.1f} t")

    # Cruise-only sub-legs (eff_dist_nm > 0) -- climb-consumed ones carry a
    # climb-altitude chosen_fl that was never actually flown at cruise, so
    # they're excluded from the step-climb schedule and the FL summary
    # below, same as run_search's mean_fl.
    cruise_mask_1d = leg["eff_dist_nm"] > 0.0
    cruise_legs = [l for l, m in zip(cc_legs, cruise_mask_1d) if m]
    cruise_chosen_fl = leg["chosen_fl"][cruise_mask_1d]

    decel_clock_utc = departure_utc_ts + pd.Timedelta(seconds=total_elapsed_s)
    schedule = _step_climb_schedule(cruise_legs, cruise_chosen_fl)

    print("\nProfile summary:")
    print(f"  decel point: {cc_legs[-1].to_id} at {decel_clock_utc:%Y-%m-%d %H:%M} UTC")
    print("  step-climb schedule (cum_nm, FL):")
    for cum_nm, fl in schedule:
        print(f"    {cum_nm:8.1f} nm  FL{fl:.0f}")
    print(f"  chosen FL: min {cruise_chosen_fl.min():.0f}, "
          f"max {cruise_chosen_fl.max():.0f}, mean {cruise_chosen_fl.mean():.0f}")
    print(f"  weight: {climb_row['mass_t']:.1f} t at top of climb, "
          f"{weight_at_barix_t:.1f} t at {cc_legs[-1].to_id} "
          f"({climb_row['mass_t'] - weight_at_barix_t:.1f} t burned)")

    cruise_time_s = total_elapsed_s - climb_time_s

    # Same two calls run_search makes: KJFK wind at departure, EGLL wind at
    # touchdown (departure + climb+cruise + arrival).
    surface_data = load_surface_npz(surface_npz_path)
    touchdown_i8 = dep_i8 + (
        np.array([total_elapsed_s + arrival_time_s]) * NS_PER_S).astype("int64")
    jfk = runways.runway_screen(surface_data, "KJFK", dep_i8)
    lhr = runways.runway_screen(surface_data, "EGLL", touchdown_i8)
    runway_penalty_s = float(jfk["penalty_s"][0] + lhr["penalty_s"][0])
    touchdown_utc_ts = pd.Timestamp(int(touchdown_i8[0]))

    print()
    for line in _runway_lines(jfk, lhr, departure_utc_ts, touchdown_utc_ts):
        print(line)
    total_block_s = climb_time_s + cruise_time_s + arrival_time_s + runway_penalty_s

    print("\nTotals, brakes-release to touchdown:")
    print(f"  departure -> top of climb: {_format_hmm(climb_time_s)}")
    print(f"  top of climb -> {cc_legs[-1].to_id:<8}: {_format_hmm(cruise_time_s)}")
    print(f"  {cc_legs[-1].to_id} -> touchdown  : {_format_hmm(arrival_time_s)} "
          f"({arrival_nm:.0f} nm)")
    print(f"  runway penalty  : {_format_hmm(runway_penalty_s)} "
          f"(KJFK {jfk['runway'][0]}, EGLL {lhr['runway'][0]})")
    print(f"  total block time: {_format_hmm(total_block_s)}")

    out_path = Path(out_path)
    display.to_csv(out_path, index=False)
    print(f"\nWrote waypoint table to {out_path}")

    return waypoint_table


# ---------------------------------------------------------------------------
# Full-flight profile, brake release -> touchdown (notebooks/day-search-results)
# ---------------------------------------------------------------------------
# The three regions of the flight have very different resolution: the cruise is
# marched sub-leg by sub-leg (march_legs), the climb is conc_climb.csv's 18
# cumulative levels, the arrival is arrival.arrival()'s four segment totals.
# flight_profile stitches them into one point sequence so the notebook can plot
# altitude/Mach/CAS/fuel over the whole flight. What is NOT known is left NaN
# rather than invented: the climb table carries no speeds, so climb points have
# only the whole-climb mean ground speed (speed_basis "mean"), and the approach
# allowance is a distance/time/fuel constant with no speed either.


def _speeds(mach, alt_ft, isa_dev_k):
    """(tas_kt, cas_kt) at Mach `mach`, pressure altitude alt_ft, ISA + isa_dev_k."""
    t_k, p_pa = isa(alt_ft * FT_TO_M)
    tas_kt = float(mach * speed_of_sound(t_k + isa_dev_k) / KT_TO_MS)
    return tas_kt, float(cas_from_mach(mach, p_pa) / KT_TO_MS)


def _envelope_mach(alt_ft, weight_t, temp_k, cruise_mach):
    """min(cruise Mach, CAS-limit Mach, total-temp Mach) at one point. Same three
    limits as limits.max_mach, but the CAS one is solved exactly (mach_from_cas)
    rather than read off limits' FL280-FL600 grid, so it also holds in the climb
    and descent."""
    _, p_pa = isa(alt_ft * FT_TO_M)
    cas_mach = mach_from_cas(float(cas_limit_kt(alt_ft, weight_t)) * KT_TO_MS, float(p_pa))
    return min(cruise_mach, cas_mach, float(mach_from_total_temp(temp_k, TOTAL_TEMP_MAX_K)))


def _point(cum_nm, elapsed_s, alt_ft, weight_t, cruise_mach, *, mach=np.nan, tas_kt=np.nan,
           cas_kt=np.nan, gs_kt=np.nan, wind_kt=np.nan, isa_dev_k=0.0, subsonic=False,
           mach_limit=None, ceiling_ft=np.nan, binding="", speed_basis="derived"):
    """One profile point. mach_limit is SUBSONIC_LIMIT_MACH where `subsonic`, else
    the envelope at ISA + isa_dev_k, unless the caller already has it (cruise)."""
    if mach_limit is None:
        if subsonic:
            mach_limit = SUBSONIC_LIMIT_MACH
        else:
            t_k = float(isa(alt_ft * FT_TO_M)[0]) + isa_dev_k
            mach_limit = _envelope_mach(alt_ft, weight_t, t_k, cruise_mach)
    return dict(cum_nm=float(cum_nm), elapsed_s=float(elapsed_s), alt_ft=float(alt_ft),
                weight_t=float(weight_t), mach=mach, tas_kt=tas_kt, cas_kt=cas_kt, gs_kt=gs_kt,
                wind_kt=wind_kt, isa_dev_k=isa_dev_k, mach_limit=float(mach_limit),
                cas_limit_kt=float(cas_limit_kt(alt_ft, weight_t)), ceiling_ft=ceiling_ft,
                binding=binding, speed_basis=speed_basis)


def _climb_segments(climb_row, tow_t, linnd_nm, cruise_mach):
    """(phase, p0, p1) tuples for brake release -> top of climb, from conc_climb.csv:
    one segment per adjacent pair of table levels (the table is cumulative from
    brake release, so brake release -> FL230 is one straight segment -- the shape
    below the first row isn't tabulated). Air distance becomes ground distance
    with the climb's own single proxy wind, the same correction _climb_profile
    applies (recovered from its ground/air distance, not re-sampled). "climb" is
    brake release -> LINND (the subsonic limit), "acceleration" LINND -> top of
    climb; LINND is inserted as an interpolated breakpoint."""
    levels = CLIMB_LEVELS_FL[CLIMB_LEVELS_FL <= TOP_OF_CLIMB_FL]
    _m, fuel_kg, air_nm, time_min = climb_to(levels, tow_t, climb_row["temp_band"])
    wind_kt = (climb_row["ground_dist_nm"] - climb_row["dist_nm"]) / (climb_row["time_min"] / 60.0)
    mean_gs_kt = climb_row["ground_dist_nm"] / (climb_row["time_min"] / 60.0)

    cum = np.concatenate([[0.0], air_nm + wind_kt * time_min / 60.0])
    elapsed = np.concatenate([[0.0], time_min * 60.0])
    alt = np.concatenate([[0.0], levels * 100.0])
    weight = np.concatenate([[tow_t], tow_t - fuel_kg / 1000.0])

    if linnd_nm is not None and 0.0 < linnd_nm < cum[-1]:
        i = int(np.searchsorted(cum, linnd_nm))
        cum, elapsed, alt, weight = (
            np.insert(a, i, np.interp(linnd_nm, cum, a)) for a in (cum, elapsed, alt, weight))
    subsonic_until = linnd_nm if linnd_nm is not None else 0.0

    def pt(j, subsonic):
        return _point(cum[j], elapsed[j], alt[j], weight[j], cruise_mach, wind_kt=wind_kt,
                      gs_kt=mean_gs_kt, subsonic=subsonic, speed_basis="mean")

    segments = []
    for j in range(len(cum) - 1):
        subsonic = cum[j + 1] <= subsonic_until + 1e-9
        segments.append(("climb" if subsonic else "acceleration", pt(j, subsonic), pt(j + 1, subsonic)))
    return segments


def _cruise_segments(cc_legs, leg, weight_per_leg, weight_at_barix_t, cruise_mach):
    """One (phase, p0, p1) per cruise sub-leg (eff_dist_nm > 0 -- the ones
    actually flown at cruise; see march_legs). Values are constant across a
    sub-leg except weight (its own burn) and what depends on weight."""
    n_legs = len(cc_legs)
    segments = []
    for i in np.flatnonzero(leg["eff_dist_nm"] > 0.0):
        fl, mach = float(leg["chosen_fl"][i]), float(leg["mach"][i])
        isa_dev_k, temp_k = float(leg["isa_dev_k"][i]), float(leg["temp_c"][i]) + C_TO_K
        _, cas_kt = _speeds(mach, fl * 100.0, isa_dev_k)
        cum1 = cc_legs[i].cum_nm
        elapsed1 = float(leg["elapsed_s"][i])
        w0 = float(weight_per_leg[i])
        w1 = float(weight_per_leg[i + 1]) if i + 1 < n_legs else weight_at_barix_t

        def pt(cum, elapsed, w):
            return _point(cum, elapsed, fl * 100.0, w, cruise_mach, mach=mach,
                          tas_kt=float(leg["tas_kt"][i]), cas_kt=cas_kt, gs_kt=float(leg["gs_kt"][i]),
                          wind_kt=float(leg["wind_kt"][i]), isa_dev_k=isa_dev_k,
                          mach_limit=float(limits.max_mach(fl, temp_k, w, cruise_mach)),
                          ceiling_ft=float(limits.ceiling_ft(w, isa_dev_k)),
                          binding=str(leg["binding"][i]), speed_basis="march")

        segments.append(("cruise",
                         pt(cum1 - float(leg["eff_dist_nm"][i]), elapsed1 - float(leg["leg_time_s"][i]), w0),
                         pt(cum1, elapsed1, w1)))
    return segments


def _arrival_segments(a, start, cruise_fl, cruise_mach_at_barix, isa_dev_k, cruise_mach):
    """decel / subsonic cruise / descent / approach, from arrival.arrival()'s
    per-segment totals. Distance, time, fuel and weight are linear within a
    segment (only the totals are tabulated); altitude is linear in distance.
    Decel Mach is linear from the cruise Mach down to M1.0; the level segment is
    M0.95; the descent flies the schedule CAS (325/350/380 kt), so its Mach is
    solved from CAS at each altitude. Temperature is ISA + the last cruise leg's
    ISA deviation throughout (arrival.py doesn't return its own). The approach
    allowance has no speed model -- only its mean ground speed."""
    schedule_ms = a["schedule_kt"] * KT_TO_MS
    n = ARRIVAL_PROFILE_STEPS
    cursor = dict(cum_nm=start["cum_nm"], elapsed_s=start["elapsed_s"], weight_t=start["weight_t"])
    segments = []

    def ramp(phase, steps, dist_nm, time_min, fuel_t, alt0, alt1, mach_at, wind_kt, subsonic):
        pts = []
        for f in np.linspace(0.0, 1.0, steps + 1):
            alt = alt0 + f * (alt1 - alt0)
            mach = mach_at(alt, f)
            tas_kt, cas_kt = _speeds(mach, alt, isa_dev_k)
            pts.append(_point(cursor["cum_nm"] + f * dist_nm, cursor["elapsed_s"] + f * time_min * 60.0,
                              alt, cursor["weight_t"] - f * fuel_t, cruise_mach, mach=mach, tas_kt=tas_kt,
                              cas_kt=cas_kt, gs_kt=tas_kt + wind_kt, wind_kt=wind_kt, isa_dev_k=isa_dev_k,
                              subsonic=subsonic))
        segments.extend((phase, p0, p1) for p0, p1 in zip(pts[:-1], pts[1:]))
        cursor.update(cum_nm=pts[-1]["cum_nm"], elapsed_s=pts[-1]["elapsed_s"], weight_t=pts[-1]["weight_t"])

    level_ft = a["level_fl"] * 100.0
    ramp("deceleration", n, a["decel_nm"], a["decel_time_min"], a["decel_fuel_t"], cruise_fl * 100.0, level_ft,
         lambda alt, f: cruise_mach_at_barix + f * (SUBSONIC_LIMIT_MACH - cruise_mach_at_barix),
         a["decel_wind_kt"], False)
    ramp("subsonic cruise", 1, a["level_nm"], a["level_time_min"], a["level_fuel_t"], level_ft, level_ft,
         lambda alt, f: LEVEL_MACH, a["level_wind_kt"], True)
    ramp("descent", n, a["descent_nm"], a["descent_time_min"], a["descent_fuel_t"], level_ft, DESCENT_END_FT,
         lambda alt, f: mach_from_cas(schedule_ms, float(isa(alt * FT_TO_M)[1])), a["descent_wind_kt"], True)

    mean_gs_kt = APPROACH_NM / (APPROACH_MIN / 60.0)
    pts = [_point(cursor["cum_nm"] + f * APPROACH_NM, cursor["elapsed_s"] + f * APPROACH_MIN * 60.0,
                  DESCENT_END_FT * (1.0 - f), cursor["weight_t"] - f * APPROACH_FUEL_T, cruise_mach,
                  gs_kt=mean_gs_kt, subsonic=True, speed_basis="mean") for f in (0.0, 1.0)]
    segments.append(("approach", pts[0], pts[1]))
    return segments


def _phase_table(segments, departure_utc_ts, runway_penalties_s):
    """One row per phase (in flight order) from the (phase, p0, p1) segments, then
    the two runway penalties (time only -- search adds them to total block time)
    and a TOTAL row. Mach/TAS/CAS means are time-weighted, wind distance-weighted,
    mean GS is distance / time; a mean is NaN where the phase has no such data
    (climb has no speeds, only the whole-climb mean GS)."""
    def wmean(values, weights):
        ok = np.isfinite(values) & (weights > 0.0)
        return float(np.average(values[ok], weights=weights[ok])) if ok.any() else np.nan

    rows = []
    for phase in dict.fromkeys(p for p, _, _ in segments):
        pairs = [(a, b) for ph, a, b in segments if ph == phase]
        first, last = pairs[0][0], pairs[-1][1]
        dur = np.array([b["elapsed_s"] - a["elapsed_s"] for a, b in pairs])
        dist = np.array([b["cum_nm"] - a["cum_nm"] for a, b in pairs])

        def mid(key):
            return np.array([(a[key] + b[key]) / 2.0 for a, b in pairs], dtype=float)

        binding_s = {}
        for a, _b in pairs:
            if a["binding"]:
                binding_s[a["binding"]] = binding_s.get(a["binding"], 0.0) + 1.0
        alts = [p["alt_ft"] for pair in pairs for p in pair]
        rows.append(dict(
            phase=phase, start_nm=first["cum_nm"], end_nm=last["cum_nm"], dist_nm=dist.sum(),
            duration_s=dur.sum(),
            start_utc=departure_utc_ts + pd.Timedelta(seconds=first["elapsed_s"]),
            end_utc=departure_utc_ts + pd.Timedelta(seconds=last["elapsed_s"]),
            fuel_t=first["weight_t"] - last["weight_t"], weight_start_t=first["weight_t"],
            weight_end_t=last["weight_t"], fl_start=first["alt_ft"] / 100.0, fl_end=last["alt_ft"] / 100.0,
            fl_max=max(alts) / 100.0,
            mean_mach=wmean(mid("mach"), dur), mean_tas_kt=wmean(mid("tas_kt"), dur),
            mean_cas_kt=wmean(mid("cas_kt"), dur),
            mean_gs_kt=dist.sum() / dur.sum() * S_PER_HOUR if dur.sum() > 0 else np.nan,
            mean_wind_kt=wmean(mid("wind_kt"), dist),
            binding=max(binding_s, key=binding_s.get) if binding_s else "",
        ))

    flown = pd.DataFrame(rows)
    total = dict(phase="TOTAL", start_nm=flown["start_nm"].iloc[0], end_nm=flown["end_nm"].iloc[-1],
                 dist_nm=flown["dist_nm"].sum(), duration_s=flown["duration_s"].sum() + sum(runway_penalties_s),
                 start_utc=flown["start_utc"].iloc[0], end_utc=flown["end_utc"].iloc[-1],
                 fuel_t=flown["fuel_t"].sum(), weight_start_t=flown["weight_start_t"].iloc[0],
                 weight_end_t=flown["weight_end_t"].iloc[-1],
                 mean_gs_kt=flown["dist_nm"].sum() / flown["duration_s"].sum() * S_PER_HOUR)
    penalty_rows = [dict(phase=f"runway penalty ({apt})", duration_s=s)
                    for apt, s in zip(("KJFK", "EGLL"), runway_penalties_s)]
    return pd.concat([pd.DataFrame(penalty_rows[:1]), flown, pd.DataFrame(penalty_rows[1:]),
                      pd.DataFrame([total])], ignore_index=True)


def flight_profile(pln_path, npz_path, local_date, local_hour, zfw_t, tow_t=None,
                   subsonic_npz_path=None, arrival_upper_npz_path=None,
                   decel_id=DECEL_WAYPOINT_ID, cruise_mach=limits.CRUISE_MACH,
                   min_landing_fuel_t=fuel.MIN_LANDING_FUEL_T, runway_penalties_s=(0.0, 0.0)):
    """The whole flight, brake release -> touchdown, for ONE candidate departure
    -- run_report's model (resolve_tow_and_arrival, so TOW is solved from zfw_t
    or is the tow_t override) laid out as data instead of printed text.

    Returns dict(phases, profile, summary):
      phases   DataFrame, one row per phase (climb / acceleration / cruise /
               deceleration / subsonic cruise / descent / approach), then the
               runway penalties and a TOTAL row -- distance, duration, fuel,
               weight, FL, mean Mach/TAS/CAS/GS/wind. duration_s of TOTAL is
               search's total_time_s for the same candidate.
      profile  DataFrame of points, two per segment (a segment's start and end,
               so a step or a limit change plots as a vertical jump): phase, seg,
               cum_nm, elapsed_s, alt_ft, fl, weight_t, fuel_remaining_t, mach,
               tas_kt, cas_kt, gs_kt, wind_kt (along-track: + is tailwind),
               mach_limit, cas_limit_kt, ceiling_ft (cruise only), binding,
               speed_basis ("march" cruise sub-leg / "derived" arrival ramps /
               "mean" climb and approach, where only a mean GS exists), lat,
               lon, clock_utc. Mach/TAS/CAS are NaN in the climb (conc_climb.csv
               has no speeds).
      summary  dict of scalars: fuel plan (fuel loaded, burn split, landing fuel
               against the reserve, flags), climb facts, arrival schedule.

    Mach limit: SUBSONIC_LIMIT_MACH from brake release to ACCEL_WAYPOINT_ID and
    from the end of the decel segment onward; the cruise/CAS/total-temp envelope
    in between (ISA temperature off the cruise, where no temperature is known)."""
    legs = build_legs(parse_pln(pln_path)["waypoints"])
    mask = climb_cruise_segment(legs, decel_id=decel_id)
    cc_idx = np.flatnonzero(mask)
    cc_legs = [legs[i] for i in cc_idx]
    arrival_legs = [legs[i] for i in np.flatnonzero(~mask)]
    arrival_nm = legs[-1].cum_nm - cc_legs[-1].cum_nm

    departure_utc_ts = pd.Timestamp(local_to_departure_utc(local_date, local_hour))
    dep_i8 = np.array([departure_utc_ts.value], dtype="int64")
    tow_arr, n_iterations, plan_flags, legs_out, weight_per_leg, climb, arrival_out = (
        resolve_tow_and_arrival(
            cc_legs, cc_idx, arrival_legs, arrival_nm, load_legs_npz(npz_path),
            load_legs_npz(subsonic_npz_path) if subsonic_npz_path is not None else None, dep_i8,
            tow_t=tow_t, zfw_t=zfw_t, min_landing_fuel_t=min_landing_fuel_t, cruise_mach=cruise_mach,
            arrival_upper_data=(load_legs_npz(arrival_upper_npz_path)
                                if arrival_upper_npz_path is not None else None)))
    plan = fuel.fuel_plan(climb, legs_out, arrival_out, zfw_t=np.array([zfw_t], dtype=float),
                          min_landing_fuel_t=min_landing_fuel_t, tow_t=tow_arr,
                          n_iterations=n_iterations, flags=plan_flags)

    leg = {k: v[0] for k, v in legs_out.items() if k not in ("accumulated_s", "weight_at_barix")}
    climb_row = {k: (v[0] if k in ("temp_band", "warm_flag") else float(v[0])) for k, v in climb.items()}
    a = {k: float(v[0]) for k, v in arrival_out.items()
         if k not in ("by_schedule", "flags") and np.ndim(v) == 1 and v.dtype != bool}
    weight_at_barix_t = float(legs_out["weight_at_barix"][0])
    # The TOW the march actually flew: the fixed point's last march ran at the
    # iterate BEFORE its final damped update (within DEFAULT_TOLERANCE_T of
    # tow_arr), so the climb table is read at that one, not tow_arr, for the
    # profile's segments to join up.
    tow_flown_t = climb_row["mass_t"] + climb_row["fuel_used_kg"] / 1000.0
    tow_t = float(tow_arr[0])

    linnd_nm = next((l.cum_nm for l in legs if l.to_id == ACCEL_WAYPOINT_ID), None)
    segments = _climb_segments(climb_row, tow_flown_t, linnd_nm, cruise_mach)
    segments += _cruise_segments(cc_legs, leg, weight_per_leg[0], weight_at_barix_t, cruise_mach)
    last_cruise = segments[-1][2]
    segments += _arrival_segments(
        a, last_cruise, float(leg["chosen_fl"][-1]), float(leg["mach"][-1]),
        float(leg["isa_dev_k"][-1]), cruise_mach)

    rows = []
    for k, (phase, p0, p1) in enumerate(segments):
        rows += [dict(seg=k, phase=phase, **p0), dict(seg=k, phase=phase, **p1)]
    profile = pd.DataFrame(rows)
    profile["fl"] = profile["alt_ft"] / 100.0
    profile["fuel_remaining_t"] = profile["weight_t"] - zfw_t
    profile[["lat", "lon"]] = [position_at_cum_nm(legs, c)[:2] for c in profile["cum_nm"]]
    profile["clock_utc"] = departure_utc_ts + pd.to_timedelta(profile["elapsed_s"], unit="s")

    phases = _phase_table(segments, departure_utc_ts, runway_penalties_s)

    one = {k: (float(v[0]) if isinstance(v, np.ndarray) and k != "flags" else v) for k, v in plan.items()}
    landing_fuel_t = one["landing_weight_t"] - zfw_t
    summary = dict(
        tow_t=tow_t, zfw_t=zfw_t, fuel_loaded_t=tow_t - zfw_t, tow_required_t=one["tow_required_t"],
        trip_fuel_t=one["trip_fuel_t"], climb_fuel_t=one["climb_fuel_t"], cruise_fuel_t=one["cruise_fuel_t"],
        arrival_fuel_t=one["arrival_fuel_t"], reserve_t=min_landing_fuel_t,
        landing_weight_t=one["landing_weight_t"], landing_fuel_t=landing_fuel_t,
        reserve_margin_t=landing_fuel_t - min_landing_fuel_t, n_iterations=n_iterations,
        fuel_flag=plan["flags"][0], arrival_flags=arrival_out["flags"][0],
        schedule_kt=int(a["schedule_kt"]), temp_band=climb_row["temp_band"],
        climb_warm_clamped=bool(climb_row["warm_flag"]), top_of_climb_nm=climb_row["ground_dist_nm"],
        linnd_nm=linnd_nm, total_time_s=float(phases["duration_s"].iloc[-1]),
    )
    return dict(phases=phases, profile=profile, summary=summary)
