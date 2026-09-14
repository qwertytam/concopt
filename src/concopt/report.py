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

from concopt import arrival, fuel, limits
from concopt.era5 import load_legs_npz
from concopt.route import build_legs, climb_cruise_segment, parse_pln
from concopt.search import (DEFAULT_TOW_T, NM_TO_M, NY_TZ, TOP_OF_CLIMB_FL,
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

    for (from_id, to_id), members in groupby(range(len(cc_legs)),
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
        lines.append("  TOW given, no fixed point -- ZFW above is what that TOW implies")
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
    """arrival.arrival()'s (or arrival.flat_arrival's) dict (n_cand=1),
    arrival_nm, and the cruise FL flown into the decel point -> the printed
    Arrival block, as a list of lines. The per-segment split is the point:
    it is what makes it obvious when the level segment is dominating."""
    a = {k: (float(v[0]) if isinstance(v, np.ndarray) and k != "flags" else v)
         for k, v in arrival_out.items() if k != "by_schedule"}
    a["flags"] = arrival_out["flags"][0]
    schedule_kt = int(a["schedule_kt"])
    header = f"Arrival (BARIX -> touchdown, {arrival_nm:.0f} nm)"

    if schedule_kt == 0:
        # --decel-descent-min forced arrival.flat_arrival -- there is no
        # real decel/level/descent split to show (see flat_arrival's
        # docstring), so print the flat legacy pair on its own rather than
        # a segment breakdown whose rows wouldn't sum to the total.
        return [
            header + " -- FLAT OVERRIDE (--decel-descent-min)",
            f"  {'total':<16}{arrival_nm:6.0f} nm  {a['time_min']:5.1f} min  "
            f"{a['fuel_t']:5.2f} t",
        ]

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
                decel_id="BARIX", out_path="report.csv",
                tow_t=None, zfw_t=None,
                min_landing_fuel_t=fuel.MIN_LANDING_FUEL_T,
                subsonic_npz_path=None, decel_descent_min=None,
                cruise_mach=limits.CRUISE_MACH,
                arrival_upper_npz_path=None):
    """The full breakdown for one candidate departure (local_date,
    local_hour, America/New_York). Reruns march_legs -- the same march
    run_search uses -- for this single candidate, then prints the fuel plan,
    the arrival breakdown, waypoint table, climb segment, profile summary
    and brakes-release-to-touchdown totals, and writes the waypoint table to
    out_path.

    zfw_t (tonnes) is the primary weight input: TOW is solved (via
    resolve_tow_and_arrival) rather than given, so the report shows the
    weight the day actually needs, arrival fuel included. tow_t overrides
    that -- given, the fixed point is skipped entirely and ZFW is instead
    derived from the trip fuel that TOW produces ("what if I actually load
    X"). Neither given falls back to a flat DEFAULT_TOW_T, same as before
    the fuel plan existed.

    subsonic_npz_path (era5.reduce_to_legs run against the post-BARIX legs)
    and arrival_upper_npz_path (era5.reduce_to_legs run against those SAME
    legs, from the UPPER_AIR_LEVELS netCDFs already downloaded for the
    cruise legs -- B6, no new CDS download) together drive the real
    arrival.arrival() model's stitched FL183-FL605 wind profile; both
    required unless decel_descent_min forces the flat legacy arrival
    instead, for comparing old and new numbers directly."""
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
            decel_descent_min=decel_descent_min, cruise_mach=cruise_mach,
            arrival_upper_data=arrival_upper_data,
        )
    )
    tow_t = float(tow_arr[0])
    # n_iterations is None exactly when resolve_tow_and_arrival took the
    # plain --tow path (no fixed point ran) -- the same discriminator
    # fuel_plan itself uses, so zfw_t is only passed through when it was
    # actually the thing being solved for.
    zfw_arr = np.array([zfw_t], dtype=float) if n_iterations is not None else None
    plan = fuel.fuel_plan(climb, legs_out, arrival_out,
                           zfw_t=zfw_arr,
                           min_landing_fuel_t=min_landing_fuel_t,
                           tow_t=tow_arr, n_iterations=n_iterations,
                           flags=plan_flags)

    for line in _fuel_plan_lines(plan):
        print(line)
    print()

    cruise_fl_at_barix = float(legs_out["chosen_fl"][0, -1])
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

    arrival_time_s = float(arrival_out["time_min"][0]) * 60.0
    cruise_time_s = total_elapsed_s - climb_time_s
    runway_penalty_s = 0.0  # Phase 4 fills this in
    total_block_s = climb_time_s + cruise_time_s + arrival_time_s + runway_penalty_s

    print("\nTotals, brakes-release to touchdown:")
    print(f"  departure -> top of climb: {_format_hmm(climb_time_s)}")
    print(f"  top of climb -> {cc_legs[-1].to_id:<8}: {_format_hmm(cruise_time_s)}")
    print(f"  {cc_legs[-1].to_id} -> touchdown  : {_format_hmm(arrival_time_s)} "
          f"({arrival_nm:.0f} nm)")
    print(f"  runway penalty  : {_format_hmm(runway_penalty_s)} (placeholder -- Phase 4)")
    print(f"  total block time: {_format_hmm(total_block_s)}")

    out_path = Path(out_path)
    display.to_csv(out_path, index=False)
    print(f"\nWrote waypoint table to {out_path}")

    return waypoint_table
