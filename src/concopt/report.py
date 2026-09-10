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

from concopt import limits
from concopt.data.conc_data import desc_time_min
from concopt.era5 import load_legs_npz
from concopt.route import build_legs, parse_pln, supersonic_segment
from concopt.search import (DEPARTURE_TO_ACCEL_S, NM_TO_M, NY_TZ, _format_hmm,
                             local_to_departure_utc, march_legs)

# 307 nm of deceleration + descent from BARIX to touchdown. No separate
# deceleration-time table exists -- only conc_desc_time.csv (descent time vs
# altitude, FL600 -> 17.1 min) -- so this default folds decel + descent into
# that one number, seeded from FL600 (the top of TARGET_FL), until Phase 4
# has a better split.
DECEL_DESCENT_NM = 307.0
DEFAULT_DECEL_DESCENT_S = float(desc_time_min(60000.0)) * 60.0


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


def _waypoint_table(ss_legs, departure_utc_ts, departure_to_accel_s, leg):
    """One row per original .pln waypoint spanned by ss_legs (accel point
    through decel point), aggregating each parent leg's subdivided sub-legs
    back up (build_legs gives every sub-leg of a parent the same
    from_id/to_id and equal length, so contiguous runs of equal (from_id,
    to_id) are exactly one parent leg, and a plain mean over them is already
    distance-weighted). `leg` is the per-sub-leg dict march_legs returned,
    already squeezed to this one candidate (1-D arrays, length len(ss_legs))."""
    accel_id = ss_legs[0].from_id
    accel_cum_nm = ss_legs[0].cum_nm - ss_legs[0].dist_nm
    accel_clock_utc = departure_utc_ts + pd.Timedelta(seconds=departure_to_accel_s)

    rows = [{
        "waypoint": accel_id,
        "cum_nm": accel_cum_nm,
        "elapsed_s": departure_to_accel_s,
        "clock_utc": accel_clock_utc,
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

    for (from_id, to_id), members in groupby(range(len(ss_legs)),
                                               key=lambda i: (ss_legs[i].from_id, ss_legs[i].to_id)):
        idx = np.array(list(members))
        last = idx[-1]
        rows.append({
            "waypoint": to_id,
            "cum_nm": ss_legs[last].cum_nm,
            "elapsed_s": leg["elapsed_s"][last],
            "clock_utc": departure_utc_ts + pd.Timedelta(seconds=float(leg["elapsed_s"][last])),
            "leg_time_s": leg["leg_time_s"][idx].sum(),
            "chosen_fl": leg["chosen_fl"][idx].mean(),
            "mach": leg["mach"][idx].mean(),
            "tas_kt": leg["tas_kt"][idx].mean(),
            "gs_kt": leg["gs_kt"][idx].mean(),
            "wind_kt": leg["wind_kt"][idx].mean(),
            "temp_c": leg["temp_c"][idx].mean(),
            "isa_dev_k": leg["isa_dev_k"][idx].mean(),
            "binding": Counter(leg["binding"][idx]).most_common(1)[0][0],
        })

    table = pd.DataFrame(rows)
    table["clock_local"] = table["clock_utc"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)
    return table


def _step_climb_schedule(ss_legs, chosen_fl):
    """[(cum_nm, FL), ...] at every sub-leg where the chosen level differs
    from the one before it -- sub-leg granularity, not waypoint-aggregated,
    so it shows the actual step climbs/descents within a leg too."""
    accel_cum_nm = ss_legs[0].cum_nm - ss_legs[0].dist_nm
    schedule = [(accel_cum_nm, float(chosen_fl[0]))]
    for i in range(1, len(chosen_fl)):
        if chosen_fl[i] != chosen_fl[i - 1]:
            start_cum_nm = ss_legs[i].cum_nm - ss_legs[i].dist_nm
            schedule.append((start_cum_nm, float(chosen_fl[i])))
    return schedule


def run_report(pln_path, npz_path, local_date, local_hour, accel_id="LINND",
                decel_id="BARIX", out_path="report.csv",
                departure_to_accel_s=DEPARTURE_TO_ACCEL_S,
                decel_descent_s=DEFAULT_DECEL_DESCENT_S,
                cruise_mach=limits.CRUISE_MACH):
    """The full breakdown for one candidate departure (local_date,
    local_hour, America/New_York). Reruns march_legs -- the same march
    run_search uses -- for this single candidate, then prints the waypoint
    table, profile summary and brakes-release-to-touchdown totals, and
    writes the waypoint table to out_path."""
    plan = parse_pln(pln_path)
    legs = build_legs(plan["waypoints"])
    mask = supersonic_segment(legs, accel_id=accel_id, decel_id=decel_id)
    ss_idx = np.flatnonzero(mask)
    ss_legs = [legs[i] for i in ss_idx]

    data = load_legs_npz(npz_path)
    departure_utc = local_to_departure_utc(local_date, local_hour)
    departure_utc_ts = pd.Timestamp(departure_utc)
    dep_i8 = np.array([departure_utc_ts.value], dtype="int64")

    legs_out, weight_per_leg = march_legs(ss_legs, ss_idx, data, dep_i8,
                                            departure_to_accel_s, cruise_mach)
    # Squeeze the n_cand=1 axis -- everything below is per sub-leg (1-D).
    scalar_keys = ("accumulated_s", "weight_at_barix")
    leg = {k: v[0] for k, v in legs_out.items() if k not in scalar_keys}
    total_elapsed_s = float(legs_out["accumulated_s"][0])
    weight_at_barix_t = float(legs_out["weight_at_barix"][0])
    weight_per_leg = weight_per_leg[0]  # (n_legs,), weight at the start of each sub-leg

    waypoint_table = _waypoint_table(ss_legs, departure_utc_ts, departure_to_accel_s, leg)

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

    accel_clock_utc = departure_utc_ts + pd.Timedelta(seconds=departure_to_accel_s)
    decel_clock_utc = departure_utc_ts + pd.Timedelta(seconds=total_elapsed_s)
    schedule = _step_climb_schedule(ss_legs, leg["chosen_fl"])

    print("\nProfile summary:")
    print(f"  accel point: {ss_legs[0].from_id} at {accel_clock_utc:%Y-%m-%d %H:%M} UTC")
    print(f"  decel point: {ss_legs[-1].to_id} at {decel_clock_utc:%Y-%m-%d %H:%M} UTC")
    print("  step-climb schedule (cum_nm, FL):")
    for cum_nm, fl in schedule:
        print(f"    {cum_nm:8.1f} nm  FL{fl:.0f}")
    print(f"  chosen FL: min {leg['chosen_fl'].min():.0f}, "
          f"max {leg['chosen_fl'].max():.0f}, mean {leg['chosen_fl'].mean():.0f}")
    print(f"  weight: {weight_per_leg[0]:.1f} t at {ss_legs[0].from_id}, "
          f"{weight_at_barix_t:.1f} t at {ss_legs[-1].to_id} "
          f"({weight_per_leg[0] - weight_at_barix_t:.1f} t burned)")

    accel_time_s = departure_to_accel_s
    cruise_time_s = total_elapsed_s - departure_to_accel_s
    runway_penalty_s = 0.0  # Phase 4 fills this in
    total_block_s = accel_time_s + cruise_time_s + decel_descent_s + runway_penalty_s

    print("\nTotals, brakes-release to touchdown:")
    print(f"  departure -> {ss_legs[0].from_id:<8}: {_format_hmm(accel_time_s)}")
    print(f"  {ss_legs[0].from_id} -> {ss_legs[-1].to_id:<8}: {_format_hmm(cruise_time_s)}")
    print(f"  {ss_legs[-1].to_id} -> touchdown  : {_format_hmm(decel_descent_s)} "
          f"({DECEL_DESCENT_NM:.0f} nm)")
    print(f"  runway penalty  : {_format_hmm(runway_penalty_s)} (placeholder -- Phase 4)")
    print(f"  total block time: {_format_hmm(total_block_s)}")

    out_path = Path(out_path)
    display.to_csv(out_path, index=False)
    print(f"\nWrote waypoint table to {out_path}")

    return waypoint_table
