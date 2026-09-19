"""Build the per-leg ERA5 .npz files that search/report/verify/inflight
take as --npz/--subsonic-npz/--arrival-upper-npz/--surface-npz.

There's no CLI subcommand for this step -- era5.reduce_to_legs/
reduce_surface_to_npz are called directly here against whichever netCDFs
are already on disk under --era5-dir (put there by
`python -m concopt.era5 --full`/`--subsonic`, see docs/RUNBOOK.md). Safe
to re-run any time more months finish downloading; each call just
overwrites its own .npz.

    poetry run python scripts/build_npz.py --pln <route.pln>

--only cruise/arrival/surface builds a subset (default: all three).
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from concopt import era5
from concopt.route import build_legs, climb_cruise_segment, parse_pln


def build_cruise(legs, cc_idx, era5_dir, out_path):
    cc_legs = [legs[i] for i in cc_idx]
    nc_paths = sorted(era5_dir.glob("era5_upper_*.nc"))
    print(f"cruise: {len(cc_legs)} legs, {len(nc_paths)} upper-air nc files -> {out_path}")
    era5.reduce_to_legs(nc_paths, cc_legs, out_path)


def build_arrival(arrival_legs, era5_dir, subsonic_out, upper_out):
    subsonic_nc = sorted(era5_dir.glob("era5_subsonic_*.nc"))
    upper_nc = sorted(era5_dir.glob("era5_upper_*.nc"))

    # A month only counts as "downloaded" once ALL its SUBSONIC_LEVEL_GROUPS
    # files exist -- era5.download_all_subsonic writes them one at a time, so
    # if it's still running (checked live, 2026-09: it writes a new file every
    # few minutes) a glob can catch a month with e.g. only its _0 file on
    # disk. Counting that month as available would feed reduce_to_legs a
    # (time, pressure_level) grid that's ragged for that one month's time
    # range -- xr.open_mfdataset's new join="exact" default (see era5.py)
    # correctly raises on that rather than the old join="outer" default's
    # silent NaN-pad, which is how this surfaced.
    month_of = lambda p: p.stem.split("_")[2]
    n_groups = len(era5.SUBSONIC_LEVEL_GROUPS)
    month_counts = Counter(month_of(p) for p in subsonic_nc)
    incomplete = {m for m, n in month_counts.items() if n != n_groups}
    if incomplete:
        print(f"NOTE: {len(incomplete)} subsonic month(s) only partially "
              f"downloaded (download_all_subsonic still running?) -- "
              f"excluding: {sorted(incomplete)}")
        subsonic_nc = [p for p in subsonic_nc if month_of(p) not in incomplete]

    # Both npz's must share the same time axis (search._build_arrival_wind_fn
    # asserts this at load time) -- restrict both to the months they share
    # if the subsonic download hasn't caught up to the upper-air one yet
    # (2 requests/month vs 1, so it lags).
    subsonic_months = {month_of(p) for p in subsonic_nc}
    upper_months = {month_of(p) for p in upper_nc}
    if subsonic_months != upper_months:
        common = subsonic_months & upper_months
        print(f"WARNING: subsonic covers {len(subsonic_months)} months, "
              f"upper-air covers {len(upper_months)} -- restricting both to "
              f"the {len(common)} shared months. Re-run once "
              f"era5.download_all_subsonic finishes for full archive coverage.")
        subsonic_nc = [p for p in subsonic_nc if month_of(p) in common]
        upper_nc = [p for p in upper_nc if month_of(p) in common]

    print(f"arrival: {len(arrival_legs)} legs, {len(subsonic_nc)} subsonic nc files -> {subsonic_out}")
    era5.reduce_to_legs(subsonic_nc, arrival_legs, subsonic_out)
    print(f"arrival: {len(arrival_legs)} legs, {len(upper_nc)} upper-air nc files -> {upper_out}")
    era5.reduce_to_legs(upper_nc, arrival_legs, upper_out)


def build_surface(era5_dir, out_path):
    nc_paths_by_airport = era5.surface_nc_paths(era5_dir)
    counts = {k: len(v) for k, v in nc_paths_by_airport.items()}
    print(f"surface: {counts} nc files -> {out_path}")
    era5.reduce_surface_to_npz(nc_paths_by_airport, out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pln", required=True, help="path to the route .pln")
    p.add_argument("--decel", default="BARIX", help="deceleration waypoint id (default: BARIX)")
    p.add_argument("--era5-dir", default="data/era5", help="directory holding the downloaded netCDFs")
    p.add_argument("--only", choices=["cruise", "arrival", "surface"], action="append",
                    help="build only these targets (default: all three)")
    args = p.parse_args()

    era5_dir = Path(args.era5_dir)
    targets = args.only or ["cruise", "arrival", "surface"]

    plan = parse_pln(args.pln)
    legs = build_legs(plan["waypoints"])
    mask = climb_cruise_segment(legs, decel_id=args.decel)
    cc_idx = np.flatnonzero(mask)
    arrival_legs = [leg for leg, cc in zip(legs, mask) if not cc]

    if "cruise" in targets:
        build_cruise(legs, cc_idx, era5_dir, era5_dir / "route_legs.npz")
    if "arrival" in targets:
        build_arrival(arrival_legs, era5_dir,
                       era5_dir / "subsonic_legs.npz", era5_dir / "arrival_upper_legs.npz")
    if "surface" in targets:
        build_surface(era5_dir, era5_dir / "surface_legs.npz")


if __name__ == "__main__":
    main()
