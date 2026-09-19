"""Download ERA5 reanalysis and reduce it to per-leg wind/temperature
arrays, plus per-airport surface wind arrays for the runway screen. cdsapi
+ xarray at the edges; everything downstream of reduce_to_legs/
reduce_surface_to_npz reads only the .npz each produces, never the netCDF.
"""
import argparse
import datetime as dt
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr

# Active Sky's historical archive starts here; no point downloading ERA5
# for a date the day-search can never actually query.
ARCHIVE_START = dt.date(2014, 8, 1)

# Route bbox + margin (N, W, S, E), JFK-LHR great circle.
UPPER_AIR_AREA = [53, -76, 38, 2]
UPPER_AIR_LEVELS = ["70", "100", "125", "150"]  # FL447-FL605
UPPER_AIR_GRID = [1.0, 1.0]

# These same netCDFs are reduced TWICE: once onto the cruise legs (the
# search's own --npz), and again onto the arrival legs (--arrival-upper-npz,
# B6) to extend arrival.py's decel-segment wind coverage above SUBSONIC_LEVELS'
# FL414 ceiling -- UPPER_AIR_AREA already covers the arrival region and 150
# joins 175 hPa (SUBSONIC_LEVELS' top) with no gap, so this needs no new CDS
# request, just a second era5.reduce_to_legs(nc_paths, arrival_legs, out_npz)
# call against the already-downloaded era5_upper_*.nc files.
# Departures are 08:00-14:00 America/New_York -> 12:00-18:00Z under EDT,
# 13:00-19:00Z under EST, plus ~4h flight time to cover the arrival end.
# Do not widen this.
UPPER_AIR_TIMES = [f"{h:02d}:00" for h in range(12, 24)]

# Subsonic cruise segment (last ~300 nm into LHR, arrival-only), FL183-FL414.
# Seven levels (175-500 hPa) join up with the 150 hPa data already downloaded.
# B6: this alone does not reach the decel segment's own wind-sampling
# midpoint from a realistic cruise level (roughly FL446-491) -- see
# UPPER_AIR_LEVELS above, which stitches on top to FL605.
SUBSONIC_LEVELS = ["175", "200", "225", "250", "300", "400", "500"]
SUBSONIC_AREA = [54, -13, 47, 2]  # N, W, S, E — arrival box only
SUBSONIC_GRID = [1.0, 1.0]
SUBSONIC_TIMES = UPPER_AIR_TIMES
# CDS caps this dataset at 4 distinct pressure levels per request, regardless
# of area size or month count -- confirmed by live bisection, 2026-09: every
# single level and every pair of SUBSONIC_LEVELS succeeded; 3- and 4-level
# combos succeeded (including at the full SUBSONIC_AREA above, not just a
# shrunk test box); 5, 6, and the full 7 were all rejected with "cost limits
# exceeded" -- even a 5-level/6-grid-cell request (a smaller total volume)
# was rejected while an unrelated 4-level/105-grid-cell request (a *larger*
# volume) had already succeeded, ruling out total data volume as the driver.
# Separately, a 12-month chunk of just 4 levels was *also* rejected, so the
# existing 1-month chunking (upper_air_months) still has to apply on top of
# the level cap -- the two are independent limits. SUBSONIC_LEVELS is
# therefore split into two <=4-level groups, one CDS request per (month,
# group) -- see download_subsonic.
#
# reduce_to_legs (below) needs no changes to consume these: confirmed by
# live trial, 2026-09, against synthetic files shaped like a 2-month x
# 2-group download (4 files, disjoint level sets, disjoint time ranges) --
# xr.open_mfdataset(paths, combine="by_coords") merges cleanly along BOTH
# the time and pressure_level axes into one (n_time, 7, lat, lon) dataset,
# no NaNs, no error. Pass every era5_subsonic_*.nc path (both groups, every
# month) to reduce_to_legs at once, same as the upper-air files.
SUBSONIC_LEVEL_GROUPS = [SUBSONIC_LEVELS[:4], SUBSONIC_LEVELS[4:]]

# Phase 4 (runways.py's crosswind/tailwind screen, and later the in-flight
# advisor) consumes these; downloaded now so both sit in the same CDS
# queue as the upper-air requests.
SURFACE_AREAS = {"KJFK": [41, -74, 40, -73], "EGLL": [52, -1, 51, 0]}
SURFACE_TIMES = [f"{h:02d}:00" for h in range(24)]

_ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]


def _year_months(year):
    """Zero-padded calendar months to request for `year`. Full 12 except
    at the two boundary years, where it's clamped to ARCHIVE_START /
    today so we never ask CDS for a date outside the downloadable range."""
    today = dt.date.today()
    if not (ARCHIVE_START.year <= year <= today.year):
        raise ValueError(
            f"year {year} outside downloadable range "
            f"{ARCHIVE_START.year}-{today.year}"
        )
    lo = ARCHIVE_START.month if year == ARCHIVE_START.year else 1
    hi = today.month if year == today.year else 12
    return [f"{m:02d}" for m in range(lo, hi + 1)]


def upper_air_months():
    """(year, month) pairs spanning the Active Sky archive: 2014-08
    through the current calendar month. One request per *month*, not per
    year: at UPPER_AIR_AREA/UPPER_AIR_GRID, the CDS cost check (a
    resolution-weighted limit, distinct from and much stricter than the
    120,000-item cap) accepts 1 month of this request (~4,464 items) but
    rejects 2 (~8,500 items) -- confirmed by live trial against the API,
    2026-09."""
    today = dt.date.today()
    y, m = ARCHIVE_START.year, ARCHIVE_START.month
    months = []
    while (y, m) <= (today.year, today.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def download_upper_air(out_dir, year, month):
    """One reanalysis-era5-pressure-levels request for one calendar
    month: u, v, t at 70/100/125/150 hPa, 12:00-23:00Z, 1x1 deg over the
    route bbox. Skips the request if the output file already exists --
    CDS queues are slow and this gets re-run."""
    out_path = Path(out_dir) / f"era5_upper_{year}{month:02d}.nc"
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind",
                "temperature",
            ],
            "pressure_level": UPPER_AIR_LEVELS,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": _ALL_DAYS,
            "time": UPPER_AIR_TIMES,
            "area": UPPER_AIR_AREA,
            "grid": UPPER_AIR_GRID,
            "data_format": "netcdf",
        },
        str(out_path),
    )
    return out_path


def _month_chunks(year, chunk_size=6):
    """_year_months(year) split into runs of at most chunk_size, in order.
    Even the tiny surface area trips the CDS cost check on a full
    12-month request (confirmed by live trial: 9 months clears it, 12
    doesn't) -- so this one also has to go in under a year at a time."""
    months = _year_months(year)
    return [months[i:i + chunk_size] for i in range(0, len(months), chunk_size)]


def download_surface(out_dir, year):
    """reanalysis-era5-single-levels requests for `year`, chunked to at
    most 6 months each per airport (see _month_chunks) -- even this tiny
    area/resolution trips the CDS cost check over a full year. 10 m u/v
    and instantaneous gust, all 24 hours, native 0.25 deg. Two small
    areas (KJFK, EGLL) rather than the route bbox -- tens of MB total.
    Feeds phase 4, not reduce_to_legs below. Returns {name: [paths]}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, area in SURFACE_AREAS.items():
        chunk_paths = []
        for chunk in _month_chunks(year):
            out_path = out_dir / f"era5_sfc_{name.lower()}_{year}_{chunk[0]}-{chunk[-1]}.nc"
            if not out_path.exists():
                cdsapi.Client().retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": [
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                            "instantaneous_10m_wind_gust",
                        ],
                        "year": [str(year)],
                        "month": chunk,
                        "day": _ALL_DAYS,
                        "time": SURFACE_TIMES,
                        "area": area,
                        "data_format": "netcdf",
                    },
                    str(out_path),
                )
            chunk_paths.append(out_path)
        paths[name] = chunk_paths
    return paths


def surface_nc_paths(out_dir):
    """{"KJFK": [...], "EGLL": [...]} of already-downloaded surface
    netCDFs in out_dir, found by filename pattern
    (era5_sfc_<airport>_*.nc) rather than replaying download_surface's
    year/chunk boundaries. Feeds reduce_surface_to_npz; works whether the
    files came from download_all_surface or were fetched by hand."""
    out_dir = Path(out_dir)
    return {
        name: sorted(out_dir.glob(f"era5_sfc_{name.lower()}_*.nc"))
        for name in SURFACE_AREAS
    }


def reduce_surface_to_npz(nc_paths_by_airport, out_npz):
    """Box-mean 10 m u/v wind + instantaneous gust time series for each
    surface airport (see surface_nc_paths), combined into one npz with
    per-airport-prefixed keys (e.g. "KJFK_time", "KJFK_u10", "KJFK_v10",
    "KJFK_i10fg"). A box mean, not an interpolation to a point: at 0.25 deg
    native resolution there is no single "airport" grid cell inside
    SURFACE_AREAS's ~1x1 deg box, and a box mean is good enough for a wind
    screen. runways.py reads this; nothing downstream reopens the netCDF."""
    arrays = {}
    for name, paths in nc_paths_by_airport.items():
        # combine="nested"/concat_dim=<time>, not "by_coords": lat/lon and
        # variables are identical across every file for one airport, only
        # time varies, so which dim to concat along is already known -- no
        # coordinate auto-detection needed. This matters because the most
        # recent chunk (download_surface's _month_chunks, current year) is
        # short and CDS truncates it further to ERA5's actual processing lag
        # (a handful of days behind "today"), so it's an irregular size next
        # to every other chunk; combine="by_coords" mis-happens to treat that
        # boundary as needing ALIGNMENT rather than concatenation, and raises
        # an AlignmentError under xarray's new join="exact" default even
        # though the two chunks are genuinely disjoint and contiguous in
        # time -- caught live, 2026-09, against the real KJFK/EGLL archive
        # (both fully reproduced: same total size, explicit concat_dim vs the
        # old join="outer" default, which had silently tolerated the same
        # ambiguity). era5.reduce_to_legs (below) keeps combine="by_coords"
        # -- it genuinely needs auto-detection across two varying dims (time
        # AND pressure_level, from era5.py's SUBSONIC_LEVEL_GROUPS), and its
        # own use_new_combine_kwarg_defaults there is doing real work (see
        # its docstring), not working around this same limitation.
        first = xr.open_dataset(str(paths[0]))
        time_dim = "valid_time" if "valid_time" in first.dims else "time"
        first.close()
        ds = xr.open_mfdataset([str(p) for p in paths], combine="nested", concat_dim=time_dim)
        box_mean = ds.mean(dim=("latitude", "longitude"))

        arrays[f"{name}_time"] = box_mean[time_dim].values
        arrays[f"{name}_u10"] = box_mean["u10"].values
        arrays[f"{name}_v10"] = box_mean["v10"].values
        arrays[f"{name}_i10fg"] = box_mean["i10fg"].values
        ds.close()

    np.savez(out_npz, **arrays)
    return Path(out_npz)


def load_surface_npz(path):
    """out_npz from reduce_surface_to_npz -> {"KJFK": {...}, "EGLL":
    {...}}, one dict of time/u10/v10/i10fg arrays per airport. Everything
    downstream (runways.py) reads this; nothing downstream reopens the
    netCDF."""
    with np.load(path) as z:
        airports = sorted({k.split("_", 1)[0] for k in z.files})
        return {
            name: {var: z[f"{name}_{var}"] for var in ("time", "u10", "v10", "i10fg")}
            for name in airports
        }


def reduce_to_legs(nc_paths, legs, out_npz):
    """Bilinearly interpolate u, v, t from the upper-air netCDFs to each
    leg's (lat_mid, lon_mid) -- one xarray .interp() call over every time
    and level at once. Result arrays are (time, level, leg); saved to
    out_npz with the time axis, pressure levels, and each leg's cum_nm/
    track_deg alongside, so the scan never has to reopen the netCDF."""
    with xr.set_options(use_new_combine_kwarg_defaults=True):
        ds = xr.open_mfdataset([str(p) for p in nc_paths], combine="by_coords")

    lat = xr.DataArray([leg.lat_mid for leg in legs], dims="leg")
    lon = xr.DataArray([leg.lon_mid for leg in legs], dims="leg")
    pt = ds.interp(latitude=lat, longitude=lon, method="linear")

    time_dim = "valid_time" if "valid_time" in pt.dims else "time"
    level_dim = "pressure_level" if "pressure_level" in pt.dims else "level"

    np.savez(
        out_npz,
        time=pt[time_dim].values,
        level=pt[level_dim].values,
        u=pt["u"].transpose(time_dim, level_dim, "leg").values,
        v=pt["v"].transpose(time_dim, level_dim, "leg").values,
        t=pt["t"].transpose(time_dim, level_dim, "leg").values,
        cum_nm=np.array([leg.cum_nm for leg in legs]),
        track_deg=np.array([leg.track_deg for leg in legs]),
    )
    ds.close()
    return Path(out_npz)


def load_legs_npz(path):
    """out_npz from reduce_to_legs -> dict of arrays. Everything
    downstream reads this; nothing downstream reopens the netCDF."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def download_all_upper_air(out_dir):
    """download_upper_air for every month in upper_air_months(), in
    order. Slow (each request is minutes in the CDS queue; ~146 months
    across the full archive) but each already-downloaded month is
    skipped, so this is safe to interrupt and re-run."""
    return [download_upper_air(out_dir, y, m) for y, m in upper_air_months()]


def download_subsonic(out_dir, year, month):
    """Two reanalysis-era5-pressure-levels requests for one calendar month
    -- one per SUBSONIC_LEVEL_GROUPS entry, since CDS caps this dataset at 4
    distinct pressure levels per request (see SUBSONIC_LEVELS comment): u, v,
    t at 175/200/225/250 hPa and at 300/400/500 hPa, 12:00-23:00Z, 1x1 deg
    over the arrival box. Skips a group's request if its output file already
    exists. Returns both paths."""
    out_path = Path(out_dir) / f"era5_subsonic_{year}{month:02d}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, levels in enumerate(SUBSONIC_LEVEL_GROUPS):
        group_path = out_path.with_stem(f"{out_path.stem}_{i}")
        if not group_path.exists():
            cdsapi.Client().retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "u_component_of_wind",
                        "v_component_of_wind",
                        "temperature",
                    ],
                    "pressure_level": levels,
                    "year": [str(year)],
                    "month": [f"{month:02d}"],
                    "day": _ALL_DAYS,
                    "time": SUBSONIC_TIMES,
                    "area": SUBSONIC_AREA,
                    "grid": SUBSONIC_GRID,
                    "data_format": "netcdf",
                },
                str(group_path),
            )
        paths.append(group_path)
    return paths


def download_all_subsonic(out_dir):
    """download_subsonic for every month in upper_air_months(), in order.
    292 requests total (146 months x 2 level-groups) -- twice
    download_all_upper_air's count, since the 7 subsonic levels don't fit
    under CDS's 4-level-per-request cap (see SUBSONIC_LEVELS comment). Each
    already-downloaded group file is skipped, so this is safe to interrupt
    and re-run."""
    paths = []
    for y, m in upper_air_months():
        paths.extend(download_subsonic(out_dir, y, m))
    return paths


def download_all_surface(out_dir):
    """download_surface for every year from ARCHIVE_START through
    today. Small and quick relative to download_all_upper_air."""
    today = dt.date.today()
    paths = {}
    for year in range(ARCHIVE_START.year, today.year + 1):
        paths[year] = download_surface(out_dir, year)
    return paths


def run_pilot(out_dir, year=2015):
    """End-to-end sanity check: download one year of upper-air data (12
    monthly requests -- see upper_air_months), run it through
    reduce_to_legs against the real JFK-LHR route, and print shapes/
    ranges. No route .pln handy here, so legs are stubbed straight across
    the bbox at 1 nm spacing -- good enough to sanity-check the
    interpolation and the data itself."""
    from concopt.route import Leg

    out_dir = Path(out_dir)
    nc_paths = [
        download_upper_air(out_dir, y, m)
        for y, m in upper_air_months() if y == year
    ]

    lats = np.linspace(40.6, 51.5, 50)   # JFK -> LHR, roughly
    lons = np.linspace(-73.8, -0.5, 50)
    legs = [
        Leg("A", "B", float(lat), float(lon), 90.0, 1.0, float(i))
        for i, (lat, lon) in enumerate(zip(lats, lons))
    ]

    out_npz = out_dir / f"pilot_legs_{year}.npz"
    reduce_to_legs(nc_paths, legs, out_npz)
    data = load_legs_npz(out_npz)

    print(f"time: {data['time'].shape}  {data['time'].min()} .. {data['time'].max()}")
    print(f"level: {data['level']}")
    for var in ("u", "v", "t"):
        arr = data[var]
        print(f"{var}: shape {arr.shape}  min {arr.min():.2f}  max {arr.max():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERA5 download/reduce (concopt phase 3a)")
    parser.add_argument("--pilot", action="store_true", help="run one year end-to-end and print shapes/ranges")
    parser.add_argument("--full", action="store_true", help="download the full archive window (upper-air + surface)")
    parser.add_argument("--subsonic", action="store_true", help="download subsonic cruise segment (arrival-only, FL183-FL414)")
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--out-dir", default="data/era5")
    args = parser.parse_args()

    if args.pilot:
        run_pilot(args.out_dir, args.year)
    elif args.full:
        download_all_upper_air(args.out_dir)
        download_all_surface(args.out_dir)
    elif args.subsonic:
        download_all_subsonic(args.out_dir)
    else:
        parser.error("nothing to do without --pilot, --full, or --subsonic")
