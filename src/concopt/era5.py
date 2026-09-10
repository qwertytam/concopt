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
UPPER_AIR_LEVELS = ["70", "100", "125", "150"]
UPPER_AIR_GRID = [1.0, 1.0]
# Departures are 08:00-14:00 America/New_York -> 12:00-18:00Z under EDT,
# 13:00-19:00Z under EST, plus ~4h flight time to cover the arrival end.
# Do not widen this.
UPPER_AIR_TIMES = [f"{h:02d}:00" for h in range(12, 24)]

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
        ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords")
        box_mean = ds.mean(dim=("latitude", "longitude"))
        time_dim = "valid_time" if "valid_time" in box_mean.dims else "time"

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
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--out-dir", default="data/era5")
    args = parser.parse_args()

    if args.pilot:
        run_pilot(args.out_dir, args.year)
    elif args.full:
        download_all_upper_air(args.out_dir)
        download_all_surface(args.out_dir)
    else:
        parser.error("nothing to do without --pilot or --full")
