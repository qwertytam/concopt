# concopt

Optimise Concorde (FS Labs, P3D v5) flight time JFK→LHR. Two tools: an offline
historical-weather day search, and an in-flight altitude advisor.

Personal project, used a handful of times. **Minimum viable, not well-engineered.**
No packaging polish beyond what's needed to install cleanly, no abstraction
for its own sake, no defensive error handling.

## Layout
- Managed with Poetry (`pyproject.toml` is PEP 621-native — poetry-core reads
  the standard `[project]` table directly, no `[tool.poetry]` block needed).
  `poetry install --extras test` sets up `.venv/` (in-project, see
  `poetry.toml`) with the package installed editable plus pytest.
  `pip install -e '.[test]'` works too, same dependency set either way.
- `src/concopt/` — package (src-layout; auto-discovered by poetry-core since
  the directory name matches the project name)
- `atmos.py`, `limits.py` — vectorised SI numpy, **no pint**. Hot path.
- `tests/` — pytest suite: `cas_formula.md` worked examples, the ISA+limits
  table, the Mmo/total-temp crossover identity, flight-level round-trip, and
  a vectorised-scan performance/NaN check. Run with `poetry run pytest` (or
  plain `pytest` inside the venv). This replaced the old ad hoc `check.py`
  script — there is no `check.py` any more.
- `asky.py` — ActiveSky HTTP client (localhost:19285)
- `era5.py` — ERA5 reanalysis download (`cdsapi`) + reduction to per-leg
  wind/temperature arrays (`xarray`, needs `dask` for the multi-file
  open). `download_upper_air(out_dir, year, month)` is one CDS request
  per **calendar month**, not per year — a full year at the route
  bbox/1° grid trips the CDS *cost* limit (a resolution-weighted check,
  separate from and much stricter than the docs' assumed 120,000-item
  cap); confirmed by live trial against the API, 2026-09. Use
  `upper_air_months()` to enumerate the (year, month) pairs spanning the
  Active Sky archive. `reduce_to_legs` bilinearly interpolates u/v/t onto
  each leg's midpoint with one `xarray.interp()` call and writes a single
  `.npz`; nothing downstream reopens the netCDF. Requires accepting the
  CDS licences for both `reanalysis-era5-pressure-levels` and
  `reanalysis-era5-single-levels` at cds.climate.copernicus.eu first (one
  manual, per-account step). Downloads land in `data/era5/`
  (gitignored — large and re-downloadable; `download_*` skip a request
  whose output file already exists). `reduce_surface_to_npz` does the same
  job for the KJFK/EGLL surface downloads (`download_surface`,
  `surface_nc_paths`) that `reduce_to_legs` does for the upper air: one
  combined `.npz`, box-mean (not point-interpolated — 0.25° native res, no
  single grid cell inside either ~1°×1° box) `u10`/`v10`/`i10fg` time
  series per airport, keyed `<airport>_time`/`_u10`/`_v10`/`_i10fg`. Feeds
  `runways.py`.
- `search.py` — the day/time scan (`concopt search`). Candidates are every
  date from `era5.ARCHIVE_START` to today at 08:00-14:00 America/New_York
  (7/day), built tz-aware with `zoneinfo` and converted to UTC so DST
  doesn't silently shift winter candidates by an hour — about 30,933 for
  the real archive. `run_search` marches the supersonic legs (from
  `route.supersonic_segment`) one at a time in a plain Python loop, but
  every op inside that loop (time-interpolation into the `era5.py` `.npz`,
  vertical interpolation of u/v/t onto the FL450-FL600 1,000 ft grid,
  `limits.best_level`) is vectorised across all candidates at once — never
  loop over candidates. Weight is a state variable, not a schedule: it
  starts at 165 t at the accel point and is burned off leg by leg from
  `conc_data.fuel_total_kgh_table` (the Air France performance table), at
  the weight/ISA-deviation the leg was actually flown at, feeding
  `limits.ceiling_ft(weight_t, isa_dev_c)` for the next leg — a still-air,
  ISA+0 run over the real supersonic segment burns 165 t down to ~111 t by
  BARIX (see the printed sanity checks, which report rather than assert).
  Assumes the `.npz` was built from the same `--pln`/`build_legs` call, so
  the leg axes line up by position — it does not re-check this. Also
  screens each candidate's KJFK departure / EGLL arrival wind through
  `runways.py` (see below) and ranks on `total_time`, not supersonic time.
- `runways.py` — runway selection and crosswind/tailwind screen (Phase 4),
  `--surface-npz` from `era5.reduce_surface_to_npz`. `RUNWAYS` is the
  geometry: JFK 22R/31L only (not 04L/13R), EGLL's parallel 09L/09R and
  27R/27L, all bearings **true** — ERA5 wind is in the true frame, and
  JFK's ~13° W variation would put every runway number 13° off if bearings
  were magnetic. Headwind/crosswind come from the same along-track/
  cross-track decomposition `limits.ground_speed` already uses against a
  leg's track, evaluated against the runway heading instead — algebraically
  identical to the textbook "wind FROM direction" formula, without an
  atan2/degrees round trip. ERA5's gust (`i10fg`, max gust in the
  preceding hour) has no direction, so the crosswind-gust test reuses the
  mean wind's direction and rescales only the magnitude onto the gust
  speed. Screen: crosswind gust ≤25 kt OK, 25–30 kt allowed but flagged,
  >30 kt unflyable on that runway; tailwind (mean) >10 kt unflyable, no
  buffer. Picks the greatest-headwind runway among those that pass; if
  none pass, the day is flagged `unflyable` at that airport but still
  reports the greatest-headwind runway and a computed time — never
  dropped. `search.DECEL_DESCENT_S` (35 min default, distinct from
  `report.DEFAULT_DECEL_DESCENT_S`) is only used here, to estimate
  touchdown clock time for sampling EGLL's arrival wind.
- `data/` — CSV limit tables + `conc_data.py` loader
- `condition.py`, `atmosphere.py`, `common.py`, `airframeflows.py`,
  `nondimensional.py` — vendored fork of the `flightcondition` package.
  **Do not read or modify these.** Legacy; retained only for the pretty
  `tostring()` output in the future in-flight display.
- `nb/max_gs.ipynb` — stale (imports a pre-2023 layout). Do not run or fix.

## Conventions
- SI internally (K, Pa, m/s, m). Convert at the edges only.
- Everything array-in / array-out. No `iterrows()`, no per-row Python loops
  in anything that touches the search — it runs over ~31,000 candidate
  departures × ~32 supersonic legs × 4 ERA5 pressure levels.
- pint is allowed **only** in display code, never in `atmos.py`/`limits.py`.

## Aircraft constants
- Mmo 2.04 (structural limit, `conc_data.MMO`); cruise is flown at
  `limits.CRUISE_MACH` (2.00 by default, `--cruise-mach` to try Mmo instead)
  — the Air France performance table's ceilings are altitudes attainable
  *at* CRUISE_MACH, so pairing them with Mmo isn't physical. Max total
  (stagnation) temperature 127 °C; service ceiling 60,000 ft.
- CAS limit table: `data/conc_cas_limit.csv`, altitude ft × weight t (105/135/165)
- Above FL430 the CAS limit is 530 kt at **all** weights — weight does not
  enter the CAS-speed calculation at all, but it does (with ISA deviation)
  drive the ceiling table below.
- Cruise/ceiling table: `data/conc_supersonic_cruise.csv`
  (`conc_data.ceiling_ft_table`/`fuel_total_kgh_table`), weight t (100-165,
  5 t steps) × ISA deviation °C (-30..+15) — 126 rows. Ceiling ranges from
  43,494 ft (165 t, ISA+15) to 60,000 ft (light + cold, clamped at the
  service ceiling); one cell (165 t, ISA-30) is thrust-limited and flagged
  in its `note`, excluded from the max_mach/ceiling cross-check in
  `tests/test_perf_table.py`.

## Known non-problems — do not "fix" these
- The CSVs have a UTF-8 BOM. Current pandas and numpy strip it. Leave it.
- `README.rst` is empty. Intentional for now.