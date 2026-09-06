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
  whose output file already exists).
- `data/` — CSV limit tables + `conc_data.py` loader
- `condition.py`, `atmosphere.py`, `common.py`, `airframeflows.py`,
  `nondimensional.py` — vendored fork of the `flightcondition` package.
  **Do not read or modify these.** Legacy; retained only for the pretty
  `tostring()` output in the future in-flight display.
- `nb/max_gs.ipynb` — stale (imports a pre-2023 layout). Do not run or fix.

## Conventions
- SI internally (K, Pa, m/s, m). Convert at the edges only.
- Everything array-in / array-out. No `iterrows()`, no per-row Python loops
  in anything that touches the search — it runs over ~25,000 candidate
  departures × ~20 route legs × ~8 levels.
- pint is allowed **only** in display code, never in `atmos.py`/`limits.py`.

## Aircraft constants
- Mmo 2.04; max total (stagnation) temperature 127 °C; service ceiling 60,000 ft
- CAS limit table: `data/conc_cas_limit.csv`, altitude ft × weight t (105/135/165)
- Above FL430 the CAS limit is 530 kt at **all** weights — weight does not
  enter the supersonic-cruise calculation at all.

## Known non-problems — do not "fix" these
- The CSVs have a UTF-8 BOM. Current pandas and numpy strip it. Leave it.
- `README.rst` is empty. Intentional for now.