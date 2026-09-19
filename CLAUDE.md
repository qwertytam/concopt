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
- `asky.py` — ActiveSky HTTP client (localhost:19285). `get_atmosphere_np`
  is the pint-free variant (plain numpy arrays), for `verify.py` and
  `inflight.py`; `get_atmosphere_as_pd` is display-only and accepts either a
  plain feet sequence or a pint Quantity. Both wrap a `ConnectionError` from
  `requests` into a `RuntimeError` naming the host/port and telling the user
  to check Active Sky is running with the historical date loaded.
  `GetAtmosphere`'s `WeatherData` is a **list of per-altitude records, every
  field a string** (confirmed live, 2026-09) -- both functions build a
  `pd.DataFrame` from that list to reshape and parse it; an earlier version
  assumed a dict-of-arrays and raised `TypeError` against a real server
  (only passed its own tests because the mocked fixture encoded the wrong
  shape -- a `verify.py`/`inflight.py` live-proof session caught it).
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

  `download_subsonic`/`download_all_subsonic` fetch the arrival-only
  subsonic cruise segment (FL183-FL414, `SUBSONIC_LEVELS`, 175-500 hPa)
  that `arrival.py`'s wind lookup needs, two files per month
  (`era5_subsonic_YYYYMM_0.nc`/`_1.nc`, `SUBSONIC_LEVEL_GROUPS`) since CDS
  caps this dataset at 4 pressure levels per request regardless of area or
  month count (confirmed by live bisection, 2026-09 — see the comment on
  `SUBSONIC_LEVEL_GROUPS`). `reduce_to_legs` needs no changes to consume
  these — confirmed by live trial, 2026-09, against synthetic files shaped
  like a multi-month × 2-group download: `open_mfdataset(combine="by_coords")`
  merges cleanly along BOTH the time and pressure_level axes into one
  dataset, no NaNs, no error. Pass every `era5_subsonic_*.nc` path (both
  groups, every month) and the *post-BARIX* legs (the complement of
  `route.climb_cruise_segment`'s mask) to `reduce_to_legs` to build the
  `--subsonic-npz` `search`/`report`/`verify --zfw` take.

  `UPPER_AIR_LEVELS` (70/100/125/150 hPa) is FL447-FL605, and
  `UPPER_AIR_AREA` already covers the arrival region — B6: `reduce_to_legs`
  run a SECOND time against those same already-downloaded `era5_upper_*.nc`
  files, but onto the *post-BARIX* legs instead of the cruise legs, builds
  `--arrival-upper-npz`, no new CDS request. 150 and 175 hPa (`SUBSONIC_LEVELS`'
  own top) are adjacent, so `search._build_arrival_wind_fn` concatenates the
  two level sets into one continuous ~FL183-FL605 profile — fixing a bug
  where the decel segment's own wind-sampling midpoint (`(cruise_fl +
  decel_end_fl) / 2`, roughly FL446-491 from a realistic FL580-600 cruise)
  fell outside `SUBSONIC_LEVELS` alone and was silently clamped to FL414's
  wind for the largest single segment of the ~307 nm arrival. Both npz's
  time axes must match (both built from `UPPER_AIR_TIMES`) — asserted,
  not assumed, since a silent mismatch there would be this same bug's
  twin. The clamp still bites at the BOTTOM: the descent segment's own
  midpoint reaches FL163.5 on the 380 kt schedule (FL182.5 on 350 kt,
  marginal; FL199 on 325 kt, clear) — faster schedules clamp, not slower
  ones, the opposite of what an earlier docstring claimed. Accepted, not
  chased further: that segment is ~60 nm of ~7.7 min and low-level wind is
  weak (well under 0.3 min of error), so covering it would cost another
  CDS download for less than `wind_at_fl`'s own `fl_clamped` flag now
  already tells us — every `wind_at_fl` call reports, per candidate,
  whether it needed the clamp, and `arrival.py` ORs the decel/descent/level
  reads into `wind_fl_clamped` (one more `_FLAG_KEYS` entry) so a clamped
  number is flagged rather than silently plausible.
- `search.py` — the day/time scan (`concopt search`). Candidates are every
  date from `era5.ARCHIVE_START` to today at 08:00-14:00 America/New_York
  (7/day), built tz-aware with `zoneinfo` and converted to UTC so DST
  doesn't silently shift winter candidates by an hour — about 30,933 for
  the real archive. `run_search` marches the climb+cruise span (brake
  release through the decel point — `route.climb_cruise_segment`, not just
  the physically-supersonic legs) one leg at a time in a plain Python loop,
  but every op inside that loop (time-interpolation into the `era5.py`
  `.npz`, vertical interpolation of u/v/t onto the FL450-FL600 1,000 ft
  grid, `limits.best_level`) is vectorised across all candidates at once —
  never loop over candidates.

  The march starts with a TOW-based climb (`data.conc_data.climb_to`,
  `conc_climb.csv`; `--tow`, default 185 t): brake release to top of climb
  (FL502) burns TOW down to a top-of-climb mass over some ground distance
  and time, both used as the cruise's starting state — replacing an older
  fixed 20-minute/165 t accel-point assumption that credited full cruise
  ground speed over hundreds of miles still spent climbing. The table's
  temp band (`isa_minus_20_to_minus_10`/`isa_minus_10_to_isa`/
  `isa_to_isa_plus_10`, discrete, never interpolated between) is picked
  from the ISA deviation over the first `CLIMB_BAND_SAMPLE_NM` (300) nm of
  route at the lowest stored ERA5 pressure level — the upper-air `.npz`
  only carries the 4 mandatory levels used for the supersonic scan, nothing
  low enough for a real climb-altitude reading, so this is a coarse proxy,
  good enough to bucket the day and correct the table's air distance for
  wind (`ground_dist_nm = dist_nm + wind_component_kt * time_min / 60`,
  the same relation `conc_descent.csv` tabulates explicitly for descent).
  Days colder than ISA-20 clamp into the coldest band silently; days
  warmer than ISA+10 clamp into the warmest band and are flagged
  (`climb_warm_clamped`/`flags` column) — clamping warm is optimistic.

  Since the climb's ground distance varies per candidate (TOW is fixed for
  a run, but temp band and the wind correction vary day to day), so does
  where the cruise starts along the route — a leg wholly inside one
  candidate's climb is wholly past another's. Rather than reconstructing
  per-candidate Leg objects, `march_legs` computes `eff_dist_nm` (n_cand,
  n_legs): the actual ground distance each candidate flew each leg, 0 for
  one wholly consumed by climb, the full leg for one wholly past it,
  a partial amount for the one leg straddling top of climb. `mean_fl`/
  `mean_wind_kt`/`mean_isa_dev_k` are `eff_dist_nm`-weighted means, so the
  climb's leading zero/partial-distance legs (whose chosen level is
  climb-altitude noise — `limits.best_level` still runs on them regardless)
  don't pollute the cruise-only averages.

  Weight is a state variable, not a schedule: after the climb it's
  integrated leg by leg from `conc_data.fuel_total_kgh_table` (the Air
  France performance table), at the weight/ISA-deviation the leg was
  actually flown at, feeding `limits.ceiling_ft(weight_t, isa_dev_c)` for
  the next leg — a still-air, ISA+0 run at TOW 185 t burns to a ~149 t
  top-of-climb mass, then to ~115 t by BARIX (see the printed sanity
  checks, which report rather than assert). Assumes the `.npz` was built
  from the same `--pln`/`build_legs` call, so the leg axes line up by
  position — it does not re-check this. Also screens each candidate's KJFK
  departure / EGLL arrival wind through `runways.py` (see below) and ranks
  on `total_time`, not supersonic time. `--out-all` writes the *full*
  ranked candidate set (raw numeric columns, ~31,000 rows) alongside
  `--out`'s top-N formatted display CSV — for `notebooks/day-search-results.ipynb`,
  which needs the whole distribution rather than just the top rows. Every
  row is stamped with its own `tow_t` (the `--tow` that run was made
  under), read back by `run_shortlist`/`concopt shortlist` (below) so a
  generated `concopt verify` command always uses the TOW that candidate
  was actually found under, not a guessed default.

  `run_shortlist`/`concopt shortlist` takes a `concopt search --out` CSV
  (`--search-csv`, default `results.csv`) and prints its top `--top`
  (default 10) rows as ready-to-run `concopt verify` command lines (date/
  hour/tow filled in from that row) — for working through a shortlist by
  hand (load the date in Active Sky, paste the command, repeat) without
  re-typing date/hour/tow each time.

  `resolve_tow_and_arrival` (shared by `run_search`, `report.run_report`,
  and `verify.run_verify` — see those below) is the one place TOW-solving
  and the arrival segment (BARIX → touchdown, `arrival.py`) are wired
  together: arrival fuel has to be *inside* `fuel.fixed_point_fuel_iteration`'s
  loop, not added after it returns, because the whole point of `arrival.py`
  is that its ~4-8 t (vs the old flat `DESCENT_FUEL_T` 2.0 t placeholder)
  feeds back into TOW and therefore into climb time. `_build_arrival_wind_fn`
  builds the `wind_at_fl` callable `arrival.arrival()` needs from TWO ERA5
  `.npz`s stitched together (B6, see `era5.py` above) — `--subsonic-npz`
  and `--arrival-upper-npz`, both `era5.reduce_to_legs` run against the
  post-BARIX legs (the complement of `route.climb_cruise_segment`'s mask,
  i.e. `~mask`), both required together unless `--decel-descent-min`
  forces the flat legacy arrival — sampled at a single representative
  arrival leg and at BARIX clock time, the same coarse single-point-proxy
  convention `_climb_conditions` uses for the climb, not a per-segment
  march. `--decel-descent-min` no longer has a default value:
  given, it forces `arrival.flat_arrival` (the exact pre-arrival.py flat
  `DECEL_DESCENT_S`/`DESCENT_FUEL_T` pair) instead of the real per-day
  model, ignoring `--subsonic-npz` entirely — for comparing old vs new
  numbers directly. Neither given (the default) requires `--subsonic-npz`;
  omitting both raises rather than silently falling back to something flat.
- `arrival.py` — the arrival segment, BARIX → touchdown, replacing the old
  flat `DECEL_DESCENT_S`/`DESCENT_FUEL_T` placeholder search/report used to
  carry. Four segments over a *route-provided* `arrival_nm` (summed
  post-decel leg distance, not a hardcoded constant — search/report compute
  it as `legs[-1].cum_nm - cc_legs[-1].cum_nm`): decel to Mach 1
  (`data.conc_data.decel_to_mach1`), level cruise at M0.95 for whatever
  distance is left over (`data.conc_data.subsonic_cruise`, B5 — see below),
  descent to 1,500 ft (`data.conc_data.descent_to_1500ft`), then a fixed
  approach allowance (`APPROACH_NM`/`APPROACH_MIN`/`APPROACH_FUEL_T`).
  `descent_direct_from_cruise` is deliberately unused — it reaches 1,500 ft
  in ~194 nm and this route has ~307, which would leave ~113 nm
  unaccounted for. All three descent speed schedules (325/350/380 kt) are
  evaluated as whole-array table lookups; `speed=380` (the default since
  B5) forces the fastest schedule — with the real subsonic table it wins
  on both time AND fuel in every temperature band (buys ~1.6 min for
  ~0.44 t, not the old placeholder's ~1.4 t), so the old cold/warm climb-
  time trade-off that used to make `speed="auto"` the default is no longer
  close. BUT 380 kt's decel_end_fl (FL312) only has subsonic-table coverage
  down to 110 t, which real BARIX mass undercuts for a good chunk of the
  160-185 t TOW range — so real callers (`fuel.py`'s
  `_arrival_from_march`, the only production call site) explicitly request
  `speed="auto"` instead, which now disqualifies a NaN-fuel schedule before
  picking by time and so falls back to 350/325 kt (100 t floors) rather
  than handing back a NaN trip fuel; `by_schedule` still exposes the full
  per-schedule breakdown for a future caller-side re-optimization on
  *total* time. Temperature band selection uses only `conc_descent.csv`'s
  two bands (`above_isa_minus_10`/`isa_minus_10_and_below` — not the climb
  table's three), from the ISA deviation at cruise level. Wind/temperature
  arrive through a caller-supplied `wind_at_fl` callable/dict — this module
  never reads era5/`.npz` files directly, so the ERA5 wiring stays entirely
  in `search.py` (`_build_arrival_wind_fn`). `flat_arrival` is the
  `--decel-descent-min` legacy override: same call signature as `arrival()`
  so `fuel.py`'s fixed point can hold either interchangeably, but returns
  the flat pre-B3 (time, fuel) pair with `schedule_kt=0` as a sentinel.

  MASS IS THE TRAP (B5): the level segment's fuel now comes from
  `conc_data.subsonic_cruise` (`conc_subsonic_cruise.csv`, 751 rows,
  trilinear over level_fl/mass_t/isa_dev_c, ragged — higher levels only
  published down to a higher minimum mass, FL410 to 125 t, FL290-330 to
  110 t, FL350+ to 100 t — dense-gridded with NaN in the gaps, never
  filled; a query needing a NaN corner returns NaN, but a corner reached
  with exactly zero interpolation weight — an exact grid hit, or an axis
  clamp — never touches a NaN neighbour it doesn't actually need. tas_kt is
  computed from `atmos.py` directly rather than read off the table, since
  it doesn't depend on mass at all and matches the transcribed column to
  within 0.5 kt). `arrival()` takes `mass_at_barix_t` (the march's
  `weight_at_barix`, BEFORE the decel burn); `_arrival_for_speed`
  subtracts that schedule's own `decel_fuel_t` internally to get the mass
  actually flying the level segment (roughly 105-120 t, not TOW's
  160-185 t) before indexing the table — reading it at the wrong mass
  roughly halves the specific range and doubles the level fuel, and the
  result looks entirely plausible. `level_mass_outside_envelope` flags a
  NaN level_fuel_t (zero leftover distance costs no fuel regardless); at
  real arrival masses for the 350/325 kt schedules this essentially never
  fires, but 380 kt alone genuinely can, for the TOW-range reason above.
  `fuel.fixed_point_fuel_iteration`'s `INITIAL_TOW_T=150` trial is light
  enough that even the 350/325 kt floor can be undercut on the very first
  pass, before the loop has seen any real trip fuel — a NaN trip fuel
  there would otherwise `np.clip` to NaN forever (clip does not resolve
  NaN), so a NaN `tow_calc` is nudged to `MTOW_T` before clamping, self-
  correcting once the march lands back inside the table.

  `wind_fl_clamped` (B6): `_wind_temp` reads an optional `fl_clamped`
  key/tuple-slot out of whatever `wind_at_fl` returns (missing means never
  clamped, so every pre-B6 test mock stays valid) and `_arrival_for_speed`
  ORs the decel/descent/level segments' three reads into one flag — see
  `era5.py`/`search.py` above for what actually sets it
  (`search._build_arrival_wind_fn`'s stitched FL183-FL605 profile still
  clamps below FL183 on the descent segment's own midpoint).
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
  dropped. Touchdown clock time (for sampling EGLL's arrival wind here) is
  each candidate's own `arrival.arrival()` output (`arrival_time_s`,
  `search.run_search`), not a flat constant any more — `search.DECEL_DESCENT_S`
  (35 min) survives only as the legacy value `--decel-descent-min` forces
  and the constant `inflight.py`'s flight recorder still compares measured
  time against (that wiring is a separate, later task).
- `verify.py` — Phase 5, `concopt verify`. The user loads a historical date/
  time in Active Sky by hand first (a static snapshot of its global weather
  model — the API takes an explicit lat/lon/altitude, so one load covers
  every point queried below, no flying required). `--zfw` (tonnes) runs the
  SAME fixed point `search`/`report` use (`search.resolve_tow_and_arrival`,
  requires `--subsonic-npz` AND `--arrival-upper-npz` too (B6) — arrival
  fuel needs the same stitched post-decel wind source search used, or the
  two TOWs won't agree), so the
  verification is flown at the weight that day was actually found under —
  an Active Sky check flown at the wrong weight undercuts the whole
  comparison. `--tow` overrides `--zfw` and skips the fixed point (and
  arrival/subsonic data) entirely, same "what if I actually load X" escape
  hatch `search`/`report` have; legs still inside the climb are excluded
  from the comparison (Active Sky's FL450-FL600 `TARGET_FL` grid doesn't
  apply to climb altitude). Takes `--points` (default 12) evenly spaced cruise
  legs, including the first and last; at each one queries Active Sky live
  for the FL450-FL600 `TARGET_FL` grid and compares against the ERA5 values
  `search.march_legs` would have used at that same point and clock time
  (reuses `march_legs`, never reimplements its interpolation). Both sources
  pick their best level with the *same* weight (the ERA5 march's
  `weight_per_leg`) and cruise Mach, so a level disagreement between them
  reflects a genuine wind/temp difference, not a weight mismatch.

  SNAPSHOT GUARD: every point is queried *before* anything is printed;
  the whole run's wind+temp is then fingerprinted (SHA-256 over the
  concatenated arrays, `_atmosphere_fingerprint`) and checked against a
  small `{"<date> <hour>:00": "<fingerprint>"}` cache at
  `data/verify_snapshot_cache.json` (gitignored, `_guard_snapshot`). A
  fingerprint match against a **different** date/hour raises rather than
  proceeding — confirmed live, 2026-09: a run against 2016-01-06 came back
  bit-identical, at every point then in use, to the 2016-02-12 run just
  before it, because Active Sky hadn't actually been reloaded. This is the
  single most valuable check here: that failure is silent, plausible, and
  produces numbers that look fine. A repeat run of the *same* date/hour is
  not an error (re-verifying without reloading is legitimate).

  The per-point table reports the TAS cost of a level mismatch
  (`_mismatch_tas_cost_kt`, both levels' `limits.max_tas` evaluated under
  Active Sky's own temperature) rather than just flagging it — a mismatch
  entirely above the Mmo knee (cruise_mach alone binding, TAS flat with
  altitude) costs ~0 kt, while one that crosses into the CAS-limited lower
  envelope costs real speed; a raw mismatch *count* conflates the two. The
  wind-delta summary also reports the sign count (e.g. "6 of 6 points
  positive") alongside mean/std — a consistent sign across points is the
  signal that separates a fixable bias from unfixable scatter, which the
  mean alone can obscure. `--csv` appends one summary row (mean/std wind
  and temp delta, wind sign count, ERA5/AS total minutes) per run to a
  given CSV, so repeated runs accumulate into something rankable instead
  of living only in scrollback.

  The recomputed "AS total time" extends the `--points` ground speeds
  across every sub-leg by nearest-point assignment — an eyeball
  approximation, not a full AS march (which would need Active Sky queried
  at every sub-leg). Needs a live, running Active Sky; not covered by the
  test suite (which mocks `asky.get_atmosphere_np`) beyond the snapshot
  guard's end-to-end test (a synthetic still-air `.npz`, no real Active
  Sky) — run it by hand against the top few `concopt search`/`concopt
  shortlist` days and eyeball whether the ranking survives, and whether
  AS/ERA5 divergence looks like a fixable constant bias or unfixable
  scatter.
- `inflight.py` — Phase 6, `concopt inflight`. Live advisor + flight
  recorder against a running Prepar3D + Active Sky, over SimConnect
  (`python-SimConnect`, localhost). Every `--interval` seconds: reads the
  sim, projects `--lookahead-nm` ahead along the loaded route
  (`route.project_along_route`), queries Active Sky there via
  `verify._as_atmosphere` (reused, not reimplemented), and feeds the result
  to `limits.best_level` **unchanged** -- same ceiling, same limits, same
  code `search.march_legs` uses. `TOTAL WEIGHT` is read live and used
  directly; there is no burn-off schedule here (contrast `search.py`'s
  state-variable weight). Prints the FL450-FL600 table with the current and
  recommended levels marked and which limit binds at the recommendation,
  then one actionable line ("CLIMB to FL530 (+109 kt, ...) -- binding:
  cruise_mach" / "HOLD FLxxx"), suppressed to a HOLD when the gain is under
  `--gain-threshold-kt` (default 3). With `--record`, runs a small state
  machine alongside it (brake release, detected as on-ground ground speed
  crossing 40 kt upward, through touchdown) writing one CSV row per interval
  and tracking each `.pln` waypoint's closest point of approach as it goes;
  elapsed time is measured off `time.monotonic()`, not `ZULU_TIME` (which
  wraps at 86,400 s and a flight can span midnight UTC). With `--compare`
  (a `concopt report --out` CSV, requires `--record`), runs
  `compare_to_report` at touchdown -- predicted vs actual per waypoint, plus
  the three numbers the recorder exists to measure: measured brake-release
  -> accel point vs that report's own predicted elapsed time there
  (`search.py`'s TOW-based climb model makes this vary by day/TOW, so it's
  read back from the report rather than a fixed constant), measured decel
  point -> touchdown vs `DECEL_DESCENT_S` (35 min — still a flat constant
  here specifically; wiring this comparison to `report`'s own per-day
  `arrival.arrival()` output is a separate, later task), and measured vs
  predicted supersonic segment time. `_level_table`/`_recommendation_line`/
  `compare_to_report` are pure and unit-tested; `run_inflight` itself needs
  a live sim and isn't (same convention as `search.run_search`/
  `verify.run_verify`).

  SimConnect caveat, proven live against this project's P3D v5 install
  (2026-09): `python-SimConnect`'s own bundled `SimConnect.dll` does not
  speak P3D v5's protocol -- `SimConnect()` doesn't raise, it **hangs
  forever**, because `SimConnect.connect()` only breaks its
  `while self.ok is False: pass` spin-wait on an OPEN event, and a version
  mismatch gets a `SIMCONNECT_EXCEPTION` back instead of OPEN, silently,
  with no timeout. Fix: `--simconnect-dll` pointing at a copy already known
  to work with the running sim -- any P3D add-on that talks SimConnect
  ships one (this project's dev machine used FSLabs's
  `Libraries\SimConnect_P3D_v5.dll`; Little Navmap's install has one too).
  Confirmed live: `PLANE_LATITUDE` read back correctly with that dll, hung
  indefinitely with the bundled one.

- `data/` — CSV limit tables + `conc_data.py` loader
- `condition.py`, `atmosphere.py`, `common.py`, `airframeflows.py`,
  `nondimensional.py` — vendored fork of the `flightcondition` package.
  **Do not read or modify these.** Legacy; retained only for the pretty
  `tostring()` output in the future in-flight display.
- `notebooks/day-search-results.ipynb` — exploratory reporting on a `concopt
  search --out-all` run: distribution of total block time (histogram +
  top-50 marked, box plot by month, wind/ISA-deviation scatter), a
  formatted top-10 table, and the winning day's profile (chosen FL vs
  `ceiling_ft`, TAS/GS/along-track wind vs distance) plus a flag-count
  summary. Loads only; every computed value reuses concopt's own
  functions (`search.march_legs` rerun for the single winning candidate —
  the same call `report.run_report` makes — `report._step_climb_schedule`,
  `limits.ceiling_ft`), it does not reimplement the march. The still-air
  ISA+0 reference figure feeds `march_legs` a synthetic zero-wind
  atmosphere rather than hand-computing a time, exploiting FL450-FL600
  sitting entirely inside the ISA's 11-20 km isothermal layer (`atmos.isa`)
  so one constant temperature is exact at every level, no pressure->
  altitude inversion needed. Needs `matplotlib` (added as a dependency for
  this).

## Conventions
- SI internally (K, Pa, m/s, m). Convert at the edges only.
- Everything array-in / array-out. No `iterrows()`, no per-row Python loops
  in anything that touches the search — it runs over ~31,000 candidate
  departures × ~32 supersonic legs × 4 ERA5 pressure levels.
- pint is allowed **only** in display code, never in `atmos.py`/`limits.py`.
- Ad hoc command output captured by hand (e.g. `concopt verify ... |
  Tee-Object -FilePath logs/verify_2016-02-12.log`) goes in `logs/`, not
  the repo root -- gitignored, `logs/.gitkeep` keeps the empty folder
  tracked.

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
- Climb table: `data/conc_climb.csv` (`conc_data.climb_to`), brake release
  to top of climb (FL502), temp band (3 discrete bands, never interpolated
  between) × TOW t (160-185, 5 t steps) × level_fl (18 levels) — 324 rows.
  `dist_nm` is air distance (no wind columns, unlike the descent tables);
  `search.py`'s `_climb_profile` applies the wind correction. Top of climb
  ranges 210-1047 nm depending on TOW/temperature — see `search.py`'s
  climb-model paragraph above.
- Descent/arrival table: `data/conc_descent.csv` (`conc_data.decel_to_mach1`/
  `descent_to_1500ft`/`descent_direct_from_cruise`/`dist_with_wind`),
  wired into `arrival.py` (see above) — speed schedule (325/350/380 kt) ×
  temp band (2 discrete bands, `above_isa_minus_10`/`isa_minus_10_and_below`
  — NOT the climb table's three) × level_fl, linearly interpolated within
  each `(table, from_supersonic_cruise, speed_kt, temp_band)` group — 291
  rows, clamped to each group's own level_fl bounds (never a shared global
  bound — an early bug clamped to the wrong group's range). `decel_end_fl`/
  `level_correction_nm_per_2000ft` are scalar constants per group, not
  interpolated. `dist_with_wind` applies the same
  `ground_nm = zero_wind_nm + wind_kt * time_min / 60` relation the climb
  table's wind correction uses, verified against all 582 printed table
  endpoints to within 1 nm.

## Known non-problems — do not "fix" these
- The CSVs have a UTF-8 BOM. Current pandas and numpy strip it. Leave it.
- `README.rst` is empty. Intentional for now.