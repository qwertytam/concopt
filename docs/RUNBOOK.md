# concopt runbook

Commands and steps to actually run the tool, start to finish: build the
weather data, shortlist a day, verify it, fly it, analyze it. For *why*
things work this way see `METHOD.md` (the search's reasoning) and
`docs/IMPLEMENTATION-PLAN.md` (project status, decisions, known gaps).
Everything below assumes `poetry install --extras test` has been run once
(`.venv/` in-project) and commands run from the repo root as
`poetry run concopt ...` / `poetry run python ...`.

Substitute your own route file for `<route.pln>` throughout — a P3D flight
plan, JFK→LHR, not checked into the repo (`tests/data/KJFKEGLL_CONC_01.pln`
is a test fixture only).

---

## 0. One-time setup

```
poetry install --extras test
poetry run pytest
```

## 1. Download ERA5 weather (one-time, re-run to top up)

Three datasets, each a separate CDS request stream, all resumable — every
`download_*` skips a request whose output file already exists, so
interrupting and re-running is always safe:

```
poetry run python -m concopt.era5 --full --out-dir data/era5       # upper-air (climb/cruise/arrival-upper) + surface (runway screen)
poetry run python -m concopt.era5 --subsonic --out-dir data/era5   # arrival subsonic cruise segment
```

Run both in the background (or in separate terminals) and expect them to
take **days**, not minutes — CDS queues each request for minutes, and the
CDS cost limit forces one request per calendar month (`--full`, ~146
months) or two per month (`--subsonic`, ~292 requests — it's about twice
as slow as `--full` for this reason). Capture output by hand if you want a
record:

```
poetry run python -m concopt.era5 --subsonic --out-dir data/era5 | Tee-Object -FilePath logs/subsonic_download.log
```

Before accepting requests for the first time, accept the CDS licences for
both `reanalysis-era5-pressure-levels` and `reanalysis-era5-single-levels`
at cds.climate.copernicus.eu — a one-time, per-account step; requests fail
until you do.

**Check progress** any time:

```
ls data/era5/era5_upper_*.nc | wc -l      # complete at 146 (2014-08 .. current month)
ls data/era5/era5_subsonic_*.nc | wc -l   # complete at 292 (146 months x 2 level-groups)
ls data/era5/era5_sfc_*.nc | wc -l        # complete at 25 per airport (50 total)
```

## 2. Reduce the netCDFs to per-leg .npz files

No CLI subcommand does this — `scripts/build_npz.py` wraps
`era5.reduce_to_legs`/`reduce_surface_to_npz` against whatever's on disk
in `data/era5`. Re-run it any time more months finish downloading; it just
overwrites the three/four output files.

```
poetry run python scripts/build_npz.py --pln <route.pln>
```

Produces, all in `data/era5/`:

| File | Built from | Legs |
|---|---|---|
| `route_legs.npz` | `era5_upper_*.nc` | climb+cruise legs (brake release → BARIX) — this is `--npz` |
| `subsonic_legs.npz` | `era5_subsonic_*.nc` | post-BARIX arrival legs — `--subsonic-npz` |
| `arrival_upper_legs.npz` | `era5_upper_*.nc`, reduced a *second* time | the SAME post-BARIX legs — `--arrival-upper-npz` |
| `surface_legs.npz` | `era5_sfc_*.nc` | KJFK/EGLL box-mean wind — `--surface-npz` |

`--only cruise`/`arrival`/`surface` builds a subset. If the subsonic
download hasn't caught up to the upper-air one yet, the script restricts
`subsonic_legs.npz`/`arrival_upper_legs.npz` to the months both actually
have and warns — `search._build_arrival_wind_fn` requires those two files
to share an identical time axis and raises otherwise. **Never pass
`route_legs.npz` as `--subsonic-npz`/`--arrival-upper-npz`** — it's built
from the same netCDFs but against the wrong legs; a leg-axis guard in
`search.py` (B8) catches this and raises rather than silently reading
mid-Atlantic wind for the arrival.

## 3. Search: shortlist the best departures

```
poetry run concopt search --pln <route.pln> ^
  --npz data/era5/route_legs.npz ^
  --surface-npz data/era5/surface_legs.npz ^
  --subsonic-npz data/era5/subsonic_legs.npz ^
  --arrival-upper-npz data/era5/arrival_upper_legs.npz ^
  --zfw <your ZFW in tonnes> ^
  --top 20 --out results.csv --out-all results_all.csv
```

(PowerShell line breaks: use `` ` `` instead of `^`.)

`--zfw` solves take-off weight per candidate by fixed-point iteration
(uplift = trip fuel + reserve), so a hot/heavy day carries its own extra
fuel rather than every candidate flying one shared weight. Omit it (and
`--tow`) and every candidate flies a flat 185 t — fine for a first pass,
not for a real comparison. `--out-all` is worth writing every real run —
it's what `notebooks/day-search-results.ipynb` (step 7 below) reads.

This ranks on ERA5 alone, which **cannot pick the single best day** —
verification against Active Sky showed a 2-minute ERA5 spread reordered
under real weather, one day moving from 3rd-best to worst (see
`docs/IMPLEMENTATION-PLAN.md`). Treat `results.csv`'s top 20 as a
shortlist to verify, not a final answer.

## 4. Shortlist → verify each candidate against Active Sky

```
poetry run concopt shortlist --pln <route.pln> --npz data/era5/route_legs.npz ^
  --subsonic-npz data/era5/subsonic_legs.npz --arrival-upper-npz data/era5/arrival_upper_legs.npz ^
  --search-csv results.csv --top 20
```

prints each candidate as a ready-to-run `concopt verify` command, built
with `--zfw` (read back from the search CSV's own `zfw_t` column) plus
both arrival npz paths — the same fixed-point/real-arrival model `search`
itself ranked with, not the flat `--tow` shortcut. (Older `results.csv`
files without a `zfw_t` column, or omitting `--subsonic-npz`/
`--arrival-upper-npz` here, fall back to printing a warning instead of a
command — pass the same npz's you searched with.)

Needs a **running Active Sky** with that historical date/time loaded by
hand first (one load covers every point queried — no flying required). A
snapshot guard fingerprints the queried weather and raises if it matches a
*different* date/hour's fingerprint — the signal that Active Sky wasn't
actually reloaded before this run. `--csv logs/verify_runs.csv` appends
one summary row per run so repeated verifications accumulate into
something rankable rather than living only in scrollback.

Work through the shortlist by hand: load a date in Active Sky, run
`verify`, note the recomputed total time and whether the FL choices/wind
delta look like a fixable bias or scatter, repeat. Pick the winner.

## 5. Report: the full per-leg plan for the winning day

```
poetry run concopt report --pln <route.pln> --npz data/era5/route_legs.npz ^
  --date 2026-01-21 --hour 14 ^
  --zfw <ZFW> ^
  --subsonic-npz data/era5/subsonic_legs.npz --arrival-upper-npz data/era5/arrival_upper_legs.npz ^
  --out report.csv
```

`report.csv` is what `concopt inflight --compare` (predicted vs actual)
and the post-flight notebook (step 8) read back — generate it for the day
you're actually about to fly before you fly it.

## 6. Fly it: the live in-flight advisor

Requires a running Prepar3D v5 + Active Sky (**not** historical — live
sim weather) and `python-SimConnect`. **P3D v5 caveat**:
`python-SimConnect`'s bundled `SimConnect.dll` hangs forever against P3D
v5's protocol (no exception, no timeout) — pass `--simconnect-dll`
pointing at a copy known to work (any SimConnect-speaking P3D add-on ships
one, e.g. FSLabs's `Libraries\SimConnect_P3D_v5.dll`, or Little Navmap's
install).

```
poetry run concopt inflight --pln <route.pln> ^
  --simconnect-dll "C:\path\to\SimConnect_P3D_v5.dll" ^
  --record data/inflight/recording_2026-01-21.csv ^
  --compare report.csv
```

Prints the FL450–FL600 table every `--interval` (default 60s) with the
current/recommended level and which limit binds, plus one actionable line
("CLIMB to FL530 (+109 kt) — binding: cruise_mach" / "HOLD FLxxx"),
suppressed to HOLD under `--gain-threshold-kt` (default 3). `--record`
starts logging at brake release (ground speed crossing 40 kt) through
touchdown; `--compare` runs the predicted-vs-actual comparison once
touchdown is detected. `--no-live` switches to plain scrolling prints
instead of the redraw-in-place display, useful when piping to a log file.

**Dry-run first** against a previous recording instead of the real sim:

```
poetry run concopt inflight --pln <route.pln> --replay data/inflight/some_earlier_recording.csv --replay-speed 60
```

## 7. Post-flight analysis

`notebooks/day-search-results.ipynb` — run right after step 3, needs only
`results_all.csv` (`--out-all`): distribution of total block time across
all ~31,000 candidates, wind/ISA scatter, and the winning day's FL/TAS/GS
profile.

`notebooks/postflight.ipynb` — run after step 6, needs the `--record` CSV
and the `report.csv` it was compared against: predicted vs actual overall
and per-waypoint, agreement across the three weather sources (ERA5,
Active Sky historical, live sim), what the advisor's recommendations were
worth, and whether the fixed approach allowance held up. Set
`GENERATE_SYNTHETIC = False` and fill in `RECORDING_CSV_PATH`/
`REPORT_CSV_PATH` at the top to point it at a real flight instead of the
synthetic self-check it ships with.

```
poetry run jupyter notebook notebooks/day-search-results.ipynb
poetry run jupyter notebook notebooks/postflight.ipynb
```

---

## Gotchas worth knowing before you hit them

- **Active Sky does not reload between runs unless you make it.** A
  `verify` run against a fresh date that comes back bit-identical to the
  previous run's numbers means the historical date wasn't actually
  reloaded in Active Sky — the snapshot guard (step 4) catches this and
  raises rather than silently proceeding.
- **`--tow` and `--zfw` are mutually exclusive escape hatches**, not
  interchangeable — `--tow` skips the fixed-point/real-arrival model
  entirely ("what if I actually loaded X"); `--zfw` is the real
  per-candidate model. Don't mix results computed under each.
- **`route_legs.npz` is not `subsonic_legs.npz`/`arrival_upper_legs.npz`**
  — same source netCDFs, different legs. Passing the wrong one raises
  (see step 2), which is much better than the alternative: it used to
  silently read mid-Atlantic wind for the arrival segment.
- **`git clean -xdf` will destroy `.venv/`** if run from inside the
  activated virtualenv (Windows can't unlink the locked `python.exe`, so
  git deletes everything else, including `pyvenv.cfg`). Deactivate first,
  or don't run it here at all.
- Everything above writes CSVs to the repo root by default
  (`results.csv`, `report.csv`) — `data/*.csv` is gitignored but root-level
  ones aren't; move or rename before committing anything by accident.
