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
fuel rather than every candidate flying one shared weight. `--zfw` is
required — TOW is an outcome of it. If the sim's trip calculator gives a
different fuel load, add `--tow <ZFW + that fuel>` to fly every candidate at
that one TOW instead (candidates whose trip needs more fuel get a
`tow_below_required` flag). `--out-all` is worth writing every real run —
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
itself ranked with (plus `--tow`, from the CSV's `tow_override_t` column, if that search used one). (Older `results.csv`
files without a `zfw_t` column, or omitting `--subsonic-npz`/
`--arrival-upper-npz` here, fall back to printing a warning instead of a
command — pass the same npz's you searched with.)

Needs a **running Active Sky** with that historical date/time loaded by
hand first (one load covers every point queried — no flying required).

### What `concopt verify` actually queries

Active Sky exposes a small local HTTP API — `http://localhost:19285/ActiveSky/API/GetAtmosphere`
by default — and `verify` only ever calls that endpoint (`asky.py`); it
never touches Active Sky's own settings. **The request itself carries no
date or time**, only `lat`/`lon`/`altitudes` — Active Sky just answers with
whatever historical weather it currently has loaded. So `--date`/`--hour`
on the command line do **not** tell Active Sky what to show you; they only
(a) pick which ERA5 candidate to compare against and (b) label the
snapshot-guard cache entry below. **You** are responsible for loading that
same date/hour inside Active Sky's own UI (its History/historical-weather
mode) before running the command — if the two don't match, `verify` has no
way to know, and prints a comparison that looks entirely plausible but
means nothing.

### Running one verification

1. Pick a candidate line from `shortlist`'s output.
2. In Active Sky itself, switch to historical weather and load that same
   date/hour (e.g. 2026-01-21, 14:00 local New York — the historical mode
   is inside Active Sky's own UI, not something concopt drives).
3. Run the printed `concopt verify` command **as-is** — copy it, don't
   retype it, since it already carries the right `--zfw`/npz paths for
   that candidate.

### What it prints

- Every one of `--points` (default 12) sampled legs is queried *before*
  anything prints — the snapshot guard needs the whole batch first.
- A fingerprint (SHA-256 of every wind/temp value returned) is checked
  against a small cache, `data/verify_snapshot_cache.json`. If it exactly
  matches a *different* date/hour's cached fingerprint, `verify` raises
  rather than printing — that's the "you forgot to reload Active Sky"
  catch (confirmed live, 2026-09: a run came back bit-identical to the
  wrong-dated run just before it). Re-running the *same* date/hour again is
  fine, not an error.
- A per-point table:

  ```text
   leg      lat       lon  AS FL  ERA5 FL   AS wind  ERA5 wind   dWind   AS temp  ERA5 temp   dTemp
     2   47.213   -42.881  FL510   FL510      82.1       76.3    +5.8     -54.2      -55.1    +0.9
  ```

  `AS FL`/`ERA5 FL` are the level each source independently picks as best
  (same weight, same cruise Mach) — a mismatch is flagged with its TAS
  cost in kt (a mismatch entirely above the Mmo knee costs ~0 kt; one
  crossing into the CAS-limited part of the envelope costs real speed).
- Summary lines: mean/std wind delta with a sign count (e.g. "6 of 6
  points positive" — the signal that separates a fixable bias from
  scatter), mean/std temp delta, level-mismatch count + TAS cost, and a
  recomputed total supersonic-segment time under each source, flagged
  `OK` or `CHECK -- large divergence`.
- With `--csv logs/verify_runs.csv`, one summary row is appended per run
  so candidates accumulate into something rankable rather than living
  only in scrollback.

### If Active Sky isn't running

```text
RuntimeError: Active Sky not responding on localhost:19285; is it running with the historical date loaded?
```

`asky.py` raises this on a plain connection refusal — check Active Sky is
actually running and reachable at `--host`/`--port` (default
`localhost:19285`) before troubleshooting anything else.

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
and the post-flight notebook (step 7) read back — generate it for the day
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
synthetic self-check it ships with. `CRUISE_MACH` at the top must match
whatever `--cruise-mach` the real flight's `report`/`inflight` runs used
(see below) — every recompute in this notebook takes it, so a mismatch
here silently compares the real flight against a plan built for the wrong
cruise speed.

```
poetry run jupyter notebook notebooks/day-search-results.ipynb
poetry run jupyter notebook notebooks/postflight.ipynb
```

---

## Flying at Mmo (M2.04) instead of M2.00

`--cruise-mach` (`search`/`report`/`verify`/`inflight`) substitutes for
`limits.CRUISE_MACH` (2.00, the Air France manual's cruise schedule) in
the cruise-speed limit. Try Mmo with `--cruise-mach 2.04` on **every**
phase for one day, consistently — a mismatched value between phases makes
predicted-vs-actual comparisons meaningless, since both ERA5 and Active
Sky pick their "best level" partly based on it:

| Phase | Change |
|---|---|
| `concopt search` | add `--cruise-mach 2.04` |
| `concopt shortlist` | nothing — it only prints commands, computes nothing itself |
| `concopt verify` | add `--cruise-mach 2.04` (must match the `search` run being verified — both sides pick their best level under the same Mach) |
| `concopt report` | add `--cruise-mach 2.04` — this produces the `report.csv` that `inflight --compare` and the post-flight notebook read back, so it has to match |
| `concopt inflight` | add `--cruise-mach 2.04` — the live advisor's own recommendations use it too |
| `notebooks/day-search-results.ipynb` | set `CRUISE_MACH = 2.04` near the top (already threaded through its `march_legs`/`max_mach` calls) |
| `notebooks/postflight.ipynb` | set `CRUISE_MACH = 2.04` near the top (threaded through every call that needs it) |

**The important caveat, straight from `limits.py`'s own comment:** the
ceiling table (`conc_supersonic_cruise.csv`, `limits.ceiling_ft`) is
**not** parametrized by cruise Mach at all — it's "the altitude attainable
at M2.00," transcribed straight from the Air France manual. Raising
`--cruise-mach` to 2.04 changes only the *speed* cap (`mach_components`'s
`cruise` term); the *altitude* cap every phase still uses is the exact
same M2.00-validated table. So a `--cruise-mach 2.04` run is really asking
"how much faster could I go at the altitudes M2.00 can reach," not "what
would the aircraft actually do at Mmo" — which is exactly why the flag's
own CLI help text says "try Mmo," not "fly at Mmo."

A second, smaller gap: the arrival model's decel table (`conc_descent.csv`,
"cruise Mach → M1.0") and the climb table are both keyed by flight
level/TOW/temperature band only, never by the cruise Mach actually flown —
`arrival.arrival()` doesn't even take a `cruise_mach` argument. So the
decel segment's fuel/time always assumes decelerating from the manual's
own M2.00 baseline, regardless of `--cruise-mach`. Neither gap is tracked
as a bug in `docs/IMPLEMENTATION-PLAN.md` — `--cruise-mach 2.04` is an
exploratory lever on cruise TAS, not a re-validated Mmo flight profile.

---

## Gotchas worth knowing before you hit them

- **Active Sky does not reload between runs unless you make it.** A
  `verify` run against a fresh date that comes back bit-identical to the
  previous run's numbers means the historical date wasn't actually
  reloaded in Active Sky — the snapshot guard (step 4) catches this and
  raises rather than silently proceeding.
- **`--zfw` is required; `--tow` is an optional override on top of it.**
  `--tow` skips the fixed point ("what if I actually loaded X", e.g. the
  sim's trip-calculator fuel load) and flies every candidate at that one
  TOW, ZFW unchanged. Don't mix results computed with and without it —
  `verify` must be given the same `--tow` the search used (the shortlist
  command does this for you).
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
