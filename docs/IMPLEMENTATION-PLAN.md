# concopt — implementation plan (rev 2)

Supersedes rev 1. Current at `93a2016` (PR 19 merged).

## Objective

Two deliverables, for the FS Labs Concorde flying KJFK→EGLL in Prepar3D v5:

1. **Pre-flight.** Given the flight plan, the zero-fuel weight and twelve years of historical weather Active Sky can reproduce, shortlist the departure dates and times giving the shortest flight, and show the reasoning.
2. **In-flight.** While flying, read the aircraft state from the sim, re-run the numbers against live Active Sky weather, and recommend the flight level that minimises remaining time.

Timed brake release to touchdown. Public repo, single user, few uses — minimum viable throughout.

### One standing correction to the objective

ERA5 cannot pick the single best day. Verification against Active Sky on seven dates showed ERA5 predicting 133–135 minutes for every one — a 2-minute spread — while Active Sky recomputed 134–141 and reordered them; one day went from ERA5's 3rd-best to Active Sky's worst. The bias is systematic: 7 of 7 runs negative, mean −18.9 kt (Active Sky reports less tailwind), p = 0.016.

So the pre-flight deliverable is **ERA5 shortlists ~20 candidates, Active Sky ranks them, you fly the winner.**

---

## Where we are

| Segment | Source | State |
|---|---|---|
| Climb and acceleration to top of climb | `conc_climb.csv` — 3 ISA bands × 18 levels × TOW 130–185 t, 648 rows | **Done** |
| Supersonic cruise-climb to the decel point | `conc_supersonic_cruise.csv` + ERA5 winds, marched leg by leg | **Done** |
| Deceleration to M1.0 | `conc_descent.csv`, 3 speed schedules × 2 bands | **Done** |
| Subsonic cruise at M0.95 | `conc_subsonic_cruise.csv` — FL290–410 × mass 100–180 t × ISA ±20, 751 rows | **Done** |
| Descent and approach | `conc_descent.csv` + a fixed allowance below 1,500 ft | **Done, allowance assumed** |
| Fuel fixed point | ZFW → uplift → TOW, vectorised across ~31,000 candidates | **Done** |
| Runway and crosswind screen | 22R/31L at JFK, 09/27 at LHR, with penalties and flags | **Done** |
| Live advisor | `inflight.py` | **Written, never run, and on the old arrival model** |
| Post-flight analysis | — | **Not started** |

Roughly 700 tests. Phases A and B are closed.

### Airframe limits — validated

`atmos.py` + `limits.py` reproduce the Air France Flying Manual across **123 of 126** performance-table cells within 0.01 Mach, with no fitted parameters — but only at cruise **M2.00**, not Mmo 2.04, which matched 46/126. The three misses are one thrust-limited cell the manual itself flags and two ISA+15 rounding cases.

Two consequences that keep mattering: the Mmo knee sits at **FL511**, above which true airspeed is flat and level choice is decided purely by wind; and maximum TAS peaks at a static air temperature of **−54.8 °C**, so both colder and warmer air are slower.

### The arrival, as built

Four segments over the ~307 nm from BARIX to touchdown, at the default 380 kt schedule from FL600 in the warm band:

| | Distance | Time | Fuel |
|---|---|---|---|
| Decel to M1.0, → FL312 | 134 nm | 9.2 min | 1.23 t |
| Level at M0.95, FL312 | 113 nm | 12.2 min | 2.38 t |
| Descent to 1,500 ft | 60 nm | 7.7 min | 0.60 t |
| Approach | 5 nm | 1.5 min | 0.30 t |
| **Total** | **312 nm** | **30.6 min** | **4.51 t** |

This replaced a flat 35 minutes and 2.0 t. Level-segment distance is the residual, so the segments always close exactly on the route's own post-decel distance — no hardcoded 307.

---

## Unexpected issues along the way

Kept because several of these will recur, and several were mine.

**Data sourcing**

- **ARCO-ERA5 was the wrong source.** My first Phase 3 draft used the zarr store on Google Cloud. It is chunked one global field per timestep, so 20 route points across 25,000 hours would have transferred about a terabyte. Switched to the CDS API with server-side subsetting.
- **CDS caps pressure levels independently of items.** Discovered live: single levels, pairs, and 3–4 level combinations all passed, but 7 did not, at a box a tenth the size that had previously forced per-month chunking. Hence `SUBSONIC_LEVEL_GROUPS` splitting 7 levels into 4 + 3.
- **The manual has a typo.** At 130 t / ISA+5 it prints total fuel 21822, but 4 × 5472 = 21888, and 21888 is what reproduces the manual's own printed 53.00 NM/t specific range. Used 21888.
- **The climb table rounds two columns independently.** `mass_t` and `fuel_used_kg` disagree by up to 0.50 t at FL502 on 16 of 18 rows — ten times the fuel loop's own 0.05 t tolerance. Reading fuel from one column and mass from the other created or destroyed half a tonne at top of climb. Fixed by treating fuel as authoritative and deriving mass, because the reserve constrains fuel.

**Weather**

- **Active Sky does not reload between runs unless you make it.** Of eight `concopt verify` runs, `2016-01-06` returned wind and temperature identical to the decimal at all six points to the run before it. Only that one pair; a fingerprint guard now catches it.
- **ERA5 and Active Sky disagree systematically,** not randomly — see the standing correction above. This is why the workflow is shortlist-then-verify rather than search-and-fly.

**My own errors, since the pattern is the useful part**

- **`git clean -xdf` destroyed the venv.** I gave the command without warning to leave the activated virtualenv first; Windows could not unlink the locked `python.exe`, so git deleted everything else including `pyvenv.cfg`.
- **Three defects in the arrival work traced to my prompts.** B3 framed the wind-coverage question around where the deceleration *ends* rather than where it starts, so the decel segment silently used FL414 wind instead of FL450–490. B5 specified `level_fuel_t = level_nm / specific_range_nm_per_t`, dividing a ground distance by a still-air specific range — which made the burn wind-independent, biased *against* the tailwind days the search exists to find. B6 asked for a clamp flag scoped so broadly it fired on essentially every candidate. In each case the implementer built exactly what I wrote; in the B5 case they even left a comment observing the result was odd. The common thread is specifying expressions precise enough to be implemented verbatim without checking them dimensionally or for saturation first.
- **The subsonic specific range placeholder was wrong by a factor of two.** I assumed 24 nm/t from published figures of 21–26 t/h; the manual gives 11.1–11.6 t/h at 110 t, so 47–49 nm/t. The figures I reasoned from are low-altitude holding, not M0.95 cruise.
- **The 380 kt recommendation reversed twice** as the data improved — first when the level segment entered the picture, then again when real specific range replaced the placeholder. It now wins in every temperature band, but only the third answer was derived from complete data.
- **Giving validation rules to a transcriber makes it fabricate.** The first subsonic transcription run told each agent that TAS must satisfy the Mach identity and fuel must rise monotonically. Where the watermark obscured a digit, the agents satisfied the rules instead of reading the page — one derived the entire TAS column, another "smoothed" two fuel rows. Both files were discarded. Transcribe blind, validate separately.

**Scale**

- **Adversarial review earned its cost every time.** Three independent passes, three real defects, all invisible to the tests because the tests shared the code's assumptions.

---

## Outstanding

### Blocking a real flight

1. **`inflight.py` is still on the flat 35-minute arrival.** `DECEL_DESCENT_S` at `inflight.py:371` and `:449`. The live advisor's "predicted remaining" and "predicted total", and the post-flight comparison, all use the old constant while the pre-flight plan they are compared against uses the real model. They disagree by roughly 4–5 minutes structurally.
2. **`run_inflight` has never executed.** Its own docstring says it is not covered by the suite. It is the longest function in the project and the only one that must work on the day. The pure pieces around it are tested; the loop that stitches them is not.
3. **The recorder schema cannot answer the questions the flight is for.** It writes `zulu_s, elapsed_s, lat_deg, lon_deg, alt_ft, mach, tas_kt, gs_kt, weight_t, last_waypoint`. Missing: the wind and temperature actually encountered, the advice the advisor gave, and whether it was followed. Without those, the flight cannot measure the ERA5 → Active Sky → actual wind chain or tell you what ignoring an advisory cost. You get one flight per dataset; the schema decides what it can teach.

### Assumed, to be measured

4. `APPROACH_NM = 5.0`, `APPROACH_MIN = 1.5`, `APPROACH_FUEL_T = 0.3` below 1,500 ft.
5. The descent-segment wind clamp below FL183 — worth under 0.3 min, now visible in its own flag.

### Accepted

6. 22 of 751 subsonic cruise cells were illegible under the page watermark and are interpolated along the temperature axis, marked `illegible = True`. Worst remaining monotonicity break is 1.0%.
7. FL230/250/270 subsonic pages are scanned but not transcribed — they fly a weight-dependent M0.86–0.93 schedule the arrival never uses.
8. Supersonic accel/decel points are `--accel`/`--decel` arguments, not carried in the `.pln`.

---

## Phase C — get flight-ready without flying

The constraint shapes the order: everything needing no sim, then things needing the sim but not a 3½-hour flight, then the flight.

### C1–C4 — no sim required

| | Task | Why now |
|---|---|---|
| **C1** | Wire the arrival model into `inflight.py` | The live advisor and the plan it compares against must use the same model |
| **C2** | Extend the recorder schema | Decides what the one flight can teach; cannot be fixed afterwards |
| **C3** | Replay harness — drive `run_inflight` from a CSV instead of SimConnect + Active Sky | Turns the untested 160-line loop into something exercisable |
| **C4** | Post-flight notebook, built against synthetic recordings from C3 | Ready before there is real data to put in it |

C2 before C3, because the harness should exercise the final schema. C1 is independent.

### C5 — sim on the ground, no flight

Build the two arrival npz files, run the search end to end, produce the top 20, verify each in Active Sky. This is deliverable #1 and it needs Active Sky loaded but not a departure — roughly 20 runs of a few minutes each.

### C6 — the flight

Fly the shortlist winner with `--record`, then run the notebook. Measures `APPROACH_NM`, `APPROACH_MIN`, `APPROACH_FUEL_T` and the ERA5 → Active Sky → actual wind chain, and exercises `run_inflight` for real — by which point C3 should have removed most of the surprises.

Prompts for C1–C4 are in `PHASE-C-PROMPTS.md`.

---

## Decisions settled

| | |
|---|---|
| Cruise Mach | 2.00 (the manual's schedule; ceilings computed at it) |
| Descent schedule | 380 kt default, `auto` available |
| Crosswind | 25 kt documented limit, 30 kt cutoff, flagged between |
| Tailwind | 10 kt, no buffer |
| Runways | JFK 22R (no penalty) / 31L (+2 min); LHR 09 (no penalty) / 27 (+5 min) |
| Reserve | 10 t at touchdown — binding, not max landing weight |
| Timing | brake release to touchdown |
| Departure window | 08:00–14:00 America/New_York |
| Archive | 2014-08-01 onward, the Active Sky historical range |
| Weather for selection | ERA5 to shortlist, Active Sky to rank |
| SimConnect | python-SimConnect's bundled DLL does not speak P3D v5; pass `--simconnect-dll` (Little Navmap's install has a working one) |
