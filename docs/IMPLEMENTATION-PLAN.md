# concopt — implementation plan (rev 3)

Current at `ea65605` (PR 21 merged). 428 tests, ~88 s.

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
| Climb to top of climb | `conc_climb.csv` — 648 rows, TOW 130–185 t | **Done** |
| Supersonic cruise-climb | `conc_supersonic_cruise.csv` + ERA5, marched leg by leg | **Done** |
| Decel to M1.0 | `conc_descent.csv`, 3 schedules × 2 bands | **Done** |
| Subsonic cruise at M0.95 | `conc_subsonic_cruise.csv` — 751 rows | **Done** |
| Descent and approach | `conc_descent.csv` + assumed allowance below 1,500 ft | **Done, allowance assumed** |
| Fuel fixed point | ZFW → uplift → TOW, vectorised across ~31,000 candidates | **Done** |
| Runway and crosswind screen | 22R/31L at JFK, 09/27 at LHR | **Done** |
| Live advisor | `inflight.py` + real arrival model | **Done, never run against a sim** |
| Flight recorder | schema v2, 41 columns | **Done** |
| Replay harness | `replay.py` — recorded or synthetic | **Done** |
| Post-flight notebook | `notebooks/postflight.ipynb` | **Done, proven on synthetic data** |
| Cross-candidate analysis | — | **Not started — see below** |

Phases A, B and C1–C4 are closed. What remains before a flight is one small fix, one known gap, and the sim itself.

### The arrival, as built

Four segments over the ~307 nm from BARIX to touchdown, at the default 380 kt schedule from FL600, warm band:

| | Distance | Time | Fuel |
|---|---|---|---|
| Decel to M1.0, → FL312 | 134 nm | 9.2 min | 1.23 t |
| Level at M0.95, FL312 | 113 nm | 12.2 min | 2.38 t |
| Descent to 1,500 ft | 60 nm | 7.7 min | 0.60 t |
| Approach | 5 nm | 1.5 min | 0.30 t |
| **Total** | **312 nm** | **30.6 min** | **4.51 t** |

Replaced a flat 35 min / 2.0 t. Level distance is the residual, so segments always close on the route's own post-decel distance.

---

## PR 20 / 21 review

**The replay harness paid for itself before the sim was ever switched on.** Four bugs, all in `run_inflight` — the one function that had never executed — and two of them would have wrecked the flight:

| Bug | How it would have shown up in the air |
|---|---|
| Brake-release detection only fired on a *crossing* of 40 kt. A first reading already above it — SimConnect connecting late, or a replay starting mid-roll — left no crossing to observe | **The recorder silently never starts.** An all-header CSV, no error, no data, flight wasted |
| `_format_hmm_or_na` handled `None` but not `NaN`, and `nan is not None` is True | **The live session crashes** on the next redraw, mid-flight |
| `--replay-speed` reached `run_inflight` but not `replay_sources()` | Clocks race apart; only visible through a CLI smoke test, not the unit tests |
| Division by zero in `wind_at_fl` if two sampled arrival levels return the same pressure | Silent NaN — **documented, not fixed** |

The first two are exactly the class of failure that only appears on first execution, and the first one is the worst kind: no exception, no output, and you don't find out until you land.

**C4 turned up a methodological trap worth keeping.** Cutting the actual flight into segments by the recorder's `phase` column gave a ~1,900 s climb/cruise "error" that was purely an artefact — `phase` cuts climb/cruise at the accel waypoint, a named .pln fix, while the model cuts it at `conc_climb.csv`'s own top-of-climb distance. Cutting by route distance instead made segment deltas small and total error near zero. A disagreement about where a boundary sits looks exactly like a model error.

Synthetic validation: total within −0.36% time / +0.06% fuel; the approach constants measure 5.03 nm / 1.48 min / 0.282 t against the assumed 5.0 / 1.5 / 0.3. Both are tautological on a synthetic flight — they prove the measurement path works, not the model.

### Two issues found in this review

1. **FIXED, PR #22.** ~~`run_shortlist` generates verify commands that can't reproduce the search.~~ `search.py:891` emitted `--tow` with neither `--subsonic-npz` nor `--arrival-upper-npz`. The `--tow` path in `verify.py` skips the fixed point and does a plain `march_legs`, so the arrival it computed was not the model the search ranked with. Every one of the 20 verifications would have compared against a differently-built arrival. `run_search` now stamps each row with its own `zfw_t` alongside `tow_t`, and `run_shortlist`/`concopt shortlist` reads `zfw_t` back and builds the verify command with `--zfw` plus both npz paths (new `--subsonic-npz`/`--arrival-upper-npz` flags) instead of `--tow`. **Unblocks C5.**

2. **FIXED, PR #22.** ~~The known NaN gap is still open.~~ `inflight.py:417-428` documented it honestly: `wind_at_fl` divided by `src_log_p[idx1] - src_log_p[idx0]`, silently zero if two of the ten sampled arrival levels returned the same pressure. Adjacent duplicate pressure levels are now collapsed before the interpolation source arrays are built; if fewer than two distinct pressures survive, the function returns `None` (the same "Active Sky didn't answer" signal already in use) rather than a silent NaN, and the live panel's arrival line now carries an explicit `[LIVE]`/`[PRE-FLIGHT]` marker.

---

## Unexpected issues along the way

Kept because several will recur, and several were mine.

**Data sourcing**

- **ARCO-ERA5 was the wrong source.** Chunked one global field per timestep — 20 route points across 25,000 hours would have moved about a terabyte. Switched to the CDS API with server-side subsetting.
- **CDS caps pressure levels independently of items.** Singles, pairs and 3–4 level combinations passed; 7 did not, at a box a tenth the size that had previously forced per-month chunking. Hence the 4 + 3 split.
- **The manual has a typo.** 130 t / ISA+5 prints total fuel 21822, but 4 × 5472 = 21888, and 21888 reproduces the manual's own 53.00 NM/t. Used 21888.
- **The climb table rounds two columns independently.** `mass_t` and `fuel_used_kg` disagree by up to 0.50 t at FL502 on 16 of 18 rows — ten times the fuel loop's own tolerance. Fixed by treating fuel as authoritative and deriving mass, since the reserve constrains fuel.

**Weather**

- **Active Sky does not reload between runs unless you make it.** One of eight verify runs returned wind and temperature identical to the decimal to the run before it. A fingerprint guard now catches it.
- **ERA5 and Active Sky disagree systematically**, not randomly — hence shortlist-then-verify.

**My own errors, since the pattern is the useful part**

- **`git clean -xdf` destroyed the venv.** Given without warning to leave the activated virtualenv first; Windows could not unlink the locked `python.exe`, so git deleted everything else including `pyvenv.cfg`.
- **Three arrival defects traced to my prompts.** B3 framed the wind-coverage question around where the deceleration *ends* rather than where it starts, so the decel segment used FL414 wind instead of FL450–490. B5 specified `level_nm / specific_range_nm_per_t`, dividing a ground distance by a still-air specific range — making the burn wind-independent and biased *against* the tailwind days the search exists to find. B6 scoped a clamp flag so broadly it fired on every candidate. Each time the implementer built exactly what I wrote; in the B5 case they left a comment observing the result was odd. The common thread: specifying expressions precise enough to implement verbatim without checking them dimensionally or for saturation first.
- **The subsonic specific range placeholder was wrong by 2×.** I assumed 24 nm/t from published 21–26 t/h figures; the manual gives 11.1–11.6 t/h at 110 t, so 47–49 nm/t. Those published figures are low-altitude holding, not M0.95 cruise.
- **The 380 kt recommendation reversed twice** as data improved. Only the third answer came from complete data.
- **Giving validation rules to a transcriber makes it fabricate.** Told that TAS must satisfy the Mach identity and fuel must rise monotonically, agents satisfied the rules instead of reading the page where the watermark obscured a digit — one derived the whole TAS column, another "smoothed" two fuel rows. Both files discarded. Transcribe blind, validate separately.

**Method**

- **Adversarial review earned its cost every time.** Three passes, three real defects, all invisible to the tests because the tests shared the code's assumptions.
- **First execution finds what review cannot.** The replay harness found four bugs in an hour that three review passes over the same file had not, because they were failures of the loop rather than of the logic.

---

## Outstanding

### Blocking

1. **ZFW.** Still unset, and it drives uplift, TOW, climb time and therefore the entire ranking. Nothing downstream is real until it is chosen.

(`run_shortlist` command generation and the live-arrival NaN gap, both formerly listed here, are fixed — PR #22, see above.)

### Assumed, to be measured by the flight

2. `APPROACH_NM = 5.0`, `APPROACH_MIN = 1.5`, `APPROACH_FUEL_T = 0.3` below 1,500 ft.
3. The descent-segment wind clamp below FL183 — under 0.3 min, visible in its own flag.

### Accepted

4. 22 of 751 subsonic cruise cells interpolated under the page watermark, marked `illegible = True`. Worst monotonicity break 1.0%.
5. FL230/250/270 subsonic pages scanned but not transcribed — a weight-dependent M0.86–0.93 schedule the arrival never uses.
6. Accel/decel points are CLI arguments, not carried in the `.pln`.

### Gap against the objective

7. **Nothing shows the reasoning across candidates.** The objective asks to identify the best departure *and show the reasoning behind the selection*. `report` explains one day in detail; `shortlist` prints twenty commands. Nothing explains why the winners win, what separates rank 1 from rank 50, how much is seasonal, or how sensitive the answer is to ZFW. That is half of deliverable #1 and it needs no sim.

---

## Phase D — while the sim is unavailable

| | Task | Needs |
|---|---|---|
| **D1** | ~~Fix `run_shortlist` to emit `--zfw` and both npz paths~~ **Done, PR #22** | nothing |
| **D2** | ~~Close the NaN gap in the live arrival wind~~ **Done, PR #22** | nothing |
| **D3** | Build the arrival npz files and run the ERA5 search end to end — produce the top 20 | ERA5 data only |
| **D4** | Cross-candidate analysis notebook — the reasoning half of deliverable #1, plus a ZFW sensitivity sweep | D3's output |

**D3 is reachable now.** Only the *verification* step needs Active Sky; the ERA5 search itself needs nothing but the downloaded data. That produces the shortlist, which is half the pre-flight deliverable.

**D4 answers the ZFW question empirically** rather than waiting for a decision: sweep ZFW across a plausible range and see whether the top-20 set actually changes. If it barely moves, the open item stops blocking.

Then, when the sim is available: verify the shortlist in Active Sky (needs Active Sky loaded, not a departure), then fly the winner with `--record` and run the notebook.

Prompts in `PHASE-D-PROMPTS.md`.

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
| Recorder | schema v2, 41 columns, missing readings written empty never as a sentinel |
