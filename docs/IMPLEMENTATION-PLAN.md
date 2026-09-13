# concopt — implementation plan

## Objective

Two deliverables, for the FS Labs Concorde flying KJFK→EGLL in Prepar3D v5:

1. **Pre-flight.** Given the flight plan, the zero-fuel weight and twelve years of historical weather Active Sky can reproduce, shortlist the departure dates and times that give the shortest flight, and show the reasoning behind the choice.
2. **In-flight.** While flying, read the aircraft state from the sim, re-run the numbers against live Active Sky weather, and recommend the flight level that minimises remaining time.

Timed **brake release for the take-off roll to touchdown**. Public repo, single user, few uses — minimum viable throughout, no abstraction for its own sake.

### One correction to the objective

ERA5 cannot pick the single best day. Verification against Active Sky on seven dates showed ERA5 predicting 133–135 minutes for every one of them — a 2-minute spread — while Active Sky recomputed 134–141 and reordered them; one day went from ERA5's 3rd-best to Active Sky's worst.

So the pre-flight deliverable is: **ERA5 shortlists ~20 candidates, Active Sky ranks them, you fly the winner.** The search's job is reducing 30,933 candidates to a couple of dozen worth checking by hand.

---

## The flight model

Five segments, brake release to touchdown:

| Segment | Source | State |
|---|---|---|
| Climb and acceleration to top of climb | `conc_climb.csv` — 3 ISA bands × 18 levels × TOW 160–185 t | **Built** |
| Supersonic cruise-climb to the decel point | `conc_supersonic_cruise.csv` + ERA5 winds, marched leg by leg | **Built** |
| Deceleration and descent to subsonic cruise | `conc_descent.csv` — 3 speed schedules × 2 temp bands | **Data present, unwired** |
| Subsonic cruise at M0.95 | needs ERA5 FL310–FL420 | **Not built, data missing** |
| Descent and landing | `conc_descent.csv` + a fixed approach allowance | **Not built** |

Plus the runway and crosswind screen at both ends, which selects 22R/31L at JFK and 09/27 at LHR, applies the 2- and 5-minute penalties, and flags days over the documented 25 kt crosswind limit.

### Airframe limits

All validated against the Air France Flying Manual — 123 of 126 performance-table cells reproduce within 0.01 Mach with no fitted parameters.

- Cruise Mach 2.00 (`CRUISE_MACH`; Mmo 2.04 available via `--cruise-mach`)
- CAS placard 530 kt above FL430, identical at every weight
- Total temperature 127 °C
- Achievable FL from the ceiling table, a 2-D function of weight and ISA deviation

Two consequences worth keeping in mind: the Mmo knee sits at **FL511**, above which true airspeed is flat and the level choice is decided purely by wind; and maximum TAS peaks at a static air temperature of **−54.8 °C**, so both colder and warmer air are slower.

---

## Weight and fuel — the change that matters most

Take-off weight is currently a command-line input defaulting to 185 t. It should be **derived**, because TOW = ZFW + fuel uplift, and the uplift is a decision rather than a given.

**Minimising uplift minimises flight time**, and the effect is large. Time and fuel to top of climb:

| Band | TOW 185 t | TOW 173 t | TOW 164 t |
|---|---|---|---|
| ISA−20 to −10 | 26 min / 18.2 t | 21 min / 15.3 t | 19 min / 13.6 t |
| ISA−10 to ISA | 41 min / 25.1 t | 27 min / 17.9 t | 22 min / 15.4 t |
| ISA to ISA+10 | **65 min / 35.7 t** | 43 min / 25.0 t | **31 min / 19.6 t** |

Departing at max take-off weight costs **5 to 34 minutes** of climb depending on temperature and how much fuel is genuinely needed. On a warm day at 185 t the climb alone burns 35.7 t — more than half the trip fuel — which is the compounding that makes this worth solving properly rather than guessing.

### The fuel fixed point

```
uplift  →  TOW = ZFW + uplift  →  fly the model  →  trip fuel
                    ↑                                   │
                    └──── uplift = trip + reserve ←──────┘
```

Contractive (the marginal fuel-to-carry-fuel is well under 1), so it converges in a handful of passes.

- **Reserve: 10 t remaining at touchdown.** This is the binding constraint, not maximum landing weight — landing weight comes out at ZFW + 10 t, which is 95–105 t for any plausible ZFW and comfortably inside the structural limit.
- Runs **per candidate day**, because trip fuel depends on that day's weather. A warm day burns more in the climb, needing more uplift, making it heavier, lengthening the climb again — so the fixed point amplifies the temperature penalty rather than merely reflecting it.
- Must be **vectorised across candidates**, not iterated per candidate: run the whole march for all ~31,000 candidates at the current TOW vector, update the vector, repeat. Roughly eight times the cost of the existing march.

### Known boundary

The climb tables cover TOW 160–185 t. Worked with the real tables, ZFW 90 t converges to TOW ≈ 163 t and ZFW 95 t to ≈ 171 t — inside the range. But ZFW 80–85 t lands at 152–158 t, **below** it. If the fixed point lands under 160 t, the lower-weight climb pages from the manual are needed.

---

## Phases

### Phase A — fuel, and the in-flight display

Unblocked, and the largest time lever left.

1. Add the subsonic ERA5 levels to `era5.py` and **start that download immediately** — CDS queue latency is the long pole for Phase B.
2. Replace `--tow` with `--zfw` plus the fixed point; `--tow` survives as an override that skips it.
3. `concopt report` prints the fuel plan: uplift, trip fuel, landing fuel, TOW, landing weight.
4. Rework the in-flight display to redraw in place rather than scroll.

### Phase B — descent, subsonic cruise, landing

Waits on the Phase A download.

5. Wire `conc_descent.csv` into `conc_data.py`. The eleven printed wind columns are **not** in the file because they reconstruct exactly: `distance(wind) = dist_zero_wind_nm + wind_kt × time_min / 60`, verified against all 582 printed ±100 kt endpoints to within 1 nm.
6. Model the arrival: decelerate to M1.0 at the schedule's end level, cruise subsonic at M0.95 at an optimised level, descend, then a fixed approach allowance from 1,500 ft. Replaces the flat `DECEL_DESCENT_S`.
7. Default to the **380 kt** descent schedule — fastest overall by 1.5–1.8 minutes and lower fuel, though by less than the raw descent times suggest, because a faster descent covers less ground and leaves more to fly level.

### Phase C — fly it

8. Fly the shortlist winner with `--record`. First real exercise of `inflight.py`, and it measures the constants that are still assumptions.
9. Post-flight notebook reading the recorder's CSV: actual against predicted, where the advice differed from what was flown and what that cost.

---

## Decisions already settled

| | |
|---|---|
| Cruise Mach | 2.00 (manual's schedule; ceilings are computed at it) |
| Descent schedule | 380 kt |
| Crosswind | 25 kt documented limit, 30 kt cutoff with a flag between |
| Tailwind | 10 kt, no buffer |
| Runways | JFK 22R (no penalty) / 31L (+2 min); LHR 09 (no penalty) / 27 (+5 min) |
| Reserve | 10 t at touchdown |
| Timing | brake release to touchdown |
| Departure window | 08:00–14:00 America/New_York |
| Archive | 2014-08-01 onward, the Active Sky historical range |
| Weather for selection | ERA5 to shortlist, Active Sky to rank |

## Open items

1. **ZFW** — decides the fuel plan and whether the lower-weight climb tables are needed.
2. **Supersonic points** are `--accel`/`--decel` arguments, not carried in the `.pln`. Fine for a minimum product; a `route.toml` beside the plan would let a new plan bring its own.
3. **Measured timings** — the climb now comes from tables, but the approach allowance below 1,500 ft is still assumed. The flight recorder measures it.
