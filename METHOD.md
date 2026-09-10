# How the day search picks a date and time

## The question being answered

Of every departure Active Sky can reproduce — 08:00 to 14:00 New York local,
every day from 1 August 2014 to today, about 30,900 candidates — which one
gives the shortest time across the supersonic segment, LINND to BARIX
(2,731 nm of a 3,158 nm route)?

## What decides it

Ground speed is true airspeed plus the along-track wind component. Both terms
depend on the flight level, so the search picks a level for every leg.

True airspeed is capped by whichever of three limits binds first:

- CRUISE_MACH, 2.00 by default (the manual's cruise Mach, not Mmo 2.04 --
  `--cruise-mach` tries the latter)
- the 530 kt CAS placard (above FL430; identical at every weight)
- total temperature 127 °C, i.e. M = sqrt(5 * (400.15/T - 1))

Below about FL505 the CAS limit binds and TAS rises about 25 kt per 1,000 ft.
Above that, CRUISE_MACH binds and TAS is flat, so the level choice becomes
purely a wind question. There is also an optimum static air temperature: TAS
peaks where the CRUISE_MACH and total-temperature limits cross, at -50.8 °C
(CRUISE_MACH=2.00; -54.8 °C at Mmo 2.04). Colder air is slower (lower speed of
sound) and warmer air is slower (temperature limit bites) — this is the one
result most likely to be counter-intuitive.

A separate, altitude-side limit -- the ceiling -- caps which levels are even
reachable, from the Air France performance table (`conc_supersonic_cruise.csv`):
the altitude attainable at CRUISE_MACH, as a function of weight *and* ISA
deviation (colder air gives more thrust and a higher ceiling; at 165 t it
ranges from 43,494 ft at ISA+15 to 52,269 ft at ISA-20). One cell (165 t,
ISA-30) is thrust-limited below what the table's own Mach/CAS/temperature
model would predict and is flagged rather than modelled.

Sensitivities, largest first:

1. along-track wind — routinely +/- 40-60 kt between days
2. temperature — about 40 kt across ISA-10 to ISA+15
3. runway direction at each end — 5 min at LHR, 2 min at JFK

## The data

ERA5 reanalysis, u/v/t on the 150, 125, 100 and 70 hPa pressure levels
(FL446/484/531/605), 1° grid, hours 12-23 Z, over a 38-53 N / 76 W-2 E box.
Because a flight level *is* a pressure altitude, pressure maps to flight level
exactly through the ISA relation — no geopotential is involved. Values are
interpolated to 1,000 ft steps between FL450 and FL600 in log-pressure.

ERA5 is not what P3D loads. Active Sky builds its historical weather from the
same underlying archives, so the ranking should carry across, but the top
candidates get checked against Active Sky directly (Phase 5) before being
believed.

## The algorithm

The route is parsed from the .pln, and legs longer than 100 nm are subdivided
so that no wind sample stands in for more than 100 nm — the raw plan has legs
up to 431 nm, which would smear across a jet-stream gradient.

For each candidate departure, march the 32 supersonic legs in order:

  1. leg clock time = departure + time elapsed so far
  2. interpolate ERA5 in time (between hourly slices) and in height (linear
     in log-pressure, onto 1,000 ft steps from FL450-FL600)
  3. compute max TAS at every level from the three speed limits
  4. ground speed = TAS + along-track wind, holding the ground track
     (solving for drift, not assuming heading equals track)
  5. discard levels above the weight-and-temperature-dependent ceiling;
     take the fastest
  6. add leg distance / ground speed to the elapsed time
  7. burn fuel for the leg (Air France table, at the weight/ISA deviation
     just used) and carry the new weight into the next leg

Weight starts at 165 t at LINND and is integrated leg by leg rather than
scheduled — a still-air, ISA+0 run burns it down to about 111 t by BARIX in
about 144 minutes. It does not change the CAS limit above FL430 — the
placard is 530 kt at every weight up there — but it does set the ceiling
(via the performance table), which is what keeps the optimiser off levels
the aircraft cannot hold, and its speed-envelope Mach at the reduced weight
later in the cruise.

## What the model does not include

Thrust and drag as such (fuel burn is read from the performance table, not
integrated from a thrust/drag model), step-climb scheduling, ATC routing,
and any day-to-day variation in the oceanic track — the same .pln is flown
every time.