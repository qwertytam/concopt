"""Every tunable number and physical/aircraft constant in one place.

A LEAF module: it imports nothing from concopt (only stdlib and numpy), so
any module -- including atmos.py's hot path and replay.py, which used to
duplicate a constant to dodge an import cycle -- can import from here freely.
tests/test_params.py enforces that.

What lives here: units and physics, the ISA model, aircraft limits, the fuel
and arrival models' tunables, the search's candidate/climb/grid settings, the
runway screen, the ERA5 download spec, Active Sky / verify settings, the
in-flight advisor's settings, and the synthetic-flight (replay) profile.

What deliberately does NOT: values read out of the performance CSVs
(conc_data's grids and CLIMB_TOW_MIN_T/MAX_T -- derived from the data, not
chosen), table labels (conc_data.CLIMB_BANDS), and output schemas
(inflight.RECORD_COLUMNS/RECORD_SCHEMA_VERSION, arrival's _SEGMENT_KEYS/
_FLAG_KEYS) -- those describe a file or a table, not a setting. CLI path
defaults (results.csv, report.csv) stay in cli.py as UX strings.

Modules import the names they need from here and keep using them under the
same names, so `search.DEFAULT_TOW_T`, `limits.CRUISE_MACH`, etc. still work.
"""
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

# =============================================================================
# Units and physical constants
# =============================================================================
FT_TO_M = 0.3048
M_PER_FL = 30.48          # metres per flight level (FL = 100 ft) == 100 * FT_TO_M
NM_TO_M = 1852.0
KT_TO_MS = NM_TO_M / 3600.0   # 1 kt in m/s
C_TO_K = 273.15               # Celsius -> Kelvin offset
LB_TO_KG = 0.45359237
INHG_TO_HPA = 33.863886666667
S_PER_HOUR = 3600.0
S_PER_DAY = 86400.0       # ZULU_TIME wraps at this
NS_PER_S = 1e9            # numpy datetime64[ns] -> seconds

GAMMA = 1.4               # ratio of specific heats, air
R = 287.05287             # J/(kg K), gas constant for air
G0 = 9.80665              # m/s2
P0 = 101325.0             # Pa, ISA sea level
T0 = 288.15               # K, ISA sea level
A0 = np.sqrt(GAMMA * R * T0)  # m/s, sea-level speed of sound

EARTH_RADIUS_NM = 3440.065    # great-circle distance (route.py)

# --- ISA layer boundaries (atmos.py) -----------------------------------------
ISA_L1_K_PER_M = -0.0065      # troposphere lapse rate, 0-11 km
ISA_H1_M = 11000.0            # troposphere/tropopause boundary
ISA_H2_M = 20000.0            # tropopause/stratosphere boundary
ISA_T_ISO_K = 216.65          # isothermal layer temperature (11-20 km)
ISA_L3_K_PER_M = 0.001        # lapse rate 20-32 km

# =============================================================================
# Aircraft: Concorde limits and cruise
# =============================================================================
MMO = 2.04                    # max operating Mach, all altitudes (structural)
TOTAL_TEMP_MAX_C = 127.0      # max stagnation temperature, all altitudes
TOTAL_TEMP_MAX_K = TOTAL_TEMP_MAX_C + C_TO_K

# The manual's cruise is flown at M2.00, not Mmo 2.04 -- its ceiling table
# (conc_data.ceiling_ft_table) is "the altitude attainable at M2.00", so
# pairing those ceilings with Mmo would let max_mach claim speeds the
# ceiling was never validated at. MMO is the structural limit; CRUISE_MACH is
# what limits.max_mach actually uses, overridable from the CLI
# (--cruise-mach) to try both.
CRUISE_MACH = 2.00

# limits.py's default weight when a caller doesn't pass one (mach_components/
# max_mach/best_level/max_tas). Mid-table, only matters for callers that
# don't care about weight.
LIMITS_DEFAULT_WEIGHT_T = 135.0

# Structural max take-off weight. Coincides with the top of conc_climb.csv's
# TOW axis (conc_data.CLIMB_TOW_MAX_T), but it is a different kind of limit:
# above this the candidate is infeasible, not merely off the end of a table.
MTOW_T = 185.0

# =============================================================================
# Fuel model (fuel.py)
# =============================================================================
# Fuel remaining at touchdown, tonnes. --min-landing-fuel on the CLI.
MIN_LANDING_FUEL_T = 10.0

DEFAULT_DAMPING = 0.5
DEFAULT_TOLERANCE_T = 0.05
DEFAULT_MAX_ITERATIONS = 30

# Where the fixed-point loop starts before it has seen any trip fuel. Only the
# iteration count depends on this -- a contractive map lands in the same
# place from any start inside the table -- so it is deliberately a plain
# constant rather than a tuned guess.
INITIAL_TOW_T = 150.0

# =============================================================================
# Arrival model (arrival.py): BARIX -> touchdown
# =============================================================================
# --- Approach allowance, 1,500 ft to touchdown -------------------------------
# ASSUMPTIONS, not table values. The Phase C flight recorder measures these.
APPROACH_NM = 5.0
APPROACH_MIN = 1.5
APPROACH_FUEL_T = 0.3

# Altitude the descent segment ends at (and the approach allowance starts at).
# DESCENT_END_FL is the same height in FL units, for the descent segment's
# mid-level wind; APPROACH_AGL_FT is the in-flight recorder's flight-phase
# cut for "approach".
DESCENT_END_FT = 1500.0
DESCENT_END_FL = DESCENT_END_FT / 100.0
APPROACH_AGL_FT = DESCENT_END_FT

# --- Level segment at M0.95 --------------------------------------------------
LEVEL_MACH = 0.95

# Typical mass at the decel waypoint -- only used as mass_at_barix_t's
# default, for the many tests below that exercise wind/band/flag behaviour
# and don't care what the level segment's fuel table lookup lands on. Real
# callers (search.py, via fuel._arrival_from_march) always pass the march's
# own weight_at_barix; MASS IS THE TRAP here (see conc_data.subsonic_cruise
# and arrival._arrival_for_speed) so nothing downstream of a real call should
# ever rely on this default firing.
#
# 118, not 110: the subsonic table's lowest levels (FL290-330, which cover
# the 380 kt schedule's FL312 decel_end_fl) are only published down to
# 110 t, and the ~1-1.5 t decel burn taken off mass_at_barix_t before that
# lookup would otherwise land BELOW the table's own floor -- a real, table-
# shaped envelope edge, not a bug, but not what an arbitrary test default
# should be tripping over.
DEFAULT_MASS_AT_BARIX_T = 118.0

# conc_descent.csv has TWO temperature bands, not the three in conc_climb.csv.
BAND_WARM = "above_isa_minus_10"
BAND_COLD = "isa_minus_10_and_below"

SCHEDULES_KT = (325, 350, 380)
FASTEST_SCHEDULE_KT = 380     # arrival()'s default forced schedule, and the
                              # in-flight panel's (time-only) live estimate

# Decel table bounds (conc_descent.csv decel_to_mach1 rows).
CRUISE_FL_MIN = 470.0
CRUISE_FL_MAX = 600.0

# =============================================================================
# Route (route.py, cli.py)
# =============================================================================
DECEL_WAYPOINT_ID = "BARIX"   # supersonic -> subsonic; ends the climb+cruise span
ACCEL_WAYPOINT_ID = "LINND"   # supersonic-span marker / in-flight span tracking
MAX_LEG_NM = 100.0            # `concopt route --max-leg-nm`: subdivide longer legs

# =============================================================================
# Search model (search.py)
# =============================================================================
NY_TZ = ZoneInfo("America/New_York")
DEPARTURE_LOCAL_HOURS = range(8, 15)  # 08:00..14:00 local, inclusive
WINTER_MONTHS = (11, 12, 1, 2)        # the search's own "is winter" sanity check

# Take-off weight default for search.march_legs when a caller passes none.
# From there, weight is a state variable: the climb burns it down to a
# top-of-climb mass (conc_data.climb_to), then it's integrated leg by leg
# from conc_data.fuel_total_kgh_table (Air France performance table) across
# the cruise, not a linear schedule against cum_nm. It drives ceiling_ft,
# which is what actually keeps the optimiser off levels the aircraft can't
# hold; it does not change max_tas above FL430 (530 kt CAS at every weight
# there). The CLI never uses this -- it always solves TOW from --zfw
# (--tow overrides it).
DEFAULT_TOW_T = MTOW_T

# The climb table's top-of-climb level (max level_fl in conc_climb.csv) --
# climb_to(TOP_OF_CLIMB_FL, tow_t, temp_band) gives the brake-release-to-
# top-of-climb time/distance/mass the march starts the cruise from.
TOP_OF_CLIMB_FL = 502.0

# cum_nm within which search._climb_conditions samples ISA deviation/wind to
# pick conc_climb.csv's temp_band and correct the climb's ground distance for
# wind -- conc_climb.csv's shortest climb span is ~210 nm, so this stays
# inside every band's actual climb, cold or warm.
CLIMB_BAND_SAMPLE_NM = 300.0

# best_level picks from this 16-level grid (1000 ft / FL10 steps) rather than
# the 4 raw ERA5 pressure levels -- interpolated per leg, not stored in the
# npz (16 levels there would be ~350 MB vs ~32 MB for 4). FL450-FL600 sits
# inside the ERA5 mandatory-level span (150-70 hPa = FL446-FL605), so every
# target is bracketed -- see the assert in search.march_legs.
TARGET_FL = np.arange(450.0, 601.0, 10.0)

DEFAULT_TOP_N = 50            # `concopt search --top`: rows written to --out
DEFAULT_SHORTLIST_TOP_N = 10  # `concopt shortlist --top`

# =============================================================================
# Runway screen (runways.py)
# =============================================================================
# Runway geometry + JFK/LHR-specific missed-approach-slot/taxi penalty, true
# bearings (see runways.py's docstring). Only 22R/31L are JFK candidates --
# not 04L/13R, which this route never uses. EGLL's runways are parallel (09L/
# 09R and 27R/27L share one true bearing each), so there is no crosswind
# relief from choosing between the pair -- only the westerly/easterly
# choice matters here.
RUNWAYS = {
    "KJFK": (
        {"name": "22R", "true_deg": 211.0, "penalty_s": 0.0},
        {"name": "31L", "true_deg": 301.0, "penalty_s": 2.0 * 60.0},
    ),
    "EGLL": (
        {"name": "09L/09R", "true_deg": 89.0, "penalty_s": 0.0},
        {"name": "27R/27L", "true_deg": 269.0, "penalty_s": 5.0 * 60.0},
    ),
}

# Screen thresholds, kt. 25-30 kt crosswind (gust) is allowed but flagged;
# above 30 kt gust, or any mean tailwind above 10 kt, is unflyable on that
# runway. No buffer on the tailwind test -- it's a hard 10 kt.
XWIND_OK_KT = 25.0
XWIND_FLAG_KT = 30.0
TAILWIND_MAX_KT = 10.0

CALM_WIND_MS = 1e-6           # mean wind below this counts as calm (avoids /0)

# =============================================================================
# ERA5 download spec (era5.py)
# =============================================================================
# Active Sky's historical archive starts here; no point downloading ERA5
# for a date the day-search can never actually query.
ARCHIVE_START = dt.date(2014, 8, 1)

# Route bbox + margin (N, W, S, E), JFK-LHR great circle.
UPPER_AIR_AREA = [53, -76, 38, 2]
UPPER_AIR_LEVELS = ["70", "100", "125", "150"]  # FL447-FL605
UPPER_AIR_GRID = [1.0, 1.0]

# These same netCDFs are reduced TWICE: once onto the cruise legs (the
# search's own --npz), and again onto the arrival legs (--arrival-upper-npz,
# B6) to extend arrival.py's decel-segment wind coverage above SUBSONIC_LEVELS'
# FL414 ceiling -- UPPER_AIR_AREA already covers the arrival region and 150
# joins 175 hPa (SUBSONIC_LEVELS' top) with no gap, so this needs no new CDS
# request, just a second era5.reduce_to_legs(nc_paths, arrival_legs, out_npz)
# call against the already-downloaded era5_upper_*.nc files.
# Departures are 08:00-14:00 America/New_York -> 12:00-18:00Z under EDT,
# 13:00-19:00Z under EST, plus ~4h flight time to cover the arrival end.
# Do not widen this.
UPPER_AIR_TIMES = [f"{h:02d}:00" for h in range(12, 24)]

# Subsonic cruise segment (last ~300 nm into LHR, arrival-only), FL183-FL414.
# Seven levels (175-500 hPa) join up with the 150 hPa data already downloaded.
# B6: this alone does not reach the decel segment's own wind-sampling
# midpoint from a realistic cruise level (roughly FL446-491) -- see
# UPPER_AIR_LEVELS above, which stitches on top to FL605.
SUBSONIC_LEVELS = ["175", "200", "225", "250", "300", "400", "500"]
SUBSONIC_AREA = [54, -13, 47, 2]  # N, W, S, E -- arrival box only
SUBSONIC_GRID = [1.0, 1.0]
SUBSONIC_TIMES = UPPER_AIR_TIMES
# CDS caps this dataset at 4 distinct pressure levels per request, regardless
# of area size or month count -- confirmed by live bisection, 2026-09: every
# single level and every pair of SUBSONIC_LEVELS succeeded; 3- and 4-level
# combos succeeded (including at the full SUBSONIC_AREA above, not just a
# shrunk test box); 5, 6, and the full 7 were all rejected with "cost limits
# exceeded" -- even a 5-level/6-grid-cell request (a smaller total volume)
# was rejected while an unrelated 4-level/105-grid-cell request (a *larger*
# volume) had already succeeded, ruling out total data volume as the driver.
# Separately, a 12-month chunk of just 4 levels was *also* rejected, so the
# existing 1-month chunking (era5.upper_air_months) still has to apply on top
# of the level cap -- the two are independent limits. SUBSONIC_LEVELS is
# therefore split into two <=4-level groups, one CDS request per (month,
# group) -- see era5.download_subsonic.
#
# era5.reduce_to_legs needs no changes to consume these: confirmed by
# live trial, 2026-09, against synthetic files shaped like a 2-month x
# 2-group download (4 files, disjoint level sets, disjoint time ranges) --
# xr.open_mfdataset(paths, combine="by_coords") merges cleanly along BOTH
# the time and pressure_level axes into one (n_time, 7, lat, lon) dataset,
# no NaNs, no error. Pass every era5_subsonic_*.nc path (both groups, every
# month) to reduce_to_legs at once, same as the upper-air files.
SUBSONIC_LEVEL_GROUPS = [SUBSONIC_LEVELS[:4], SUBSONIC_LEVELS[4:]]

# Phase 4 (runways.py's crosswind/tailwind screen, and later the in-flight
# advisor) consumes these; downloaded now so both sit in the same CDS
# queue as the upper-air requests.
SURFACE_AREAS = {"KJFK": [41, -74, 40, -73], "EGLL": [52, -1, 51, 0]}
SURFACE_TIMES = [f"{h:02d}:00" for h in range(24)]
SURFACE_CHUNK_MONTHS = 6      # era5._month_chunks: even the tiny surface area
                              # trips the CDS cost check on a full year

ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]  # CDS "day" list, every month

# =============================================================================
# Active Sky and verify (asky.py, verify.py)
# =============================================================================
ACTIVE_SKY_HOST = "localhost"
ACTIVE_SKY_PORT = 19285
ACTIVE_SKY_URL_BASE = "http://"

VERIFY_DEFAULT_N_POINTS = 12
# `concopt verify` prints CHECK when the recomputed AS total differs from the
# ERA5 total by more than this.
VERIFY_TOTAL_TOLERANCE_MIN = 5.0

# Cache of {"<date> <hour>:00": "<sha256 of that run's AS wind+temp>"},
# under the repo (not gitignored's ephemeral data/era5/ or the .csv
# outputs -- this is small, hand-inspectable state) -- see
# verify._guard_snapshot.
SNAPSHOT_CACHE_PATH = Path("data/verify_snapshot_cache.json")

# =============================================================================
# In-flight advisor and recorder (inflight.py)
# =============================================================================
DEFAULT_INTERVAL_S = 60.0
DEFAULT_LOOKAHEAD_NM = 100.0
DEFAULT_GAIN_THRESHOLD_KT = 3.0

# On-ground ground speed the take-off roll is judged to have started at --
# "SIM ON GROUND true -> ground speed rising through ~40 kt", per spec.
BRAKE_RELEASE_GS_KT = 40.0

# The default interval is tuned for cruise, where a 60 s sample resolves a
# ~3.5 h flight fine. It cannot resolve the arrival's short segments:
# APPROACH_MIN is 1.5 min, so 60 s sampling gives it ONE or TWO rows
# -- not enough to say whether the 1.5 min / 5 nm / 0.3 t approach
# allowance is right, which is one of the three things this recording
# exists to answer. Below LOW_ALT_FT the tick tightens to
# LOW_ALT_INTERVAL_S (~18 samples in that 1.5 min instead of 1). The
# trigger is pressure altitude, not AGL: alt_ft is always available,
# whereas agl_ft is an optional SimConnect read and must never gate control
# flow. This also tightens the take-off roll and initial climb, which is
# free and useful -- the climb model's first minutes are a modelled
# quantity too.
LOW_ALT_FT = 10000.0
LOW_ALT_INTERVAL_S = 5.0

# Minimum wall-clock spacing between live arrival.arrival() refreshes. The
# live arrival estimate costs a 10-level Active Sky query
# (inflight._build_live_arrival_wind_fn); at LOW_ALT_INTERVAL_S that would run
# 12x more often than at cruise, for a ~30 min quantity that does not move
# that fast. At the default interval this changes nothing (60 s tick, 60 s
# refresh); it only bites at the tightened low-altitude tick, where the
# previous estimate is reused and the recorded arrival_time_min repeats.
ARRIVAL_REFRESH_S = 60.0

# Sample flight levels for the live arrival wind profile, spanning the same
# FL183-FL605 envelope search._build_arrival_wind_fn stitches together from
# two ERA5 npz's pre-flight (SUBSONIC_LEVELS + UPPER_AIR_LEVELS).
# Active Sky takes altitude directly (no pressure levels to reuse), so this
# is a plain FL ladder rather than real pressure levels -- the vertical
# interpolation in inflight._build_live_arrival_wind_fn still works in
# log(pressure), against Active Sky's OWN returned pressure at each sampled
# altitude.
ARRIVAL_WIND_SAMPLE_FL = np.array(
    [183.0, 230.0, 280.0, 330.0, 380.0, 430.0, 480.0, 530.0, 580.0, 605.0])

SIMCONNECT_REQUEST_TIME_MS = 200  # AircraftRequests(_time=...): read cache age

# =============================================================================
# Synthetic flight for replay/notebook (replay.build_synthetic_flight)
# =============================================================================
# Shape of a plausible Concorde flight, hung on the model's own control
# points. These are STAND-INS -- they exist so the recorder can be exercised
# end to end without a sim, not to claim real performance.
REPLAY_SAMPLE_INTERVAL_S = 15.0
REPLAY_SEED = 0
REPLAY_AS_WIND_BIAS_KT = 3.0  # Active Sky reads the sim wind offset by this

# Ground roll control points, brake release -> liftoff:
# (elapsed_s, cum_nm, alt_ft, mach, tas_kt, gs_kt). The last row crosses
# BRAKE_RELEASE_GS_KT between it and the row before.
REPLAY_GROUND_ROLL = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (20.0, 0.05, 0.0, 0.02, 15.0, 15.0),
    (40.0, 0.2, 0.0, 0.09, 60.0, 60.0),
)
REPLAY_LIFTOFF = (55.0, 0.4, 200.0, 0.18, 110.0, 110.0)  # same columns

REPLAY_TOP_OF_CLIMB_MACH = 0.95
REPLAY_TOP_OF_CLIMB_TAS_KT = 550.0
REPLAY_DECEL_END_MACH = 1.0
REPLAY_LEVEL_SPEED_FRACTION = 0.95   # level segment TAS = schedule_kt * this
REPLAY_DESCENT_END_MACH = 0.3
REPLAY_DESCENT_END_TAS_KT = 220.0
REPLAY_TOUCHDOWN_MACH = 0.15
REPLAY_TOUCHDOWN_TAS_KT = 140.0

# "A little noise so it is not a tautology" -- small enough that mach/gs stay
# physically sane and the position jitter stays well inside build_legs'
# nearest-leg search tolerance, but large enough that replaying the profile
# is not just reading the control points back verbatim.
REPLAY_MACH_NOISE_FRAC = 0.003
REPLAY_GS_NOISE_KT = 2.0
REPLAY_POSITION_NOISE_NM = 0.1
