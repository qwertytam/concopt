# Terminal Commands

Windows terminal commands for copy/paste into a Windows command line to run the concopt pipeline. Assumes the ERA5 weather has already been downloaded.

Refer to RUNBOOK.md for full list of commands and explanations.

---

## 0. Set env variables

```bat
SET PLN=data\CONC_KJFK_EGLL_SPEED02.pln
SET ACCL=40N073W
SET DECL=51N005W
SET ZFW=84
SET CRUISE_MACH=2.04

SET SIMCONNECT_DLL="D:\P3D_Add-ons\FSLabs\Libraries\SimConnect_P3D_v5.dll"

SET ROUTE_LEGS=data\era5\route_legs.npz
SET SURFACE_LEGS=data\era5\surface_legs.npz
SET SUBSONIC_LEGS=data\era5\subsonic_legs.npz
SET ARRIVAL_UPPER_LEGS=data\era5\arrival_upper_legs.npz

SET TOP_N=50
```

## 1. Set the results folder

`Get-Date` doesn't depend on the Windows date format, unlike `%DATE%`. Pasted at the prompt it's
`%i`; inside a `.bat` file it would be `%%i`.

Paste this block on its own. It lists the existing results folders, newest
first, then asks which one to use. Press Enter for a new timestamped folder,
or type an existing one (e.g. `data\results\20261023_0630`) to carry on with
it. The prompt must be the last line you paste, or the next pasted line
becomes its answer.

```bat
for /f %i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do SET STAMP=%i
SET RESULTS_DIR=data\results\%STAMP%
dir /b /ad /o-n data\results
SET /P RESULTS_DIR=Results folder [Enter = new %RESULTS_DIR%, or type an existing one]: 
```

Then this block uses that folder if it exists, or creates it, and says which:

```bat
if exist "%RESULTS_DIR%\" (echo Using existing folder %RESULTS_DIR%) else (mkdir "%RESULTS_DIR%" && echo Created folder %RESULTS_DIR%)
SET RESULTS_FILE=%RESULTS_DIR%\results.csv
SET RESULTS_ALL_FILE=%RESULTS_DIR%\results_all.csv
SET REPORT_SELECTED_DAYTIME_FILE=%RESULTS_DIR%\report_selected_daytime.csv
SET INFLIGHT_RECORDING_FILE=%RESULTS_DIR%\recording.csv
```

Use just the bare `SET` command to list all existing environment variables.

## 2. Reduce netCDF files to npz

Re-run whenever the `.pln` or `DECL` changes -- the `.npz` files are cut to the
route's legs, and `search` raises if they don't match.

```bat
poetry run python scripts/build_npz.py --pln %PLN% --decel %DECL%
```

## 3. Shortlist best departures

```bat
poetry run concopt search --pln %PLN% --decel %DECL% --thrust-reversers --npz %ROUTE_LEGS% --surface-npz %SURFACE_LEGS% --subsonic-npz %SUBSONIC_LEGS% --arrival-upper-npz %ARRIVAL_UPPER_LEGS% --zfw %ZFW% --cruise-mach %CRUISE_MACH% --top %TOP_N% --out %RESULTS_FILE% --out-all %RESULTS_ALL_FILE%
```

## 4. Verify against ActiveSky

```bat
poetry run concopt shortlist --pln %PLN% --decel %DECL% --npz %ROUTE_LEGS% --subsonic-npz %SUBSONIC_LEGS% --arrival-upper-npz %ARRIVAL_UPPER_LEGS% --search-csv %RESULTS_FILE% --top %TOP_N%
```

## 5. Generate full per-leg plan report for the selected winning day time

`DEP_DATE`, not `DATE`: setting `DATE` replaces cmd's built-in `%DATE%` for the
rest of the session (`SET DATE=` with nothing after it undoes that).

```bat
SET DEP_DATE=2026-01-21
SET LOCAL_HOUR=12
SET REFRESH_INTERVAL=10
SET WEATHER_LOOKAHEAD_NM=0
```

```bat
poetry run concopt report --pln %PLN% --decel %DECL% --thrust-reversers --npz %ROUTE_LEGS% --date %DEP_DATE% --hour %LOCAL_HOUR% --zfw %ZFW% --cruise-mach %CRUISE_MACH% --surface-npz %SURFACE_LEGS% --subsonic-npz %SUBSONIC_LEGS% --arrival-upper-npz %ARRIVAL_UPPER_LEGS% --out %REPORT_SELECTED_DAYTIME_FILE%
```

## 6. Running in flight

Either run everything on the sim PC (6a), or run only the bridge on the sim PC and
the advisor on another machine such as a Mac (6b). Pick one: the bridge and a
standalone `inflight` can't share the one SimConnect connection.

### 6a. All on the sim PC

```bat
poetry run concopt inflight --pln %PLN% --accel %ACCL% --decel %DECL% --cruise-mach %CRUISE_MACH% --thrust-reversers --simconnect-dll %SIMCONNECT_DLL% --record %INFLIGHT_RECORDING_FILE% --compare %REPORT_SELECTED_DAYTIME_FILE% --interval %REFRESH_INTERVAL%  --lookahead-nm 0
```

### 6b. Bridge on the sim PC, advisor on a Mac

On the **sim PC** (this replaces 6a; stop any running `inflight` first). Allow
Windows Firewall access on the private network if it asks (TCP 19286):

```bat
poetry run concopt bridge --simconnect-dll %SIMCONNECT_DLL%
```

It prints `Bridge serving http://0.0.0.0:19286/state`, then stays quiet. From the
Mac, `curl http://<sim-pc-ip>:19286/state` should return the live state as JSON.

On the **Mac** (zsh; the `.pln`, the report CSV for `--compare` and the results folder
must be on the Mac too). `SIM_PC` is the sim PC's IP address:

```zsh
export SIM_PC=192.168.11.123
export PLN=data/CONC_KJFK_EGLL_SPEED02.pln ACCL=40N073W DECL=51N005W CRUISE_MACH=2.04 REFRESH_INTERVAL=10
export INFLIGHT_RECORDING_FILE=data/results/<folder>/recording.csv
export REPORT_SELECTED_DAYTIME_FILE=data/results/<folder>/report_selected_daytime.csv
```

```zsh
poetry run concopt inflight --pln $PLN --accel $ACCL --decel $DECL --cruise-mach $CRUISE_MACH --thrust-reversers --state-url http://$SIM_PC:19286/state --host $SIM_PC --record $INFLIGHT_RECORDING_FILE --compare $REPORT_SELECTED_DAYTIME_FILE --interval $REFRESH_INTERVAL --lookahead-nm 0
```

- `--state-url` replaces `--simconnect-dll` (which goes on `bridge`; the two can't be combined).
- `--host $SIM_PC` points Active Sky at the sim PC (port 19285, already reachable remotely).
- Press `o` to log N1/N2 does not work on the Mac (Windows console only).
- A poll that fails or times out (1.5 s) skips that tick with a "No usable data this tick" notice.
- Start before brake release, or pass `--brake-release-utc HH:MM:SS`, so elapsed time counts from takeoff.

