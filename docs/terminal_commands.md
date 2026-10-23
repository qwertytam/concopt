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

```bat
for /f %i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do SET STAMP=%i
SET RESULTS_DIR=data\results\%STAMP%
```

```bat
if not exist "%RESULTS_DIR%\" mkdir "%RESULTS_DIR%"
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
poetry run concopt search --pln %PLN% --decel %DECL% --npz %ROUTE_LEGS% --surface-npz %SURFACE_LEGS% --subsonic-npz %SUBSONIC_LEGS% --arrival-upper-npz %ARRIVAL_UPPER_LEGS% --zfw %ZFW% --cruise-mach %CRUISE_MACH% --top %TOP_N% --out %RESULTS_FILE% --out-all %RESULTS_ALL_FILE%
```

## 4. Verify against ActiveSky

```bat
poetry run concopt shortlist --pln %PLN% --decel %DECL% --npz %ROUTE_LEGS% --subsonic-npz %SUBSONIC_LEGS% --arrival-upper-npz %ARRIVAL_UPPER_LEGS% --search-csv %RESULTS_FILE% --top %TOP_N%
```

## 5. Generate full per-leg plan report for the selected winning day time

`DEP_DATE`, not `DATE`: setting `DATE` replaces cmd's built-in `%DATE%` for the
rest of the session (`SET DATE=` with nothing after it undoes that).

```bat
SET DEP_DATE=2015-12-30
SET LOCAL_HOUR=08
```

```bat
poetry run concopt report --pln %PLN% --decel %DECL% --npz %ROUTE_LEGS% --date %DEP_DATE% --hour %LOCAL_HOUR% --zfw %ZFW% --cruise-mach %CRUISE_MACH% --surface-npz %SURFACE_LEGS% --subsonic-npz %SUBSONIC_LEGS% --arrival-upper-npz %ARRIVAL_UPPER_LEGS% --out %REPORT_SELECTED_DAYTIME_FILE%
```

## 6. Running in flight

```bat
poetry run concopt inflight --pln %PLN% --accel %ACCL% --decel %DECL% --cruise-mach %CRUISE_MACH% --simconnect-dll %SIMCONNECT_DLL% --record %INFLIGHT_RECORDING_FILE% --compare %REPORT_SELECTED_DAYTIME_FILE%
```
