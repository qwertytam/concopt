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
SET RESULTS_DIR=data\results\%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%
SET RESULTS_FILE=%RESULTS_DIR%\results.csv
SET RESULTS_ALL_FILE=%RESULTS_DIR%\results_all.csv
SET REPORT_SELECTED_DAYTIME_FILE=%RESULTS_DIR%\report_selected_daytime.csv
SET INFLIGHT_RECORDING_FILE=%RESULTS_DIR%\recording_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%.csv
```

## 2. Reduce netCDF files to npz

```bat
poetry run concopt reduce --pln %PLN% --decel %DECL%
```

## 3. Shortlist best departures

```bat
poetry run concopt search --pln %PLN% --decel %DECL% --npz %ROUTE_LEGS% 0-surface-npz %SURFACE_LEGS% --subsonic-npz %SUBSONIC_LEGS% --arrival-upper-npz %ARRIVAL_UPPER_LEGS% --zfw %ZFW% --top %TOP_N% --out %RESULTS_FILE% --out-all %RESULTS_ALL_FILE%
```

## 4. Verify against ActiveSky

```bat
poetry run concopt shortlist --pln %PLN% --decel %DECL% --npz %ROUTE_LEGS% --subsonic-npz %SUBSONIC_LEGS% --arrival-upper-npz %ARRIVAL_UPPER_LEGS% --search-csv %RESULTS_FILE% --top %TOP_N%
```

## 5. Generate full per-leg plan report for the selected winning day time

```bat
SET DATE=2016-02-12
SET LOCAL_HOUR=13
```

```bat
poetry run concopt report --pln %PLN% --decel %DECL% --npz %ROUTE_LEGS% --date %DATE% --hour %LOCAL_HOUR% --zfw %ZFW% --surface-npz %SURFACE_LEGS% --subsonic-npz %SUBSONIC_LEGS% --arrival-upper-npz %ARRIVAL_UPPER_LEGS% --out %REPORT_SELECTED_DAYTIME_FILE%
```

## 6. Running in flight

```bat
poetry run concopt inflight --pln %PLN% --accel %ACCL% --decel %DECL% --cruise-mach %CRUISE_MACH% --simconnect-dll %SIMCONNECT_DLL% --record %INFLIGHT_RECORDING_FILE% --compare %REPORT_SELECTED_DAYTIME_FILE%
```
