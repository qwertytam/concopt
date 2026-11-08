"""The advisor side of the bridge: a state_source that polls concopt.bridge
over HTTP, so `concopt inflight --state-url` can run on a Mac while P3D,
Active Sky and FSUIPC stay on the Windows PC. Imports nothing from inflight
(no SimConnect needed here); `requests` is already a dependency.

THE CONTRACT -- GET <url> returns one JSON object, exactly the dict
inflight._read_state(aq) merged with inflight._read_n2(fsuipc) returns
(units already converted there, never here):

  required (None/non-finite -> run_inflight skips the tick):
    lat_deg, lon_deg, alt_ft, mach, tas_kt, gs_kt, weight_t   number | null
  also always present:
    on_ground  bool;   zulu_s  number | null
  optional, recorder/panel only (null = the read did not come back):
    fuel_kg, sim_wind_kt, sim_wind_dir_deg, sim_temp_c, sim_pressure_hpa,
    agl_ft, vs_fpm, cas_kt, track_deg, eng1_n2..eng4_n2   number | null

NONE SURVIVES THE ROUND TRIP. null <-> None, and nothing here fills a gap
in: a substituted 0 for a dropped weight_t is a silently wrong number
(TOTAL_WEIGHT dropped mid-cruise on 2026-10-23); None is a visible skipped
tick. Float NaN is left as the JSON token NaN (python's json reads it back
as nan), which _check_sim_state rejects the same way.

NO RETRY. A poll that fails, times out, answers non-200, or returns
something that isn't the contract raises RuntimeError -- the exception
run_inflight already turns into "No usable data this tick" and a skipped
row. Retrying here would hide a dead sim behind a stalled display.
"""
import requests

from concopt.params import REMOTE_STATE_TIMEOUT_S

STATE_KEYS = (
    "lat_deg", "lon_deg", "alt_ft", "mach", "tas_kt", "gs_kt", "weight_t",
    "on_ground", "zulu_s",
    "fuel_kg", "sim_wind_kt", "sim_wind_dir_deg", "sim_temp_c",
    "sim_pressure_hpa", "agl_ft", "vs_fpm", "cas_kt", "track_deg",
    "eng1_n2", "eng2_n2", "eng3_n2", "eng4_n2",
)


def remote_state_source(url, timeout_s=REMOTE_STATE_TIMEOUT_S):
    """A state_source() that polls the bridge at url: the whole state dict
    per call, one request."""

    def state_source():
        try:
            response = requests.get(url, timeout=timeout_s)
        except requests.RequestException as exc:
            raise RuntimeError(f"bridge at {url} did not answer: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(f"bridge at {url} answered HTTP {response.status_code}: "
                               f"{response.text[:120]!r}")
        try:
            state = response.json()
        except ValueError as exc:
            raise RuntimeError(f"bridge at {url} sent invalid JSON: {exc}") from exc
        if not isinstance(state, dict):
            raise RuntimeError(f"bridge at {url} sent {type(state).__name__}, not an object")
        missing = [k for k in STATE_KEYS if k not in state]
        if missing:
            raise RuntimeError(f"bridge at {url} state is missing {missing} "
                               "(bridge and advisor versions differ?)")
        return state

    return state_source
