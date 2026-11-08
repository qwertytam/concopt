"""`concopt bridge`: runs on the Windows PC that has Prepar3D, and serves the
live sim state over HTTP so the advisor can run elsewhere (`concopt inflight
--state-url http://<this-pc>:19286/state`, see concopt.remote for the JSON
contract). Active Sky is not proxied -- point inflight's --host/--port at
this PC.

Every read goes through inflight's own _connect / _connect_fsuipc /
_live_state_source -- the same calls run_inflight makes -- so unit
conversions (lb -> kg, rad -> deg, inHg -> hPa) live in exactly one place,
_read_state. FSUIPC absent -> _connect_fsuipc returns None with a printed
note, the N2 fields come back null, and the bridge keeps serving.

One GET per tick returns the whole dict. Each request does a fresh read
(under a lock: SimConnect isn't safe to read from two threads at once).
A read that raises answers HTTP 500 with {"error": ...}; the client turns
that into a skipped tick.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from concopt.inflight import _connect, _connect_fsuipc, _live_state_source
from concopt.params import BRIDGE_BIND_HOST, BRIDGE_PORT, BRIDGE_STATE_PATH


def _json_default(obj):
    """numpy scalars (np.float32, np.bool_, ...) -> plain python; anything
    else is a bug worth a TypeError rather than a guess."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != BRIDGE_STATE_PATH:
            self._send(404, {"error": f"only {BRIDGE_STATE_PATH} is served"})
            return
        try:
            with self.server.read_lock:
                state = self.server.read_state()
            json.dumps(state, default=_json_default)  # fail here, not mid-response
        except Exception as exc:  # the bridge must outlive a bad read
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(200, state)

    def log_message(self, *args):  # one line per poll would bury the console
        pass


def make_server(read_state, bind=BRIDGE_BIND_HOST, port=BRIDGE_PORT):
    """A ThreadingHTTPServer answering GET BRIDGE_STATE_PATH with
    read_state() as JSON. port=0 picks a free one (tests)."""
    server = ThreadingHTTPServer((bind, port), _Handler)
    server.read_state = read_state
    server.read_lock = threading.Lock()
    return server


def run_bridge(port=BRIDGE_PORT, bind=BRIDGE_BIND_HOST, simconnect_dll=None):
    sm, aq = _connect(simconnect_dll)
    fsuipc = _connect_fsuipc()
    server = make_server(_live_state_source(aq, fsuipc), bind, port)
    print(f"Bridge serving http://{bind}:{port}{BRIDGE_STATE_PATH} -- Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
