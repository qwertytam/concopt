"""Acceptance tests for concopt.asky: the plain-feet/pint dual input on
get_atmosphere_as_pd, the pint-free get_atmosphere_np variant, and the
connection-error wrapping -- all against a fake requests.get, no live
Active Sky needed.
"""
import numpy as np
import pytest
import requests

from concopt import asky
from concopt.units import unit

# List of per-altitude records, every field a string -- the real shape,
# confirmed live against Active Sky (2026-09); it used to be assumed to be a
# dict of arrays, which raised against a real server.
_WEATHER_DATA = [
    {"Altitude": "45000", "WindDirection": "270.0", "WindSpeed": "90.0",
     "Pressure": "147.0", "Temperature": "-56.5"},
    {"Altitude": "46000", "WindDirection": "280.0", "WindSpeed": "95.0",
     "Pressure": "143.0", "Temperature": "-57.0"},
]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = "ok"
        self.content = b"ok"

    def json(self):
        return self._payload


def test_get_atmosphere_wraps_connection_error(monkeypatch):
    """A requests.exceptions.ConnectionError becomes a RuntimeError naming
    the host/port and telling the user to check Active Sky is running --
    not a bare ConnectionError."""
    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _raise)

    with pytest.raises(RuntimeError, match="Active Sky not responding on localhost:19285"):
        asky.get_atmosphere(40.0, -70.0, [45000])


def test_get_atmosphere_as_pd_accepts_plain_feet(monkeypatch):
    """get_atmosphere_as_pd no longer requires a pint Quantity -- a plain
    list of feet works too."""
    captured = {}

    def _fake_get(req):
        captured["req"] = req
        return _FakeResponse({"WeatherData": _WEATHER_DATA})

    monkeypatch.setattr(requests, "get", _fake_get)

    df = asky.get_atmosphere_as_pd(40.0, -70.0, [45000, 46000])

    assert "altitudes=45000|46000" in captured["req"]
    assert df.Altitude.iloc[0].magnitude == pytest.approx(45000.0)


def test_get_atmosphere_as_pd_still_accepts_pint_quantity(monkeypatch):
    """The pint Quantity input the docstring always promised still works,
    converted to feet before building the request."""
    captured = {}

    def _fake_get(req):
        captured["req"] = req
        return _FakeResponse({"WeatherData": _WEATHER_DATA})

    monkeypatch.setattr(requests, "get", _fake_get)

    alts = unit.Quantity([45000.0, 46000.0], "ft")
    asky.get_atmosphere_as_pd(40.0, -70.0, alts)

    assert "altitudes=45000|46000" in captured["req"]


def test_get_atmosphere_as_pd_passes_through_host_and_port(monkeypatch):
    """host_addr/port reach the actual request -- previously silently
    dropped when alts.to('ft').magnitude was inlined."""
    captured = {}

    def _fake_get(req):
        captured["req"] = req
        return _FakeResponse({"WeatherData": _WEATHER_DATA})

    monkeypatch.setattr(requests, "get", _fake_get)

    asky.get_atmosphere_as_pd(40.0, -70.0, [45000], host_addr="192.168.1.5", port=12345)

    assert "192.168.1.5:12345" in captured["req"]


def test_get_atmosphere_np_returns_plain_arrays(monkeypatch):
    """No pint anywhere in the return -- five plain numpy arrays in
    Active Sky's response order."""
    monkeypatch.setattr(requests, "get",
                         lambda req: _FakeResponse({"WeatherData": _WEATHER_DATA}))

    alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c = asky.get_atmosphere_np(
        40.0, -70.0, [45000, 46000]
    )

    for arr in (alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c):
        assert isinstance(arr, np.ndarray)
    assert alt_ft.tolist() == [45000.0, 46000.0]
    assert temp_c.tolist() == pytest.approx([-56.5, -57.0])
