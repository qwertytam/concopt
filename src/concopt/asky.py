#!/usr/bin/env python
"""Get ActiveSky weather conditions

Author: Tom Marshall

:copyright: 2023 Tom Marshall
:license: MIT License, see LICENSE for more details.
"""

import requests
import logging
import pandas as pd

from concopt.params import ACTIVE_SKY_HOST, ACTIVE_SKY_PORT, ACTIVE_SKY_URL_BASE


def get_atmosphere(lat, lon, alts,
                   host_addr=ACTIVE_SKY_HOST, port=ACTIVE_SKY_PORT):
    """ Get atmospherical conditions from ActiveSky for given position

    Arguments:
        lat -- latitude for given position, decimal degrees
        lon -- longitude for given position, decimal degrees
        alts -- list of altitudes to get conditions for in feet
        host_addr -- ActiveSky host address, 'localhost' by default
        port -- ActiveSky port, 19285 by default

    Returns
        atmosphere in json format; see ActiveSky API documentation for further
        detail
    """
    end_point = '/ActiveSky/API/GetAtmosphere?'
    latr = f"lat={lat}"
    lonr = f"&lon={lon}"
    alts = [str(int(x)) for x in alts] # confirm elements are string  and integers before concat
    alts_joined = "|".join(alts)
    altsr = f"&altitudes={alts_joined}"

    req = f"{ACTIVE_SKY_URL_BASE}{host_addr}:{port}{end_point}{latr}{lonr}{altsr}"
    logging.info(f"Getting atmosphere for request {req}")

    try:
        r = requests.get(req)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Active Sky not responding on {host_addr}:{port}; is it "
            "running with the historical date loaded?"
        ) from e
    logging.info(
        f"Received response with status code {r.status_code}\n{r.text}")

    # Active Sky answers a bad request with the bare text "Error", not JSON.
    # (r.content is bytes, so it can never equal a str -- compare r.text.)
    if r.text.strip() == "Error":
        raise RuntimeError(
            f"Active Sky at {host_addr}:{port} returned 'Error' for {req}; "
            "check the historical date/time is loaded and the request is valid"
        )

    return r.json()


def get_atmosphere_np(lat, lon, alts_ft,
                      host_addr=ACTIVE_SKY_HOST, port=ACTIVE_SKY_PORT):
    """ Get atmospherical conditions from ActiveSky for given position as
    plain numpy arrays, for verify.py and the in-flight advisor.

    Arguments:
        lat -- latitude for given position, decimal degrees
        lon -- longitude for given position, decimal degrees
        alts_ft -- sequence of altitudes to get conditions for, in feet
        host_addr -- ActiveSky host address, 'localhost' by default
        port -- ActiveSky port, 19285 by default

    Returns
        (alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c), each a
        1-D numpy array in ActiveSky's response order (not necessarily
        sorted -- align to a target grid by alt_ft, not by position).
    """
    atmos = get_atmosphere(lat, lon, alts_ft, host_addr=host_addr, port=port)
    # WeatherData is a list of per-altitude records, every field a string
    # (confirmed live against Active Sky, 2026-09).
    wd = pd.DataFrame(atmos['WeatherData']).astype(float)
    return (
        wd['Altitude'].to_numpy(),
        wd['WindDirection'].to_numpy(),
        wd['WindSpeed'].to_numpy(),
        wd['Pressure'].to_numpy(),
        wd['Temperature'].to_numpy(),
    )
