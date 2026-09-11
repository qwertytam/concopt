#!/usr/bin/env python
"""Get and set ActiveSky weather conditions

Author: Tom Marshall

:copyright: 2023 Tom Marshall
:license: MIT License, see LICENSE for more details.
"""

import requests
import logging
import numpy as np
import pandas as pd
import pint_pandas

from concopt.utils import convert_dms_to_dd
from concopt.units import ppunit

# defaults
url_base = 'http://'

def get_atmosphere(lat, lon, alts,
                   host_addr='localhost', port=19285, tude_units='dd'):
    """ Get atmospherical conditions from ActiveSky for given position

    Arguments:
        lat -- latitude for given position
        lon -- longitude for given position
        alts -- list of altitudes to get conditions for in feet
        host_addr -- ActiveSky host address, 'localhost' by default
        port -- ActiveSky port, 19285 by default
        tude_units -- units for lat and lon; decimal or 'dd' by default; can
            also be 'dms' for degrees, minutes, seconds in format 'dd-mm-ss.sssN'

    Returns
        atmosphere in json format; see ActiveSky API documentation for further
        detail
    """
    if tude_units == 'dms':
        logging.info(f"Converting tudes to dd")

        logging.info(f"lat: {lat}   lon: {lon}")
        lat = convert_dms_to_dd(lat)
        lon = convert_dms_to_dd(lon)

        logging.info(f"lat: {lat}   lon: {lon}")


    end_point = '/ActiveSky/API/GetAtmosphere?'
    latr = f"lat={lat}"
    lonr = f"&lon={lon}"
    alts = [str(int(x)) for x in alts] # confirm elements are string  and integers before concat
    alts_joined = "|".join(alts)
    altsr = f"&altitudes={alts_joined}"

    req = f"{url_base}{host_addr}:{port}{end_point}{latr}{lonr}{altsr}"
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

    if r.content == "Error":
        json = None
    else:
        json = r.json()

    return json


def get_atmosphere_as_pd(lat, lon, alts,
                         host_addr='localhost', port=19285, tude_units='dd'):
    """ Get atmospherical conditions from ActiveSky for given position as a
    pandas dataframe. Display-only -- see get_atmosphere_np for the
    pint-free variant everything else (verify.py, the in-flight advisor)
    should use.

    Arguments:
        lat -- latitude for given position
        lon -- longitude for given position
        alts -- list of altitudes to get conditions for, in feet, as a plain
            sequence OR a pint Quantity (anything with a .to() method)
        host_addr -- ActiveSky host address, 'localhost' by default
        port -- ActiveSky port, 19285 by default
        tude_units -- units for lat and lon; decimal or 'dd' by default; can
            also be 'dms' for degrees, minutes, seconds in format 'dd-mm-ss.sssN'

    Returns
        atmosphere in json format; see ActiveSky API documentation for further
        detail
    """
    alts_ft = alts.to('ft').magnitude if hasattr(alts, 'to') else alts
    atmos = get_atmosphere(lat, lon, alts_ft,
                           host_addr=host_addr, port=port, tude_units=tude_units)
    # WeatherData is a list of per-altitude records, every field a string
    # (confirmed live against Active Sky, 2026-09) -- not the dict-of-arrays
    # this used to assume, which raised on a real server. pd.DataFrame on a
    # list of dicts handles both the reshape and the string->float parse.
    atmos = pd.DataFrame(atmos['WeatherData']).astype('float32')

    atmos.Altitude = pint_pandas.PintArray(atmos.Altitude, dtype="feet")
    atmos.WindDirection = pint_pandas.PintArray(atmos.WindDirection,dtype="degrees")
    atmos.WindSpeed = pint_pandas.PintArray(atmos.WindSpeed, dtype="knots")
    atmos.Pressure = pint_pandas.PintArray(atmos.Pressure, dtype="hPa")
    atmos.Temperature = pint_pandas.PintArray(atmos.Temperature, ppunit.Unit('degC'))

    return atmos


def get_atmosphere_np(lat, lon, alts_ft,
                      host_addr='localhost', port=19285, tude_units='dd'):
    """ Get atmospherical conditions from ActiveSky for given position as
    plain numpy arrays -- no pint, for verify.py and the in-flight advisor
    (atmos.py/limits.py's hot path never touches pint either).

    Arguments:
        lat -- latitude for given position
        lon -- longitude for given position
        alts_ft -- sequence of altitudes to get conditions for, in feet
        host_addr -- ActiveSky host address, 'localhost' by default
        port -- ActiveSky port, 19285 by default
        tude_units -- units for lat and lon; decimal or 'dd' by default; can
            also be 'dms' for degrees, minutes, seconds in format 'dd-mm-ss.sssN'

    Returns
        (alt_ft, wind_dir_deg, wind_speed_kt, pressure_hpa, temp_c), each a
        1-D numpy array in ActiveSky's response order (not necessarily
        sorted -- align to a target grid by alt_ft, not by position).
    """
    atmos = get_atmosphere(lat, lon, alts_ft,
                           host_addr=host_addr, port=port, tude_units=tude_units)
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