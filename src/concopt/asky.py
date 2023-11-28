#!/usr/bin/env python
"""Get and set ActiveSky weather conditions

Author: Tom Marshall

:copyright: 2023 Tom Marshall
:license: MIT License, see LICENSE for more details.
"""

import requests
import logging
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
    altsr = f"&altitudes={"|".join(alts)}"

    req = f"{url_base}{host_addr}:{port}{end_point}{latr}{lonr}{altsr}"
    logging.info(f"Getting atmosphere for request {req}")

    r = requests.get(req)
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
    pandas dataframe
    
    Arguments:
        lat -- latitude for given position
        lon -- longitude for given position
        alts -- list of altitudes to get conditions for
        host_addr -- ActiveSky host address, 'localhost' by default
        port -- ActiveSky port, 19285 by default
        tude_units -- units for lat and lon; decimal or 'dd' by default; can
            also be 'dms' for degrees, minutes, seconds in format 'dd-mm-ss.sssN'
    
    Returns
        atmosphere in json format; see ActiveSky API documentation for further
        detail
    """
    atmos = get_atmosphere(lat, lon,
                           alts.to('ft').magnitude,
                           tude_units=tude_units)
    atmos = pd.DataFrame.from_dict(atmos['WeatherData'], dtype='Float32')

    atmos.Altitude = pint_pandas.PintArray(atmos.Altitude, dtype="feet")
    atmos.WindDirection = pint_pandas.PintArray(atmos.WindDirection,dtype="degrees")
    atmos.WindSpeed = pint_pandas.PintArray(atmos.WindSpeed, dtype="knots")
    atmos.Pressure = pint_pandas.PintArray(atmos.Pressure, dtype="hPa")
    atmos.Temperature = pint_pandas.PintArray(atmos.Temperature, ppunit.Unit('degC'))

    return atmos