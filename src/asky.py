import requests
import logging
from utils import convert_dms_to_dd

# defaults
url_base = 'http://'

def get_atmosphere(lat, lon, alts, host_addr='localhost', port=19285, tude_units='dd'):
    if tude_units == 'dms':
        logging.info(f"Converting tudes to dd")

        logging.info(f"lat: {lat}   lon: {lon}")
        lat = convert_dms_to_dd(lat)
        lon = convert_dms_to_dd(lon)
    
        logging.info(f"lat: {lat}   lon: {lon}")

    
    end_point = '/ActiveSky/API/GetAtmosphere?'
    latr = f"lat={lat}"
    lonr = f"&lon={lon}"
    alts = [str(x) for x in alts] # confirm elements are string before concat
    altsr = f"&altitudes={"|".join(alts)}"

    req = f"{url_base}{host_addr}:{port}{end_point}{latr}{lonr}{altsr}"
    logging.info(f"Getting atmosphere for request {req}")

    r = requests.get(req)
    logging.info(
        f"Received response with status code {r.status_code}\n{r.text}")
    
    return r.json()