import logging
import numpy as np
import pandas as pd
import pint
import pint_pandas
import matplotlib.pyplot as plt
from flightcondition import FlightCondition, unit
from src.conditions import calc_gs_for_atmosphere, get_sound_speed
from src.conditions import calc_mach_from_cas, calc_tas_from_mach, stagnation_temp
from src.conditions import mach_from_temps
from src.asky import get_atmosphere_as_pd
from data.conc_data import get_cas_limit, get_mach_limit, get_total_temp_limit

def main():
    # ureg = pint.UnitRegistry()
    # Q_ = ureg.Quantity

    # --------------------------------
    # Get weather for given position
    lat = '10-57-21.24S'
    lon = '115-51-38.16E'
    tude_unts = 'dms'

    # alts = list(range(50*10**3, 60*10**3 + 1, 1*10**3))
    alts = list(range(40*10**3, 60*10**3 + 1, 1*10**3))
    atmos = get_atmosphere_as_pd(lat, lon, alts, tude_units=tude_unts)

    # --------------------------------
    # Get ambient speed of sound
    atmos.insert(loc=len(atmos.columns), column='SpeedSound', value=None)
    for idx, row in atmos.iterrows():
        ss = get_sound_speed(
            altitude=row.Altitude,
            static_temp=row.Temperature)
        ss_units = ss.units
        atmos.iloc[idx, len(atmos.columns)-1] = ss.magnitude

    atmos.SpeedSound = pint_pandas.PintArray(atmos.SpeedSound, dtype=ss_units)

    # --------------------------------
    # Get airframe cas, mach and temp limits
    atmos['CASLimit'] = pint_pandas.PintArray(
        get_cas_limit(atmos.Altitude), dtype='knots')
    atmos['MachLimit'] = get_mach_limit(atmos.Altitude)
    atmos['TotalTempLimit'] = get_total_temp_limit(atmos.Altitude)
    atmos.TotalTempLimit = pint_pandas.PintArray(
        atmos.TotalTempLimit, dtype='degC')

    # --------------------------------
    # Get max mach for each of cas and temp limits
    atmos.insert(loc=len(atmos.columns), column='MachFromCASLimit', value=None)
    for idx, row in atmos.iterrows():
        mach = calc_mach_from_cas(
            cas=row.CASLimit.to('m/s').magnitude,
            altitude=row.Altitude,
            static_temp=row.Temperature.to('degC').magnitude)
        atmos.iloc[idx, len(atmos.columns)-1] = mach.magnitude

    atmos.MachFromCASLimit = atmos.MachFromCASLimit.astype('float64')

    atmos['MachfromTotalTempLimit'] = mach_from_temps(atmos.TotalTempLimit,
                                                     atmos.Temperature)

    # --------------------------------
    # Get tas limit and gs
    atmos['MaxMach'] = atmos.loc[:, ['MachLimit', 'MachFromCASLimit', 'MachfromTotalTempLimit']].min(axis=1)
    atmos['MaxTAS'] = atmos.SpeedSound * atmos.MaxMach
    atmos['GS'] = calc_gs_for_atmosphere(atmos, atmos.MaxTAS, 100)

    # --------------------------------
    # Find maximum gs and altitude
    max_gs_row = atmos.GS.argmax()
    print(atmos.iloc[max_gs_row, :])
    
    plt.plot(atmos.GS.pint.to('knots').values.quantity.m,
             atmos.Altitude.pint.to('ft').values.quantity.m,
             marker='o', color='k')
    plt.show()    

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    main()
