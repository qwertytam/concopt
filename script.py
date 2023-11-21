import logging
from pint import UnitRegistry
import pint_pandas
from flightcondition import FlightCondition, unit
from src.conditions import calc_gs_for_atmosphere, get_sound_speed
from src.conditions import calc_mach_from_cas, calc_tas_from_mach, stagnation_temp
from src.conditions import mach_from_temps
from src.asky import get_atmosphere_as_pd
def main():
    ureg = UnitRegistry()
    Q_ = ureg.Quantity

    # lat = 29.6
    # lon = -98.33
    lat = '10-57-21.24S'
    lon = '115-51-38.16E'
    tude_unts = 'dms'

    alts = list(range(50*10**3, 60*10**3 + 1, 1*10**3))
    atmos = get_atmosphere_as_pd(lat, lon, alts, tude_units=tude_unts)

    atmos.insert(loc=len(atmos.columns), column='SpeedSound', value=None)
    for idx, row in atmos.iterrows():
        ss = get_sound_speed(
            altitude=row.Altitude,
            static_temp=row.Temperature)
        ss_units = ss.units
        atmos.iloc[idx, len(atmos.columns)-1] = ss.magnitude

    atmos.SpeedSound = pint_pandas.PintArray(atmos.SpeedSound, dtype=ss_units)
    
    mach = 2.04
    heading = 100
    atmos = calc_gs_for_atmosphere(atmos, mach, heading)    
    print(atmos)
    print(atmos.dtypes)


    alt = 50189*unit('ft')
    fc = FlightCondition(h=alt, units='SI')
    fc = FlightCondition(h=alt.magnitude/fc.H.to('ft').magnitude * alt, units='SI') 
    isa = +10*unit('delta_degC')
    fc.T = fc.T + isa
    Vc = 530*unit('knots')

    mach = calc_mach_from_cas(Vc, altitude=alt, isa_temp_diff=isa)
    tas = calc_tas_from_mach(mach, altitude=alt, isa_temp_diff=isa)
    total_temp = stagnation_temp(fc.T, mach)
    print(f"Result out:\n" +
          f"FL{(fc.h.to('ft').magnitude/100):.0f}  {fc.h.to('ft'):~.0f}\n" +
          f"CAS {Vc.to('knots'):~.0f}\n" +
          f"OAT {fc.T.to('degK'):~.1f}   {fc.T.to('degC'):~.1f}  " +
          f"ISA {(fc.T - isa).to('delta_degC'):~.1f}\n" +
          f"Local M1 {fc.a.to('knots'):~.0f}\n" +
          f"Mach {mach:~.3f}\n" +
          f"TAS {tas.to('knots'):~.0f}\n" +
          f"Static temp {fc.T.to('degC'):~.1f} " +
          f"Total temp {total_temp.to('degC'):~.1f}")


    max_temp = Q_(127, ureg.degC)
    static_temp = [-65, -56.5, -45]
    
    for st in static_temp:
        max_mach = mach_from_temps(max_temp, Q_(st, ureg.degC))
        print(f"Temp limtied mach: {max_mach:~.3f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    main()
