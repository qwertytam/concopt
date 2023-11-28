import logging
import numpy as np
import warnings
from scipy.optimize import minimize_scalar
import pint_pandas
from flightcondition import FlightCondition, unit, dimless
from .constants import p_0, gamma, a_0


def calc_ias_temp_diff(altitude, static_temp):
    """ Calculate difference between given static ambient temperature and
    ISA standard conditions
    
    Arguments:
        altitude -- altitude
        static_temp -- static ambient temperature
    
    Returns
        temperature difference between ambient and standard
    """
    fc = FlightCondition(h=altitude, units='SI')
    static_temp = static_temp.to('degC').magnitude*unit('degC')

    return static_temp - fc.T


def get_sound_speed(altitude, static_temp=None, isa_temp_diff=None):
    """ Calculate speed of sound given altitude and temperature
    
    Arguments:
        altitude -- altitude
        static_temp -- static ambient temperature; optional, if not provided 
            then assumes ISA conditions
        isa_temp_diff -- increase or decrease to ISA temperature for given
            altitude
    
    Returns
        speed of sound
    """
    fc = FlightCondition(h=altitude, units='SI')

    if isa_temp_diff is None:
        isa_temp_diff = 0*unit('delta_degC')
    
    if static_temp is not None:
        if static_temp.units == 'degree_Celsius':
            static_temp = static_temp.magnitude*unit('degC')
            
        fc.T = static_temp + isa_temp_diff
    else:
        fc.T = fc.T + isa_temp_diff

    return fc.a

def calc_ground_speed(wind_speed, wind_dir, true_air_speed, heading):
    """ Calculated ground speed given wind speed, wind direction, aircraft
    true air speed, and aircraft heading.
    
    Solves the wind triangle problem using the law of cosines.
    
    Arguments needs to be parsed as pint units e.g., `wind_dir.unit('degrees')`
    
    Arguments:
        wind_speed -- wind speed
        wind_dir -- wind direction
        true_air_speed -- aircraft true air speed
        heading -- aircraft heading
    
    Returns
        ground speed
    """
    psi = (wind_dir - heading).pint.to('radians').values.quantity

    gs_units = true_air_speed.pint.units
    gs = wind_speed**2 + true_air_speed**2 + \
        2*wind_speed*true_air_speed*np.cos(psi)
    gs = np.sqrt(gs.values.quantity.astype(np.longfloat))
    gs = pint_pandas.PintArray(gs, dtype=gs_units)

    return gs


def saint_venant_formula(mach):
    """ Saint Venant formula for calcuating pressure ratio for given mach number
    
    The pressure ratio as defined here is the pressure difference divided by the
    static pressure. Pressure difference is the total pressure at the mouth of
    the machmeter less the static pressure.
    
    To be used when the pressure ratio is less than or equal to 0.893 i.e., for
    subsonic speeds.
    
    Ref: Flight Mechancis of High-Performance Flight, Nguyuen X. Vinh, equation
    3.42
    
    Arguments:
        mach -- mach number
    
    Returns
        pressure ratio
    """
    return (1 + (gamma - 1)/2 * mach**2)**(gamma/(gamma - 1)) - 1


def lord_rayleigh_formula(mach):
    """ Lord Rayleigh formula for calcuating pressure ratio for given mach number
    
    The pressure ratio as defined here is the pressure difference divided by the
    static pressure. Pressure difference is the total pressure at the mouth of
    the machmeter less the static pressure.
    
    To be used when the pressure ratio is greater than 0.893 i.e., for
    supersonic speeds.
    
    Ref: Flight Mechancis of High-Performance Flight, Nguyuen X. Vinh, equation
    3.44
    
    Arguments:
        mach -- mach number
    
    Returns
        pressure ratio
    """
    warnings.filterwarnings("error")
    try:
        expn = 1/(gamma - 1)
        scaler = ((gamma + 1)/2)**((gamma + 1) * expn)
        scaler = scaler * (2 * expn)**expn
        mscaler = 2 * gamma * expn
        eq = (scaler * mach**mscaler)/(mscaler * mach**2 - 1)**expn - 1
    except RuntimeWarning as rtw:
        logging.warning(f"rayleight formula produced:: {rtw}")
        eq = None

    warnings.resetwarnings()
    return eq


def calc_pressure_ratio(speed_ratio):
    """ Calcuating pressure ratio for a given speed ratio
    
    The pressure ratio as defined here is the pressure difference divided by the
    static pressure. Pressure difference is the total pressure at the mouth of
    the machmeter less the static pressure.
    
    To be used when the pressure ratio is greater than 0.893 i.e., for
    supersonic speeds.
    
    Ref: Flight Mechancis of High-Performance Flight, Nguyuen X. Vinh, equations
    3.42 and 3.44
    
    Arguments:
        speed_ratio -- mach number or calibrated air speed divided by ISA sea
            level speed of sound
    
    Returns
        pressure ratio
    """
    pratio_cutoff = 0.893
    sv = saint_venant_formula(speed_ratio)
    lr = lord_rayleigh_formula(speed_ratio)
    pr = sv if sv < pratio_cutoff else lr    
    return pr


def pressure_ratio_from_cas(cas):
    """ Calcuating pressure ratio for a given calibrated air speed
    
    The pressure ratio as defined here is the pressure difference divided by the
    static pressure. Pressure difference is the total pressure at the mouth of
    the machmeter less the static pressure.
    
    Arguments:
        cas -- calibrated air speed
    
    Returns
        pressure ratio
    """
    return calc_pressure_ratio((cas / a_0).magnitude)


def pressure_ratio_from_mach(mach):
    """ Calcuating pressure ratio for a given mach number
    
    The pressure ratio as defined here is the pressure difference divided by the
    static pressure. Pressure difference is the total pressure at the mouth of
    the machmeter less the static pressure.
    
    Arguments:
        mach -- mach number
    
    Returns
        pressure ratio
    """
    return calc_pressure_ratio(mach)


def pressure_ratio_error(mach, cas_pr):
    """ Calcuating difference in pressure ratios for a given mach
    number
    
    The pressure ratio as defined here is the pressure difference divided by the
    static pressure. Pressure difference is the total pressure at the mouth of
    the machmeter less the static pressure. This formula is most likely used 
    with a solve function to iteratively find the mach number that equals a
    calibrated air speed that was used to calcualted dp_p_in
    
    Arguments:
        cas_pr -- pressure ratio for a given calibrated air speed
        mach -- mach number
    
    Returns
        difference (error) between parsed cas_pr and calculated pressure ratio
        for given mach
    """
    return abs(pressure_ratio_from_mach(mach) - cas_pr)


def get_ambient_pressure(altitude, static_temp=None, isa_temp_diff=None):
    """ Calculate ambient static pressure given altitude and temperature
    
    Arguments:
        altitude -- altitude
        static_temp -- static ambient temperature; optional, if not provided 
            then assumes ISA conditions
        isa_temp_diff -- increase or decrease to ISA temperature for given
            altitude
    
    Returns
        static pressure
    """
    fc = FlightCondition(h=altitude, units='SI')
    
    if isa_temp_diff is None:
        isa_temp_diff = 0*unit('delta_degC')
    
    if static_temp is not None:
        if type(static_temp) is np.float32:
            static_temp = static_temp*unit('degC')
        fc.T = static_temp + isa_temp_diff
    else:
        fc.T = fc.T + isa_temp_diff

    return fc.p


def calc_mach_from_cas(cas, altitude, static_temp=None, isa_temp_diff=None):
    """ Calculate mach number for a given calibrated air speed, altitude and
     temperature.
     
     Iterative search for mach within bounds [0, 5]
    
    Arguments:
        cas -- calibrated air speed
        altitude -- altitude
        static_temp -- static ambient temperature; optional, if not provided 
            then assumes ISA conditions
        isa_temp_diff -- increase or decrease to ISA temperature for given
            altitude
    
    Returns
        mach number
    """
    static_pressure = get_ambient_pressure(altitude, static_temp, isa_temp_diff)
    pr_from_vc = pressure_ratio_from_cas(cas) * p_0 / static_pressure

    mbounds = (0, 5)
    res = minimize_scalar(pressure_ratio_error,
                          bounds=mbounds,
                          args=(pr_from_vc.magnitude),
                          options={'maxiter': 25})
    
    return res.x*dimless
    
    
def calc_tas_from_mach(mach, altitude, static_temp=None, isa_temp_diff=None):
    """ Calculate true air speed for a given mach, altitude and
     temperature.
    
    Arguments:
        mach -- mach number
        altitude -- altitude
        static_temp -- static ambient temperature; optional, if not provided 
            then assumes ISA conditions
        isa_temp_diff -- increase or decrease to ISA temperature for given
            altitude
    
    Returns
        mach number
    """
    return mach * get_sound_speed(altitude, static_temp, isa_temp_diff)

def temp_ratio(mach):
    """ Calculate ratio between ambient static and stagnation temperature for
    given mach
    
    Arguments:
        mach -- mach number
    
    Returns
        ratio of stagnation temperature divided by static ambient temperature
    """
    return (1 + (gamma - 1)/2 * mach**2)*dimless

def stagnation_temp(static_temp, mach):
    """ Calculate stagnation temperature for given ambient static temperature
    and mach
    
    Arguments:
        static_temp -- ambient static temperature
        mach -- mach number
    
    Returns
        stagnation temperature aka total temperature
    """
    return static_temp * temp_ratio(mach)

def mach_from_temps(total_temp, static_temp):
    """ Calculate mach number for given stagnation aka total temperature and
    ambient static temperature
    
    Arguments:
        total_temp -- total aka stagnation temperature
        static_temp -- ambient static temperature
    
    Returns
        mach number
    """
    tr = total_temp.pint.to('degK') / static_temp.pint.to('degK')
    return np.sqrt(2/(gamma - 1) * (tr.values.quantity.m - 1))

    
def calc_gs_for_atmosphere(atmosphere, tas, heading):
    """ Calculate ground speed given atmospheric conditions, mach and heading
    
    Arguments:
        atmosphere -- pandas dataframe containg the columns `WindSpeed`, 
            `WindDirection` and `SpeedSound`
        tas -- true air speed aka `TAS`
        heading -- aircraft heading to calculate ground speed for in degrees
    
    Returns
        atmosphere with additional columns `Heading`, `TAS` and `GroundSpeed`
    """
    atmosphere['Heading'] = heading
    atmosphere.Heading = pint_pandas.PintArray(atmosphere.Heading,
                                               dtype='degrees')
    gs = calc_ground_speed(atmosphere.WindSpeed,
                                                  atmosphere.WindDirection,
                                                  tas,
                                                  atmosphere.Heading)

    return gs