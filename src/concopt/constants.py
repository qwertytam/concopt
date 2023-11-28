from importlib.resources import files
from numpy import genfromtxt

from concopt.units import unit, dimless

fp_atmosphere = files("concopt").joinpath("data/atmosphere.csv")
std_atmosphere = genfromtxt(fp_atmosphere,
                            delimiter=',',
                            names=True,
                            unpack=True)

class PhysicalConstants():
    """ General physical constants. Air properties assume a calorically perfect
    gas in non-extreme conditions (approximately: T < 400 K and p < 20 psi).

    Attributes:
        g (acceleration): Acceleration due to gravity
        R_star (energy/temperature/mass): Gas constant for air
        gamma (dimless): Ratio of specific heats for air
        R_earth (length): Radius of Earth
    """
    g = 9.80665 * unit('m/s^2')
    R_star = 287.05287 * unit('J(k kg)')
    gamma = 1.4 * dimless
    R_earth = 6.356766e3 * unit('km')


class AtmosphereConstants():
    """Atmospheric constants

    Attributes:
        p_0 (pressure): Sea level ambient pressure
        T_ice (temperature): Sea level ice point temperature
        T_0 (temperature): Sea level ambient temperature
        rho_0 (density): Sea level ambient density
    """

    # Sea level
    p_0 = 101.325e3 * unit('Pa')
    T_ice = 273.15 * unit('K')
    T_0 = 288.15 * unit('K')
    rho_0 = 1.225 * unit('kg/m^3')

    # Layer base geopotential heights
    H_base = std_atmosphere[0] * unit('km')

    # Layer base temperatures
    T_base = std_atmosphere[1] * unit('K')

    # Layer temperature lapse rates
    T_lapse_rate = std_atmosphere[2] * 1e-3 * unit('K/m')

    # Layer base pressures
    p_base = std_atmosphere[3] * unit('Pa')
    
