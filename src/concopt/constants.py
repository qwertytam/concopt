#!/usr/bin/env python
"""Dimensioned constants.

Modification of https://github.com/MattCJones/flightcondition/

Initial Author: Matthew C. Jones
Email: matt.c.jones.aoe@gmail.com

Subsequent Author: Tom Marshall

:copyright: 2021 Matthew C. Jones
:copyright: 2023 Tom Marshall
:license: MIT License, see LICENSE for more details.
"""

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
    g = 9.80665 * unit('meter/second^2')
    R_star = 287.05287 * unit('joule/(kelvin kilogram)')
    gamma = 1.4 * dimless
    R_earth = 6.356766e3 * unit('kilometer')


class AtmosphereConstants():
    """Atmospheric constants

    Attributes:
        p_0 (pressure): Sea level ambient pressure
        T_ice (temperature): Sea level ice point temperature
        T_0 (temperature): Sea level ambient temperature
        rho_0 (density): Sea level ambient density
    """

    # Sea level
    p_0 = 101.325e3 * unit('pascal')
    T_ice = 273.15 * unit('kelvin')
    T_0 = 288.15 * unit('kelvin')
    rho_0 = 1.225 * unit('kilogram/meter^3')

    # Layer base geopotential heights
    H_base = std_atmosphere[0] * unit('kilometer')

    # Layer base temperatures
    T_base = std_atmosphere[1] * unit('kelvin')

    # Layer temperature lapse rates
    T_grad = std_atmosphere[2] * 1e-3 * unit('kelvin/meter')

    # Layer base pressures
    p_base = std_atmosphere[3] * unit('pascal')
    
