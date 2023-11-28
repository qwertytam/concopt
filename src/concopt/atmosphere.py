#!/usr/bin/env python
"""Compute atmospheric properties from standard atmosphere model.

Modification of https://github.com/MattCJones/flightcondition/

Initial Author: Matthew C. Jones
Email: matt.c.jones.aoe@gmail.com

Subsequent Author: Tom Marshall

:copyright: 2021 Matthew C. Jones
:copyright: 2023 Tom Marshall
:license: MIT License, see LICENSE for more details.
"""

import warnings

import numpy as np

from concopt.constants import PhysicalConstants as Phys
from concopt.constants import AtmosphereConstants as Atmo
from concopt.common import AliasAttributes, DimensionalData,\
    _property_decorators
from concopt.units import unit

class Layer(DimensionalData):
    """Class to compute and store layer data. """

    names_dict = {
        'H_base': 'base_geopotential_height',
        'T_base': 'base_geopotential_temperature',
        'T_grad': 'temperature_gradient',
        'p_base': 'base_static_pressure',
    }

    def __init__(self, H_arr, units=None):
        """Initialize Layer nested class.

        Args:
            H_arr (length): Geopotential altitude
            units (str): Set to 'US' for US units or 'SI' for SI
        """
        H_arr = np.atleast_1d(H_arr)
        self.units = units

        self._H_base = np.zeros_like(H_arr) * unit('m')
        self._T_base = np.zeros_like(H_arr) * unit('K')
        self._T_grad = np.zeros_like(H_arr) * unit('K/m')
        self._p_base = np.zeros_like(H_arr) * unit('Pa')

        for idx, H in enumerate(H_arr):
            jdx = __class__._layer_idx(H)
            self._H_base[idx] = Atmo.H_base[jdx]
            self._T_base[idx] = Atmo.T_base[jdx]
            self._T_grad[idx] = Atmo.T_grad[jdx]
            self._p_base[idx] = Atmo.p_base[jdx]

        # Initialize access by full quantity name through .byname.<name>
        self.byname = AliasAttributes(
            varsobj_arr=[self, ], names_dict_arr=[__class__.names_dict, ])

    @staticmethod
    def _layer_idx(H):
        """Find index for layer data.

        Args:
            H_arr (length): Geopotential altitude

        Returns:
            int: Index of layer
        """
        idx = None
        for idx, H_base in enumerate(Atmo.H_base[:-1]):
            H_base_np1 = Atmo.H_base[idx+1]
            if H_base <= H < H_base_np1:
                return idx
        else:  # no break
            raise ValueError("H out of bounds.")

    def tostring(self, units=None, max_sym_chars=None,
                 max_name_chars=None, pretty_print=True):
        """Output string representation of class object.

        Args:
            units (str): Set to 'US' for US units or 'SI' for SI
            max_sym_chars (int): Maximum characters in symbol name
            max_name_chars (int): Maximum characters iin full name
            pretty_print (bool): Pretty print output

        Returns:
            str: String representation
        """
        if units is not None:
            self.units = units

        if self.units == 'US':
            H_base_units   = 'kft'
            T_base_units   = 'degR'
            T_grad_units   = 'degR/kft'
            p_base_units   = 'lbf/ft^2'
        else:  # SI units
            H_base_units   = 'km'
            T_base_units   = 'degK'
            T_grad_units   = 'degK/km'
            p_base_units   = 'Pa'

        # Insert longer variable name into output
        if max_sym_chars is None:
            max_sym_chars = max([len(v) for v in self.names_dict.keys()])
        if max_name_chars is None:
            max_name_chars = max([len(v) for v in self.names_dict.values()])

        H_base_str = self._vartostr(var=self.H_base, var_str='H_base',
                                    to_units=H_base_units,
                                    max_sym_chars=max_sym_chars,
                                    max_name_chars=max_name_chars,
                                    fmt_val="10.5g", pretty_print=pretty_print)
        T_base_str = self._vartostr(var=self.T_base, var_str='T_base',
                                    to_units=T_base_units,
                                    max_sym_chars=max_sym_chars,
                                    max_name_chars=max_name_chars,
                                    fmt_val="10.5g", pretty_print=pretty_print)
        T_grad_str = self._vartostr(var=self.T_grad, var_str='T_grad',
                                    to_units=T_grad_units,
                                    max_sym_chars=max_sym_chars,
                                    max_name_chars=max_name_chars,
                                    fmt_val="10.5g", pretty_print=pretty_print)
        p_base_str = self._vartostr(var=self.p_base, var_str='p_base',
                                    to_units=p_base_units,
                                    max_sym_chars=max_sym_chars,
                                    max_name_chars=max_name_chars,
                                    fmt_val="10.5g", pretty_print=pretty_print)

        # Assemble output string
        repr_str = (f"{H_base_str}\n{T_base_str}\n"
                    f"{T_grad_str}\n{p_base_str}")
        return repr_str
    
    @_property_decorators
    def H_base(self):
        """Layer base geopotential altitude :math:`H_{base}` """
        return self._H_base

    @_property_decorators
    def T_base(self):
        """Layer base temperature :math:`T_{base}` """
        return self._T_base

    @_property_decorators
    def T_grad(self):
        """Layer base temperature gradient :math:`T_{grad}` """
        return self._T_grad

    @_property_decorators
    def p_base(self):
        """Layer base pressure :math:`p_{base}` """
        return self._p_base


class Atmosphere(DimensionalData):
    """Compute quantities from International Civil Aviation Organization (ICAO)
    1993, which extends the US 1976 Standard Atmospheric Model to 80 km.

    Usage:
        from flightcondition import Atmosphere, unit

        # Compute atmospheric data for a scalar or array of altitudes
        h = [0.0, 44.2, 81.0] * unit('km')
        atm = Atmosphere(h)

        # Uncomment to print all atmospheric quantities:
        #print(f"\n{atm}")

        # Uncomment to print while specifying abbreviated output in US units:
        #print(f"\n{atm.tostring(units='US')}")

        # See also the linspace() function from numpy, e.g.
        # h = linspace(0, 81.0, 82) * unit('km')

        # Access individual properties and convert to desired units: "
        p, T, rho, nu, a, k = atm.p, atm.T, atm.rho, atm.nu, atm.a, atm.k
        print(f"\nThe pressure in psi is {p.to('psi'):.3g}")
        # >>> The pressure in psi is [14.7 0.024 0.000129] psi

        # Compute additional properties such as mean free path
        # Explore the class data structure for all options
        print( f"\nThe mean free path = {atm.MFP:.3g}")
        # >>> The mean free path = [7.25e-08 4.04e-05 0.00564] yd
    """

    names_dict = {
        'h': 'geometric_altitude',
        'H': 'geopotential_altitude',
        'p': 'pressure',
        'T': 'temperature',
        'rho': 'density',
        'a': 'sound_speed',
        'g': 'gravity',
        'wdir': 'wind_direction',
        'wspd': 'wind_speed',
    }

    def __init__(self, H=None, units=None, **kwargs):
        """Input geopotential altitude - object contains the corresponding
        atmospheric quantities.

        See class definition for special hidden arguments such as `h_kft` to
        input a scalar that automatically converts to the kilo-foot dimension,
        or `h_km` for kilometers.

        Args:
            H (length): Geopotential altitude - aliases are 'alt', 'altitude'
            units (str): Set to 'US' for US units or 'SI' for SI
        """
        # Check units and set initially
        if units in dir(unit.sys):
            self.units = units
        else:
            self.units = 'SI'

        # Compute altitude bounds
        self._H_min = Atmo.H_base[0]
        self._H_max = Atmo.H_base[-1]
        self._h_min = __class__._h_from_H(self._H_min)
        self._h_max = __class__._h_from_H(self._H_max)

        # Process altitude input
        # Check for hidden aliases
        H_aliases = ['alt', 'altitude']
        if H is None:
            H = __class__._arg_from_alias(H_aliases, kwargs)

        # Check if special H_kft syntactic sugar is used
        H_kft_aliases = ['H_kft', 'z_kft', 'kft']
        if H is None:
            H_kft = __class__._arg_from_alias(H_kft_aliases, kwargs)
            if H_kft is not None:
                H = H_kft * unit('kft')

        # Check if special H_km syntactic sugar is used
        H_km_aliases = ['H_km', 'z_km', 'km']
        if H is None:
            H_km = __class__._arg_from_alias(H_km_aliases, kwargs)
            if H_km is not None:
                H = H_km * unit('km')

        # Default to 0 kft
        if H is None:
            H = 0 * unit('ft')
        self.H = H

        # Further process unit system
        if units not in dir(unit.sys):  # check if usable system
            self.units = 'SI'

        # Initialize access by full quantity name through .byname.<name>
        self.byname = AliasAttributes(
            varsobj_arr=[self, ], names_dict_arr=[__class__.names_dict, ])

    def tostring(self, units=None, max_sym_chars=None,
                 max_name_chars=None, pretty_print=True):
        """String representation of data structure.

        Args:
            units (str): Set to 'US' for US units or 'SI' for SI
            max_sym_chars (int): Maximum characters in symbol name
            max_name_chars (int): Maximum characters iin full name
            pretty_print (bool): Pretty print output

        Returns:
            str: String representation of class object
        """
        if units is not None:
            self.units = units

        if self.units == 'US':
            h_units   = 'kft'
            H_units   = 'kft'
            p_units   = 'lbf/ft^2'
            T_units   = 'degR'
            rho_units = 'slug/ft^3'
            a_units   = 'ft/s'
            g_units   = 'ft/s^2'
            wdir_units   = 'degrees'
            wspd_units   = 'ft/s'
        else:  # SI units
            h_units   = 'km'
            H_units   = 'km'
            p_units   = 'Pa'
            T_units   = 'degK'
            rho_units = 'kg/m^3'
            a_units   = 'm/s'
            g_units   = 'm/s^2'
            wdir_units   = 'degrees'
            wspd_units   = 'm/s'

        # Insert longer variable name into output
        if max_sym_chars is None:
            max_sym_chars = max([len(v) for v in self.names_dict.keys()])
        if max_name_chars is None:
            max_name_chars = max([len(v) for v in self.names_dict.values()])

        h_str = self._vartostr(var=self.h, var_str='h', to_units=h_units,
                               max_sym_chars=max_sym_chars,
                               max_name_chars=max_name_chars,
                               fmt_val="10.5g", pretty_print=pretty_print)
        H_str = self._vartostr(var=self.H, var_str='H', to_units=H_units,
                               max_sym_chars=max_sym_chars,
                               max_name_chars=max_name_chars,
                               fmt_val="10.5g", pretty_print=pretty_print)
        p_str = self._vartostr(var=self.p, var_str='p', to_units=p_units,
                               max_sym_chars=max_sym_chars,
                               max_name_chars=max_name_chars,
                               fmt_val="10.5g", pretty_print=pretty_print)
        T_str = self._vartostr(var=self.T, var_str='T', to_units=T_units,
                               max_sym_chars=max_sym_chars,
                               max_name_chars=max_name_chars,
                               fmt_val="10.5g", pretty_print=pretty_print)
        fmt_val_rho = "10.4e" if units == 'US' else "10.5g"
        rho_str = self._vartostr(var=self.rho, var_str='rho',
                                 to_units=rho_units,
                                 max_sym_chars=max_sym_chars,
                                 max_name_chars=max_name_chars,
                                 fmt_val=fmt_val_rho,
                                 pretty_print=pretty_print)
        a_str = self._vartostr(var=self.a, var_str='a', to_units=a_units,
                               max_sym_chars=max_sym_chars,
                               max_name_chars=max_name_chars,
                               fmt_val="10.5g", pretty_print=pretty_print)
        g_str = self._vartostr(var=self.g, var_str='g', to_units=g_units,
                               max_sym_chars=max_sym_chars,
                               max_name_chars=max_name_chars,
                               fmt_val="10.5g", pretty_print=pretty_print)
        wdir_str = self._vartostr(var=self.wdir, var_str='wdir', to_units=wdir_units,
                               max_sym_chars=max_sym_chars,
                               max_name_chars=max_name_chars,
                               fmt_val="10.5g", pretty_print=pretty_print)
        wspd_str = self._vartostr(var=self.wspd, var_str='wspd', to_units=wspd_units,
                               max_sym_chars=max_sym_chars,
                               max_name_chars=max_name_chars,
                               fmt_val="10.5g", pretty_print=pretty_print)

        # Assemble output string
        repr_str = (f"{h_str}\n{H_str}\n{p_str}\n{T_str}\n{rho_str}\n"
                    f"{a_str}\n{g_str}")#\n{wdir_str}\n{wspd_str}")

        return repr_str

    @staticmethod
    @unit.wraps(unit.m, (unit.m, unit.m))
    def _H_from_h(h, R_earth=Phys.R_earth):
        """Convert geometric to geopotential altitude.

        :math:`H = \\frac{R_{earth} h}{R_{earth} + h}`

        Args:
            h (length): Geometric altitude
            R_earth (length): Radius of the Earth

        Returns:
            length: Geopotential altitude
        """
        H = R_earth*h/(R_earth + h)
        return H

    @staticmethod
    @unit.wraps(unit.m, (unit.m, unit.m))
    def _h_from_H(H, R_earth=Phys.R_earth):
        """Convert geopotential to geometric altitude.

        :math:`h = \\frac{R_{earth} H}{R_{earth} - H}`

        Args:
            H (length): Geopotential altitude
            R_earth (length): Radius of the Earth

        Returns:
            length: Geometric altitude
        """
        h = R_earth*H/(R_earth - H)
        return h

    @staticmethod
    @unit.wraps(unit('W/m/K').units, unit.K)
    def _thermal_conductivity(T_K):
        """Compute thermal conductivity.

        Args:
            T_K (temperature): Dimensional Temperature at altitude, which is
                automatically converted to Kelvin

        Returns:
            power/length/temperature: Geometric altitude in W/m/K
        """
        # Empirical formula requires T in Kelvin
        c1 = 0.002648151
        k = c1*T_K**1.5 / (T_K + (245.4*10**(-12/T_K)))
        return k

    @property
    def units(self):
        """Get unit system to use: 'SI', 'US', etc.  Available unit systems
        given by dir(unit.sys).

        Returns:
            str: Unit system
        """
        return self._units

    @units.setter
    def units(self, units):
        """Set unit system to use: 'SI', 'US', etc.  Available unit systems
        given by dir(unit.sys).

        Args:
            units (str): Unit system
        """
        if units not in dir(unit.sys):
            warnings.warn(f"'{units} is not available. Try one of the "
                          f"following: {dir(unit.sys)}")
            return
        else:
            self._units = units
            unit.default_system = units
            if hasattr(self, 'layer'):
                self.layer.units = units

    @_property_decorators
    def h(self):
        """Get geometric altitude :math:`h`

        Returns:
            length: Geometric altitude
        """
        return self._h

    @_property_decorators
    def H(self):
        """Geopotential altitude :math:`H` """
        return self._H

    @H.setter
    def H(self, H):
        """Set geopotential altitude :math:`H`

        Check that input is of type Quantity from pint package. Check that
        input is length dimension.  Check bounds.  Format as array even if
        scalar input.

        Args:
            H (length): Input scalar or array of altitudes
        """
        tofloat = 1.0
        H = np.atleast_1d(H) * tofloat

        H = H.magnitude * unit(str(H.units))  # force local unit registry

        if len(np.shape(H)) > 1:
            raise TypeError("Input must be scalar or 1-D array.")

        if (H < self._H_min).any() or (self._H_max < H).any():
            raise ValueError(
                f"Input altitude is out of bounds "
                f"({self._H_min:.5g} < h < {self._H_max:.5g})"
            )

        # Update quantities
        self._H = H
        self._h = self._h_from_H(self.H)
        self.layer = Layer(self.H, units=self.units)
        self._T = None
        self._p = None
        self._wdir = None
        self._wspd = None

    @_property_decorators
    def p(self):
        """Air pressure :math:`p` """
        # Only compute p if not set by user
        if self._p is not None:
            p = self._p
        else:
            H_base = np.atleast_1d(self.layer.H_base)
            T_base = np.atleast_1d(self.layer.T_base)
            T_grad = np.atleast_1d(self.layer.T_grad)
            p_base = np.atleast_1d(self.layer.p_base)

            H = np.atleast_1d(self.H)
            T = np.atleast_1d(self.T)
            g_0 = Phys.g
            R_star = Phys.R_star

            p = np.zeros_like(H) * unit('Pa')

            # Pressure equation changes between T_grad == 0 and T_grad != 0
            s = T_grad == 0
            p[s] = p_base[s]*np.exp((-g_0/(R_star*T[s]))*(H[s] - H_base[s]))

            s = T_grad != 0
            p[s] = p_base[s]*(
                1 + (T_grad[s]/T_base[s])*(H[s] - H_base[s])
            )**((1/T_grad[s])*(-g_0/R_star))

        return p

    @p.setter
    def p(self, p):
        """Override ambient air pressure """
        # Check that p is same size as H
        if np.size(p) != np.size(self._H):
            raise AttributeError("Input array must be same size as altitude")
        self._p = p

    @_property_decorators
    def T(self):
        """Ambient air temperature :math:`T` """
        # Only compute T if not user set
        if self._T is not None:
            T = self._T
        else:
            T_grad = np.atleast_1d(self.layer.T_grad)
            H_base = np.atleast_1d(self.layer.H_base)
            T_base = np.atleast_1d(self.layer.T_base)
            T = T_base + T_grad*(self.H - H_base)

        return T

    @T.setter
    def T(self, T):
        """Override ambient air temperature """
        # Check that T is same size as H
        if np.size(T) != np.size(self._H):
            raise AttributeError("Input array must be same size as altitude")
        self._T = T

    @_property_decorators
    def rho(self):
        """Ambient air density :math:`\\rho` """
        p = self.p
        T = self.T
        R_star = Phys.R_star
        rho = p/(R_star*T)
        return rho

    @_property_decorators
    def a(self):
        """Ambient speed of sound :math:`a` """
        T = self.T
        gamma = Phys.gamma
        R_star = Phys.R_star
        a = np.sqrt(gamma*R_star*T)
        return a

    @_property_decorators
    def g(self):
        """Gravitational acceleration at altitude :math:`g` """
        h = self.h
        g_0 = Phys.g
        R_earth = Phys.R_earth
        g = g_0*(R_earth/(R_earth + h))**2
        return g


    @_property_decorators
    def wdir(self):
        """Wind direction :math:`wdir` """
        # Only compute wdir if not user set
        if self._wdir is not None:
            wdir = self._wdir
        else:
            wdir = Atmo.wdir_0

        return wdir

    @wdir.setter
    def wdir(self, wdir):
        """Set wind direction """
        # Check that wdir is same size as H
        if np.size(wdir) != np.size(self._H):
            raise AttributeError("Input array must be same size as altitude")
        self._wdir = wdir

        
    @_property_decorators
    def wspd(self):
        """Wind speed :math:`wspd` """
        # Only compute wspd if not user set
        if self._wspd is not None:
            wspd = self._wspd
        else:
            wspd = Atmo.wpsd_0

        return wspd

    @wspd.setter
    def wspd(self, wspd):
        """Set wind speed """
        # Check that wspd is same size as H
        if np.size(wspd) != np.size(self._H):
            raise AttributeError("Input array must be same size as altitude")
        self._wspd = wspd