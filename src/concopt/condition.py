#!/usr/bin/env python
"""Easily convert between Mach number, true airspeed (TAS), calibrated airspeed
(CAS), and equivalent airspeed (EAS) for given altitude(s).  Additional flight
condition data and atmospheric data is computed.

Dependencies: numpy, pint

Author: Matthew C. Jones
Email: matt.c.jones.aoe@gmail.com

:copyright: 2021 Matthew C. Jones
:license: MIT License, see LICENSE for more details.
"""

import warnings

import numpy as np

from concopt.atmosphere import Atmosphere
from concopt.constants import PhysicalConstants as Phys
from concopt.common import AliasAttributes, _property_decorators
from concopt.isentropicflow import IsentropicFlow
from concopt.nondimensional import NonDimensional
from concopt.units import unit, dimless, check_dimensioned, \
    to_base_units_wrapper

class FlightCondition(Atmosphere):
    """Easily convert between Mach number, true airspeed (TAS), calibrated
    airspeed (CAS), and equivalent airspeed (EAS) for given altitude(s).
    Additional flight condition data and atmospheric data is computed.

    All inputs must be dimensional unit quantities.

    Usage:
        from flightcondition import FlightCondition, unit

        # Compute flight condition at 3 km, Mach 0.5
        fc = FlightCondition(h=3*unit('km'), M=0.5)

        # Uncomment to print summary of flight condition quantities:
        #print(f"{fc}")

        # Uncomment to print abbreviated output in US units:
        #print(f"\n{fc.tostring(full_output=False, units="US")}")

        # Convert true, calibrated, equivalent airspeeds
        KTAS = fc.TAS.to('knots')
        KCAS = fc.CAS.to('knots')
        KEAS = fc.EAS.to('knots')
        print(f"Flying at {KTAS.magnitude:.4g} KTAS,"
            f" which is {KCAS.magnitude:.4g} KCAS,"
            f" or {KEAS.magnitude:.4g} KEAS")
        # >>> Flying at 319.4 KTAS, which is 277.7 KCAS, or 275.1 KEAS

        # Access atmospheric data (see Atmosphere class for more)
        h, p, T, rho, nu, a = fc.h, fc.p, fc.T, fc.rho, fc.nu, fc.a
        print(f"The ambient temperature at {h.to('km'):.4g} is {T:.4g}")
        # >>> The ambient temperature at 3 km is 268.7 K

        # Change airspeed to 300 KEAS and altitude to 12 kft
        fc.EAS = 300 * unit('knots')
        fc.h = 12 * unit('kft')
        #print(f"{fc}")  # uncomment to print output

        # Recompute for a range of altitudes at 275.14 knots-equivalent
        # airspeed with a characteristic length scale of 10 meters
        fc = FlightCondition(h=[0, 9.8425, 20]*unit('kft'),
                            EAS=275.14*unit('kt'),
                            L=10*unit('m'))

        # Compute additional derived quantities - explore the class for more!
        print(f"\nThe dynamic pressure in psi is {fc.q_inf.to('psi'):.3g}")
        # >>> The dynamic pressure in psi is [1.78 1.78 1.78] psi
        print(f"The Reynolds number is {fc.Re:.3g}")
        # >>> The Reynolds number is [9.69e+07 8.82e+07 7.95e+07]
        h_yplus100 = fc.wall_distance_from_yplus(100)
        print(f"The wall distance where y+=100 is {h_yplus100.to('in'):.3g}")
        # >>> The wall distance where y+=100 is [0.0126 0.0138 0.0153] in

        # Alternatively access quantities by their full name
        print(fc.TAS == fc.byname.true_airspeed)
        # >>> [ True  True  True]

        # Or by their sub-categories: `byalt`, `byvel`, or `bylen`
        print(fc.byvel.TAS == fc.byvel.byname.true_airspeed)
        # >>> [ True  True  True]
    """
    # Define dictionaries mapping variables to their full names
    _byalt_names = Atmosphere.names_dict
    _byvel_names = {
        'TAS': 'true_airspeed',
        'CAS': 'calibrated_airspeed',
        'M': 'mach_number',
        'q_inf': 'dynamic_pressure',
        'q_c': 'impact_pressure',
        'p0': 'stagnation_pressure',
        'T0': 'stagnation_temperature',
    }

    names_dict = {}
    names_dict.update(_byalt_names)
    names_dict.update(_byvel_names)

# =========================================================================== #
#                       GENERAL FUNCTIONS & PROPERTIES                        #
# =========================================================================== #

    def __init__(
        self, H=None, p=None, M=None, TAS=None, CAS=None,
        units=None, **kwargs,
    ):
        """Constructor based on altitude and input velocity in terms of Mach
        number, TAS, CAS, or EAS.  Input altitude, one format of velocity, and
        length scale.  Reynolds number can be input as an alternative to either
        velocity or length scale but not both.  All inputs must be dimensional
        unit quantities.

        Args:
            H (length): Geopotential altitude - aliases are 'alt', 'altitude'
            p (pressure): Pressure altitude - aliases are 'p_alt',
                'pressure_altitude'
            M (dimless): Velocity as Mach number - aliases are 'mach', 'Mach',
                'M_inf', 'mach_number'
            TAS (speed): Velocity as true airspeed - aliases are 'tas',
                'true_airspeed', 'U_inf', 'V_inf'
            CAS (speed): Velocity as calibrated airspeed - aliases are 'cas',
                'calibrated_airspeed'
            units (str): Set to 'US' for US units or 'SI' for SI
            full_output (bool): Set to True for full output
        """
        # Initialize Atmosphere super class
        super().__init__(H=H, units=units, **kwargs)
        self._byalt_tostring = super().tostring

        # Computer sea level properties
        self._atm0 = Atmosphere(H=0*unit('kft'))

        # Need to cycle self.units to properly set default units
        self.units = self.units

        # Check for hidden aliases
        M_aliases = ['mach', 'Mach', 'M_inf', 'mach_number']
        if M is None:
            M = __class__._arg_from_alias(M_aliases, kwargs)
        TAS_aliases = ['tas', 'true_airspeed', 'U_inf', 'V_inf', 'VTAS',
                       'vtas']
        if TAS is None:
            TAS = __class__._arg_from_alias(TAS_aliases, kwargs)
        CAS_aliases = ['cas', 'calibrated_airspeed', 'VCAS', 'vcas']
        if CAS is None:
            CAS = __class__._arg_from_alias(CAS_aliases, kwargs)

        # By default hold CAS constant when altitude is changed
        self._holdconst_vel_var = 'CAS'

        # Assign input velocity quantity
        if M is not None:
            self.M = M
        elif TAS is not None:
            self.TAS = TAS
        elif CAS is not None:
            self.CAS = CAS
        else:
            # Use Mach=0 if no velocity is input
            self.M = 0*dimless

        # Check that computations are within valid Mach number limits
        M_ = np.atleast_1d(self.M)
        self._mach_min = 0 * dimless
        self._mach_max = 3 * dimless
        if (M_ < self._mach_min).any() or (self._mach_max < M_).any():
            raise ValueError(
                f"Mach number is out of bounds "
                f"({self._mach_min:.5g} < M_ < {self._mach_max:.5g})"
            )

        # Set up behind-the scenes quantity access features
        # Set up sub-categories asdict functions
        self._byalt_asdict = self._asdict_template(
            names_dict=self._byalt_names)
        self._byvel_asdict = self._asdict_template(
            names_dict=self._byvel_names)

        # Set alias references by name _byalt_byname.<name>
        self._byalt_byname = AliasAttributes(
            varsobj_arr=[self, ],
            names_dict_arr=[self._byalt_names, ]
        )

        # Set alias references by name _byvel_byname.<name>
        self._byvel_byname = AliasAttributes(
            varsobj_arr=[self, ],
            names_dict_arr=[self._byvel_names, ]
        )

        # Set alias references by name generally .byname.<name>
        self.byname = AliasAttributes(
            varsobj_arr=[self, self, self],
            names_dict_arr=[
                self._byalt_names,
                self._byvel_names,
            ]
        )

        # Set variable aliases by altitude properties to .byalt.<var>
        byalt_vars = {key: key for key in self._byalt_names.keys()}
        self.byalt = AliasAttributes(
            varsobj_arr=[self, self, self, self, self, self, ],
            names_dict_arr=[
                byalt_vars,
                {"_byalt_names": "names_dict"},
                {"_byalt_tostring": "tostring"},
                {"full_output": "full_output"},
                {"_byalt_byname": "byname"},
                {"_byalt_asdict": "asdict"},
            ]
        )

        # Set variable aliases by velocity properties to .byvel.<var>
        byvel_vars = {key: key for key in self._byvel_names.keys()}
        self.byvel = AliasAttributes(
            varsobj_arr=[self, self, self, self, self, self, ],
            names_dict_arr=[
                byvel_vars,
                {"_byvel_names": "names_dict"},
                {"_byvel_tostring": "tostring"},
                {"full_output": "full_output"},
                {"_byvel_byname": "byname"},
                {"_byvel_asdict": "asdict"},
            ]
        )

        # Set all property variable references to .byvar.<var>
        self.byvar = AliasAttributes(
            varsobj_arr=[self.byalt, self.byvel, self],
            names_dict_arr=[byalt_vars, byvel_vars]
        )

    def tostring(self, units=None, pretty_print=True):
        """String representation of data structure.

        Args:
            full_output (bool): Set to True for full output
            units (str): Set to 'US' for US units or 'SI' for SI
            pretty_print (bool): Pretty print output

        Returns:
            str: String representation
        """

        # Determine units
        if units is not None:
            self.units = units

        # Check that quantity array sizes are consistent, otherwise raise error
        self._check_compatible_array_sizes(raise_warning=True,
                                           raise_error=True)

        # Determine maximum characters to add spaces for and assemble string
        max_sym_chars = max([  # length of variables
            max([len(v) for v in self._byalt_names.keys()]),
            max([len(v) for v in self._byvel_names.keys()]),
        ])
        max_name_chars = max([  # length of variable names
            max([len(v) for v in self._byalt_names.values()]),
            max([len(v) for v in self._byvel_names.values()]),
        ])

        # Build output strings from sub-categories
        alti_str = self._byalt_tostring(self.units,
                                        max_sym_chars=max_sym_chars,
                                        max_name_chars=max_name_chars,
                                        pretty_print=pretty_print)
        spee_str = self._byvel_tostring(self.units,
                                        max_sym_chars=max_sym_chars,
                                        max_name_chars=max_name_chars,
                                        pretty_print=pretty_print)

        unit_str = self.units
        top_hdr = f"              Flight Condition (units={unit_str})"
        lin_str = "==========================================================="
        alt_hdr = "------------------  Altitude Quantities  ------------------"
        vel_hdr = "------------------  Velocity Quantities  ------------------"

        repr_str = (f"{lin_str}\n{top_hdr}\n{lin_str}"
                    f"\n{alt_hdr}" f"\n{alti_str}"
                    f"\n{vel_hdr}" f"\n{spee_str}"
                    )
        return repr_str

    @staticmethod
    def _check_compatible_array_size(arr1, arr2,
                                     arr_name1="Arr1", arr_name2="Arr2",
                                     raise_warning=False, raise_error=False):
        """Check that two arrays have valid sizes. Non-singular arrays must be
        equal in size.

        Args:
            arr1 (array): Array 1
            arr2 (array): Array 2
            arr_name1 (str): Array name 1
            arr_name2 (str): Array name 2
            raise_error (bool): Whether or not to raise an error.
            raise_warning (bool): Whether or not to raise a warning.

        Returns:
            bool: True if valid sizes else false.
        """
        arr_size1 = np.size(arr1)
        arr_size2 = np.size(arr2)
        msg = (f"{arr_name1} array size {arr_size1} and {arr_name2} array size"
               f" {arr_size2} are incompatible! Non-singular arrays must be "
               "equal in size.")
        if arr_size1 > 1 and arr_size2 > 1:
            if not arr_size1 == arr_size2:
                if raise_warning:
                    warnings.warn(msg.format(arr_size1, arr_size2))
                if raise_error:
                    raise AttributeError(msg.format(arr_size1, arr_size2))
                return False
        return True

    def _check_compatible_array_sizes(self, raise_warning=False,
                                      raise_error=False):
        """Check that altitude, velocity, and length array sizes are valid.  If
        two sub-category sizes are non-equal and greater than 1, the underlying
        functions of the class may fail.  This check can detect problems
        beforehand.

        Args:
            raise_error (bool): Whether or not to raise an error.
            raise_warning (bool): Whether or not to raise a warning.

        Returns:
            bool: True if valid sizes else false.
        """
        iscompatible = True
        msg = "{} arrays of size {} and {} arrays of size {} are incompatible."

        if not __class__._check_compatible_array_size(self.H, self.M):
            warnings.warn(msg.format("Altitude", np.size(self.H), "Velocity",
                          np.size(self.M)))
            iscompatible = False

        if raise_error and not iscompatible:
            raise AttributeError("Aborting! Array sizes are incompatible. "
                                 "See previous warnings.")

        return iscompatible

    @staticmethod
    def _reshape_arr1_like_arr2(arr1, arr2, raise_warning=False):
        """Reshape arr1 to match size of arr2 if size(arr1)==1.

        Args:
            arr1 (object): Array 1
            arr2 (object): Array 2
            raise_warning (bool): Whether or not to raise a warning.
        """
        if np.size(arr2) > 1 and np.size(arr1) == 1:
            if (np.size(arr1) != np.size(arr2)):
                # Scale up length quantities to match altitude size
                arr1 = np.ones_like(arr2)*arr1
        return arr1

    @classmethod
    def _preprocess_arr(cls, input_arr, input_alt=None, input_vel=None):
        """Check that input is correctly typed, then size input array. If
        scalar, leave as scalar.

        Args:
            input_arr (object): Input array (or scalar)
            input_alt (object): Input array of altitude level
            input_vel (object): Input array of velocity level (when assessing
                length scale as input_arr)

        Returns:
            object: Sized array (or scalar)
        """
        check_dimensioned(input_arr)

        # Force local unit registry
        input_arr = input_arr.magnitude * unit(str(input_arr.units))

        # Require input, base, and secondary arrays to be singular or
        # non-singular but equal in size
        sized_arr = cls._reshape_arr1_like_arr2(input_arr, input_alt)
        if input_vel is not None:
            sized_arr = cls._reshape_arr1_like_arr2(input_arr, input_vel)

        tofloat = 1.0
        sized_arr = (sized_arr * tofloat)

        return sized_arr

# =========================================================================== #
#                        LENGTH FUNCTIONS & PROPERTIES                        #
# =========================================================================== #
    @_property_decorators
    def H(self):
        """Override Atmosphere getter for altitude.

        Returns:
            length: Geopotential altitude
        """
        # return super().H  # doesn't work
        return super(FlightCondition, self.__class__).H.fget(self)

    @H.setter
    def H(self, H):
        """Override Atmosphere setter to properly dimension other quantities.

        Args:
            H (length): Input scalar or array of altitudes
        """
        # super().H = H  # doesn't work
        super(FlightCondition, self.__class__).H.fset(self, H)

        # Set airspeed and length array sizes if they are singular and altitude
        # is non-singular
        if hasattr(self, '_M'):
            if __class__._check_compatible_array_size(
                    arr1=H, arr2=self._M,
                    arr_name1=self.names_dict['h'],
                    arr_name2=self.names_dict['M'],
                    raise_warning=True):
                if self._holdconst_vel_var == 'M':
                    self.M = __class__._reshape_arr1_like_arr2(self._M, H)
                elif self._holdconst_vel_var == 'TAS':
                    self.TAS = __class__._reshape_arr1_like_arr2(self._TAS, H)
                elif self._holdconst_vel_var == 'CAS':
                    self.CAS = __class__._reshape_arr1_like_arr2(self._CAS, H)
                else:  # default to holding M constant
                    self.M = __class__._reshape_arr1_like_arr2(self._M, H)

# =========================================================================== #
#                       VELOCITY FUNCTIONS & PROPERTIES                       #
# =========================================================================== #
    def _process_input_velocity(self, vel_arr, vel_name="Velocity"):
        """Check that the input velocity array size is compatible. If so, shape
        altitude and length.

        Args:
            vel_arr (array): Velocity array

        Returns:
            bool: True if compatible else False
        """
        if hasattr(self, '_H'):
            if __class__._check_compatible_array_size(
                    arr1=vel_arr, arr2=self._H,
                    arr_name1=vel_name, arr_name2=self.names_dict["H"],
                    raise_warning=True, raise_error=True):
                self._H = __class__._reshape_arr1_like_arr2(self._H, vel_arr)

    def _byvel_tostring(self, full_output=None, units=None, max_sym_chars=None,
                        max_name_chars=None, pretty_print=True):
        """String representation of data structure.

        Args:
            full_output (bool): Set to True for full output
            units (str): Set to 'US' for US units or 'SI' for SI
            max_sym_chars (int): Maximum characters in symbol name
            max_name_chars (int): Maximum characters in full name
            pretty_print (bool): Pretty print output

        Returns:
            str: String representation
        """

        # Set default unit system
        if units is None:
            units = self.units

        if units == 'US':
            TAS_units   = 'knots'
            CAS_units   = 'knots'
            q_inf_units     = 'lbf/ft^2'
            q_c_units     = 'lbf/ft^2'
            p0_units    = 'lbf/ft^2'
            T0_units    = 'degR'
        else:  # default to SI units
            TAS_units   = 'm/s'
            CAS_units   = 'm/s'
            q_inf_units     = 'Pa'
            q_c_units     = 'Pa'
            p0_units    = 'Pa'
            T0_units    = 'degK'

        # Insert longer variable name into output
        if max_sym_chars is None:
            max_sym_chars = max([len(v) for v in self.names_dict.keys()])
        if max_name_chars is None:
            max_name_chars = max([len(v) for v in self.names_dict.values()])

        M_str = self._vartostr(
            var=self.M, var_str='M', to_units='',
            max_sym_chars=max_sym_chars, max_name_chars=max_name_chars,
            fmt_val="10.5g", pretty_print=pretty_print) + "\n"
        TAS_str = self._vartostr(
            var=self.TAS, var_str='TAS', to_units=TAS_units,
            max_sym_chars=max_sym_chars, max_name_chars=max_name_chars,
            fmt_val="10.5g", pretty_print=pretty_print) + "\n"
        CAS_str = self._vartostr(
            var=self.CAS, var_str='CAS', to_units=CAS_units,
            max_sym_chars=max_sym_chars, max_name_chars=max_name_chars,
            fmt_val="10.5g", pretty_print=pretty_print) + "\n"
        q_inf_str = self._vartostr(
            var=self.q_inf, var_str='q_inf', to_units=q_inf_units,
            max_sym_chars=max_sym_chars, max_name_chars=max_name_chars,
            fmt_val="10.5g", pretty_print=pretty_print) + "\n"
        q_c_str = self._vartostr(
            var=self.q_c, var_str='q_c', to_units=q_c_units,
            max_sym_chars=max_sym_chars, max_name_chars=max_name_chars,
            fmt_val="10.5g", pretty_print=pretty_print) + "\n"
        p0_str = self._vartostr(
            var=self.p0, var_str='p0', to_units=p0_units,
            max_sym_chars=max_sym_chars, max_name_chars=max_name_chars,
            fmt_val="10.5g", pretty_print=pretty_print) + "\n"
        T0_str = self._vartostr(
            var=self.T0, var_str='T0', to_units=T0_units,
            max_sym_chars=max_sym_chars, max_name_chars=max_name_chars,
            fmt_val="10.5g", pretty_print=pretty_print) + "\n"

        # Assemble output string
        repr_str = (f"{M_str}{TAS_str}{CAS_str}"
                        f"{q_inf_str}{q_c_str}{p0_str}{T0_str}")
        return repr_str

    @staticmethod
    def _impact_pressure(M, p):
        """Compute impact pressure

        Args:
            M (dimless): Mach number
            p (pressure): Static pressure

        Returns:
            pressure: Impact pressure
        """
        # p0 = p + q_c
        y = Phys.gamma_air
        p0 = IsentropicFlow.p0_by_p(M, y)*p
        q_c = p0 - p
        return q_c

    @to_base_units_wrapper
    def _M_from_TAS(self, TAS):
        """Compute Mach number from true airspeed.

        Args:
            TAS (speed): True airspeed

        Returns:
            dimless: Mach number
        """
        a_inf = self.a
        M = NonDimensional.mach_number(U=TAS, a=a_inf)
        return M

    @to_base_units_wrapper
    def _TAS_from_M(self, M):
        """Compute true airspeed from Mach number.

        Args:
            M (dimless): Mach number

        Returns:
            speed: True airspeed
        """
        a_inf = self.a
        TAS = NonDimensional.mach_velocity(M, a_inf)
        return TAS

    @to_base_units_wrapper
    def _q_c_from_CAS(self, CAS):
        """Compute impact pressure from calibrated airspeed (accounting for
           compressibility).

        Args:
            CAS (speed): Calibrated airspeed

        Returns:
            pressure: Impact pressure
        """
        a_h0 = self._atm0.a
        p_h0 = self._atm0.p
        # Account for compressibility with the isentropic flow equation
        M_ = NonDimensional.mach_number(U=CAS, a=a_h0)
        q_c = __class__._impact_pressure(M=M_, p=p_h0)
        return q_c

    @to_base_units_wrapper
    def _CAS_from_q_c(self, q_c):
        """Compute calibrated airspeed from impact pressure (accounting for
           compressibility).

        Args:
            CAS (speed): Calibrated airspeed

        Returns:
            pressure: Impact pressure

        """
        a_h0 = self._atm0.a
        p_h0 = self._atm0.p
        # Account for compressibility with the isentropic flow equation
        # M_should = __class__._isentropic_mach(p0=q_c, p=p_h0)  # DEPRECATED
        M = IsentropicFlow.M_from_p0_by_p((q_c+p_h0)/p_h0)
        CAS = NonDimensional.mach_velocity(M, a_h0)  # subsonic
        return CAS

    @to_base_units_wrapper
    def _M_from_q_c(self, q_c):
        """Compute Mach number from impact pressure.

        Args:
            q_c (pressure): Impact pressure

        Returns:
            dimless: Mach number
        """
        p_inf = self.p
        # Isentropic flow equation
        # M_should = __class__._isentropic_mach(p0=q_c, p=p_inf)  # DEPRECATED
        M = IsentropicFlow.M_from_p0_by_p((q_c+p_inf)/p_inf)

        return M

    @to_base_units_wrapper
    def _q_c_from_M(self, M):
        """Compute impact pressure from Mach number.

        Args:
            M: Mach number

        Returns:
            impact pressure
        """
        p_inf = self.p
        # Solve for impact pressure from isentropic flow equation:
        q_c = __class__._impact_pressure(M=M, p=p_inf)
        return q_c

    @staticmethod
    @to_base_units_wrapper
    def _q_inf_from_TAS(TAS, rho):
        """Compute dynamic pressure from true airspeed.

        Args:
            TAS (speed): True airspeed
            rho (density): Ambient density

        Returns:
            pressure: Dynamic pressure
        """
        q_inf = 0.5*rho*TAS**2
        return q_inf

    @_property_decorators
    def M(self):
        """Get Mach number :math:`M` """
        return self._M

    @M.setter
    def M(self, M):
        """Mach number :math:`M` """
        M *= dimless  # add dimless for raw float input
        self._M = __class__._preprocess_arr(M, input_alt=self.H)
        self._process_input_velocity(self._M, vel_name=self.names_dict['M'])

        self._TAS = self._TAS_from_M(self.M)
        self._q_c = self._q_c_from_M(self.M)
        self._CAS = self._CAS_from_q_c(self.q_c)

        # Out of all velocity quantities, hold this one constant over altitude
        self._holdconst_vel_var = 'M'

    @_property_decorators
    def TAS(self):
        """Get true airspeed. """
        return self._TAS

    @TAS.setter
    def TAS(self, TAS):
        """Set true airspeed. """
        self._TAS = __class__._preprocess_arr(TAS, input_alt=self.H)
        self._process_input_velocity(
            self._TAS, vel_name=self.names_dict['TAS'])

        self._M = self._M_from_TAS(TAS)
        self._EAS = self._EAS_from_TAS(self.TAS, self.M)
        self._q_c = self._q_c_from_M(self.M)
        self._CAS = self._CAS_from_q_c(self.q_c)

        # Out of all velocity quantities, hold this one constant over altitude
        self._holdconst_vel_var = 'TAS'

    @_property_decorators
    def CAS(self):
        """Get calibrated airspeed. """
        return self._CAS

    @CAS.setter
    def CAS(self, CAS):
        """Calibrated airspeed. """
        self._CAS = __class__._preprocess_arr(CAS, input_alt=self.H)
        self._process_input_velocity(
            self._CAS, vel_name=self.names_dict['CAS'])

        self._q_c = self._q_c_from_CAS(self.CAS)
        self._M = self._M_from_q_c(self.q_c)
        self._TAS = self._TAS_from_M(self.M)
        self._EAS = self._EAS_from_TAS(self.TAS, self.M)

        # Out of all velocity quantities, hold this one constant over altitude
        self._holdconst_vel_var = 'CAS'

    @_property_decorators
    def q_c(self):
        """Get impact pressure :math:`q_c`"""
        return self._q_c

    @_property_decorators
    def q_inf(self):
        """Get dynamic pressure :math:`q_\\infty`"""
        q_inf = __class__._q_inf_from_TAS(TAS=self.TAS, rho=self.rho)
        return q_inf

    @_property_decorators
    def p0(self):
        """Get stagnation pressure :math:`p_0`"""
        M = self.M
        p = self.p
        y = Phys.gamma_air
        p0 = IsentropicFlow.p0_by_p(M, y)*p
        return p0

    @_property_decorators
    def T0(self):
        """Get stagnation temperature :math:`T_0`"""
        M = self.M
        T = self.T
        y = Phys.gamma_air
        T0 = IsentropicFlow.T0_by_T(M, y)*T
        return T0

    @Atmosphere.T.setter
    def T(self, T):
        """Override ambient air temperature """
        # Check that T is same size as H
        if np.size(T) != np.size(self._H):
            raise AttributeError("Input array must be same size as altitude")
        self._T = T + 200*unit('delta_degC')
        self.M = self.M