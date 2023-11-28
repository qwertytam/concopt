#!/usr/bin/env python
"""Provide dimensioned units for math operations

Author: Tom Marshall

:copyright: 2023 Tom Marshall
:license: MIT License, see LICENSE for more details.
"""
from functools import wraps
from pint import UnitRegistry
import pint_pandas

unit = UnitRegistry(system='mks')

unit.default_format = '~P'
dimless = unit('dimensionless')

pint_pandas.PintType.ureg = unit
ppunit = pint_pandas.PintType.ureg

def to_base_units_wrapper(func):
    """Function decorator to convert output variable units to base units.

    Args:
        func (callable): Function to wrap

    Returns:
        callable: Called, wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        if output is not None:
            output.ito_base_units()
        return output
    return wrapper
