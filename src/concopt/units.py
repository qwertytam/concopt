#!/usr/bin/env python
"""Provide dimensioned units for math operations

Author: Tom Marshall

:copyright: 2023 Tom Marshall
:license: MIT License, see LICENSE for more details.
"""

from pint import UnitRegistry
import pint_pandas

unit = UnitRegistry(system='mks')

unit.default_format = '~P'
dimless = unit('dimensionless')

pint_pandas.PintType.ureg = unit
ppunit = pint_pandas.PintType.ureg