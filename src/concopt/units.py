from pint import UnitRegistry
import pint_pandas

unit = UnitRegistry(system='mks')

unit.default_format = '~P'
dimless = unit('dimensionless')

pint_pandas.PintType.ureg = unit
ppunit = pint_pandas.PintType.ureg