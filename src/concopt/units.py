from pint import UnitRegistry

unit = UnitRegistry(system='mks')

unit.default_format = '~P'
dimless = unit('dimensionless')
