from pint import UnitRegistry    

def convert_dms_to_dd(tude):
    multiplier = 1 if tude[-1] in ['N', 'E'] else -1
    dd = sum(float(x) / 60 ** n for n, x in enumerate(tude[:-1].split('-')))
    return multiplier * dd

def convert_degc_to_degK(degC):
    ureg = UnitRegistry()
    Q_ = ureg.Quantity
    temp = Q_(degC, ureg.degC)
    return temp.to('degK').magnitude