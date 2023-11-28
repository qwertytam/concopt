from concopt.units import unit

def convert_dms_to_dd(tude):
    """ Convert a latitude or longitude degrees-minutes-seconds to decimal
    
    Arguments:
    tude -- longitude or latitude to convert to decimal. Inpurt format is 
        'dd-mm-ss.sssN', where N is N, E, S, W is cardinal direction i.e.,
        North, East, South or West
        
    Returns:
    dd.dddd in decimal format with N and E positive, W and S negative by
        convention
    """
    multiplier = 1 if tude[-1] in ['N', 'E'] else -1
    sexagesimal_base = 60
    dd = sum(
        float(x) / sexagesimal_base ** n for n, x in enumerate(tude[:-1].split('-'))
        )
    return multiplier * dd*unit('degree')