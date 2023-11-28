import numpy as np
import pandas as pd
import pint_pandas
from scipy.interpolate import interpn

fn = './data/conc_cas_limit.csv'
conc_cas_limit = pd.read_csv(
    fn,
    header=0,
    dtype=np.int32,
    )

def get_cas_limit(altitudes, weight):
    """ Get calibrated air speed limit for given altitude(s) and weight
    
    Arguments:
        altitudes -- altitudes to get calibrated air speed limit for
        weight -- aircraft weight to get cas limit for
    
    Returns
        calibrated air speed limit
    """
    # altitudes = np.array(altitudes.pint.to('feet').values.quantity.m)
    print(f"\naltitudes:\n{altitudes}\n")
    weights = np.array(weight.to('tons').magnitude,
                      len(altitudes))
    print(f"\nweights:\n{weights}\n")
    alt_wgts = np.column_stack(altitudes, weights)
    print(f"\nalt_wgts:\n{alt_wgts}\n")
    x_alts = np.array(conc_cas_limit['alt']).astype(np.float32) # in feet
    print(f"\nx_alts:\n{x_alts}\n")
    y_wgts = np.array(conc_cas_limit.columns.values)[1:].astype(np.float32)  # in tons
    print(f"\ny_wgts:\n{y_wgts}\n")
    z_cas = np.array(conc_cas_limit.iloc[:, 1:])  # in knots
    print(f"\nz_cas:\n{z_cas}\n")
    cas_limit = interpn((x_alts, y_wgts), z_cas, alt_wgts)
    print(f"\ncas_limit:\n{cas_limit}\n")
    return cas_limit

def get_mach_limit(altitudes):
    """ Get mach limit for given altitude(s)
    
    Arguments:
        altitudes -- altitudes to get calibrated air speed limit for
    
    Returns
        mach limit
    """
    return 2.04  # max mach for all altitudes


def get_total_temp_limit(altitudes):
    """ Get total aka stagnation temperature limit for given altitude(s)
    
    Arguments:
        altitudes -- altitudes to get calibrated air speed limit for
    
    Returns
        total temperature in degrees Celsius
    """
    return 127  # max total temperature in degrees celsius for all altitudes
