from pint import UnitRegistry
from flightcondition.constants import PhysicalConstants as physicals
from flightcondition.constants import AtmosphereConstants as atmosphericals

ureg = UnitRegistry()
kts_per_ms = 1 * ureg.knots
kts_per_ms = kts_per_ms.to('m/s').magnitude

R_star = physicals.R_air.magnitude
gamma = physicals.gamma_air.magnitude

p_0 = atmosphericals.p_0.magnitude
T_ice = atmosphericals.T_ice.magnitude
rho_0 = atmosphericals.rho_0.magnitude

mins_per_deg = 60
