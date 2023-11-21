from flightcondition.constants import PhysicalConstants as physicals
from flightcondition.constants import AtmosphereConstants as atmosphericals
from flightcondition import FlightCondition, unit

mins_per_deg = 60

gamma = physicals.gamma_air.magnitude

p_0 = atmosphericals.p_0.magnitude

a_0 = FlightCondition(h=0*unit('km'), units='SI').a