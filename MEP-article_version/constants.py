"""This module defines project-level constants."""

# Physical constants
Cp = 1005.  # Cp_air = air calorific coefficient [J.kg-1.K-1]
g = 9.80665  # = g0 = standard Earth gravity [m.s-2]
massMolAir = 28.97  # molar mass of air [g/mol]
massMolH2O = 18.01524  # molar mass of H2O
massMolCO2 = 44.0  # molar mass of CO2
massVolO3 = 0.002144  # mass per unit volume of O3 [g.m-3]
# TODO: It might be 2.2e6. Check that.
La = 2.5e6  # latent energy coefficient
# TODO: one day, put the dependence of R with humidity : PV = (1+0.61q)T, where q = m_water / m_air
R = 287.  # specific air constant [J.K-1.kg-1]
Rgaz = 8.314472  # gas constant [J.K-1.mol-1]
p0 = 1013.25  # see level pressure
irSigma = 5.6704e-8  # sigma = Stefan-Boltzmann constant [W.m-2.K-1]
Sover4 = 342.  # S/4 solar constant
Eref = Sover4  # reference energy, used to have adimensioned variables closer to 1
Tref = pow(Sover4/irSigma, 0.25)  # reference temperature, used to have adimensioned variables closer to 1
Lref = 31570.56  # T = 1 year = 3600*24*365.4 s, L (m/yr) = l (kg/m2/s)*T(s)/rho(kg/m3))
