from scipy.constants import N_A

rho = 2.17 # g/cm^3
m_m = 58.44 # g/mol
a = (5.6096)**3 # Å^3
a_meter = a * 10**(-30) # m^3

z = (N_A * rho * a) / m_m
print(z)