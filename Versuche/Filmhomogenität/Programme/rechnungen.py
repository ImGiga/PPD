import numpy as np
from scipy.constants import N_A
c_inf = 9.5
l = 1.54e-10
m_w = 35
m_n = 0.10415

n = m_w/m_n

r_rms_1 = np.sqrt(c_inf * l**2 *  n) 
print(r_rms_1)

const = 2.84e-8
c_0 = 153

r_rms_2 = const * (m_w/c_0)**(1/3)
print(r_rms_2)

c_rc = m_w / (N_A * (r_rms_1/2)**3)

print(c_rc)

rho = 10e-10

c_Wc = 2**(3/2) * m_w/(N_A * (rho * n * l)**(3/2))
print(c_Wc)

c_rr = 2**(3/2) * m_w/(N_A * (n*l)**3)
print(c_rr)

print(794*300/250)