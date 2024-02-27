import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from icecream import ic

rel_path = "C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/Röntgenbeugung/Daten/Messung 1/"
h = 4.135136*1e-18    # keV s
c = 299792458   # m / s
d = 1.93146 # A
lambda_Theo = np.array([0.17892, 0.17892, 0.17942, 0.17960, 0.18310, 0.18327, 0.18438, 0.18518, 0.20901, 0.21383,
                       1.06146, 1.06786, 1.09857, 1.24430, 1.26271, 1.28181, 1.30164, 1.42112, 1.47631, 1.48745]) # A

def calc_lambda(degree, n):
    return 2*d*np.sin(np.radians(degree))/n

def calc_theta(lam, n):
    return np.arcsin(n*lam / 2.0 / d) * 180 / np.pi

def correct_Theta(theta,m,b):
    return m*theta + b

def calc_mu(I, I0, lam):
    return -1.0 / lam * np.log(I / I0)

theta_Theo = np.arcsin(np.array([1.06786, 1.09857, 1.24430, 1.28181, 1.47631]) / (2.0*d)) * 360 / (2.0*np.pi)
omega_Mess = np.array([15.919, 16.452, 18.730, 19.32, 22.415])
m, b = np.polyfit(x=omega_Mess, y=theta_Theo, deg=1)
ic(m, b)
# Plot
# plt.plot(omega_Mess, theta_Theo, ls="none", marker="o", markersize=4) 
# plt.plot(omega_Mess, m*omega_Mess+b)
# plt.show()


def load_data(path):
    w, I, non = np.loadtxt(rel_path + path, skiprows=22, unpack=True)
    return w, I

T_ref_back, I_ref_back = load_data("REFHINT.STN")
T_ref_front, I_ref_front = load_data("REFVORN.STN")
T_met_back, I_met_back = load_data("METHINT.STN")
T_met_front, I_met_front = load_data("METVORN.STN")

T_ref_ges = np.concatenate((T_ref_front[:np.argmin(abs(T_ref_front-22.5))+1], T_ref_back))
T_ref_corr = correct_Theta(theta=T_ref_ges, m=m, b=b)
lambda_ref = calc_lambda(T_ref_corr, 1)
I_ref_ges = np.concatenate((I_ref_front[:np.argmin(abs(T_ref_front-22.5))+1], I_ref_back*1.5/5))
I_ref_filt = savgol_filter(I_ref_ges, 15, 3)
T_met_ges = np.concatenate((T_met_front[:np.argmin(abs(T_met_front-22.5))+1], T_met_back))
T_met_corr = correct_Theta(theta=T_met_ges, m=m, b=b)
lambda_met = calc_lambda(T_met_corr, 1)
I_met_ges = np.concatenate((I_met_front[:np.argmin(abs(T_met_front-22.5))+1], I_met_back*1.5/5))
I_met_filt = savgol_filter(I_met_ges, 15, 3)

ic( (lambda_ref == lambda_met).all() )
mu = calc_mu(I_met_filt, I_ref_filt, lambda_ref)

# Plot
# plt.plot(T_ref_ges, I_ref_ges, T_ref_ges, I_ref_filt)
# plt.plot(T_met_ges, I_met_ges, T_met_ges, I_met_filt)
# plt.show()

plt.plot(lambda_ref, I_ref_ges)
plt.plot(lambda_met, I_met_ges)
plt.show()

# plt.plot(lambda_ref, mu)
# plt.show()