import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

h = 4.135136*1e-18    # keV s
c = 299792458   # m / s

d = 1.93146 # A

lambda_Theo = np.array([0.17892, 0.17892, 0.17942, 0.17960, 0.18310, 0.18327, 0.18438, 0.18518, 0.20901, 0.21383,
                       1.06146, 1.06786, 1.09857, 1.24430, 1.26271, 1.28181, 1.30164, 1.42112, 1.47631, 1.48745]) # A
Index_liste = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0], dtype=bool)

Theta_Theo = np.arcsin(lambda_Theo / (2*d))                 # rad
#omega_Theo = (Theta_Theo * 360 / (2*np.pi))[Index_liste]                      # degree

omega_Theo = np.arcsin(np.array([1.06786, 1.09857, 1.24430, 1.28181, 1.47631]) / (2.0*d)) *360 / (2.0*np.pi)
omega_Mess = np.array([15.919, 16.452, 18.730, 19.32, 22.415])

m, b = np.polyfit(x=omega_Mess, y=omega_Theo, deg=1)
print(m, b)

plt.plot(omega_Mess, omega_Theo, ls="none", marker="o", markersize=4) 
plt.plot(omega_Mess, m*omega_Mess+b)
plt.show()


w_messHINT, I_messHINT, non = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/Röntgenbeugung/Daten/Messung 1/REFHINT.STN", 
                  skiprows=22, unpack=True)
w_mess, I_mess, non = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/Röntgenbeugung/Daten/Messung 1/REFVORN.STN", 
                  skiprows=22, unpack=True)
w_m, I_m_vorn, non = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/Röntgenbeugung/Daten/Messung 1/METVORN.STN", 
                  skiprows=22, unpack=True)

w_gh, I_hg, non = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/Röntgenbeugung/Daten/Messung 1/Rest/HG10_231.STN", 
                  skiprows=22, unpack=True)

# plt.plot(w_mess, I_mess/max(I_mess), alpha=0.4, marker="o", markersize=1)
# plt.plot(w_mess, savgol_filter(I_mess/max(I_mess), 10, 3), c="green", zorder=10)
# plt.show()
# plt.plot(w_messHINT, I_messHINT/max(I_messHINT), alpha=0.4, ls="none", marker="o", markersize=1)
# plt.plot(w_messHINT, savgol_filter(I_messHINT/max(I_messHINT), 20, 3), c="green", zorder=10)
# plt.show()

# plt.plot(w_m, I_m_vorn/max(I_m_vorn), alpha=0.4, ls="none", marker="o", markersize=1)
# plt.plot(w_m, savgol_filter(I_m_vorn/max(I_m_vorn), 20, 3), c="green", zorder=10)
# plt.show()

def omega_correct(freq):
    return m*freq + b
omega_real = omega_correct(w_mess)
lambda_corr = 2*d*np.sin(np.radians(omega_real))

# plt.plot(lambda_corr, I_mess/max(I_mess))
# plt.show()
plt.plot(2*d*np.sin(np.radians(omega_correct(w_m))), I_m_vorn)
plt.show()

# plt.plot(w_gh, I_hg/max(I_hg), marker="o", markersize=1)
# plt.show()