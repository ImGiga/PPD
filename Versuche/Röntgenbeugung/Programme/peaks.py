import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
# Einlesen der Daten
theta, intens = np.loadtxt('/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Röntgenbeugung/Daten/Messung 2/G1NACL.DAT', unpack=True)

plt.plot(theta, intens, label='Messwerte', c='b')
# intens = savgol_filter(intens, 15, 3)
# plt.plot(theta, intens, label='Gefilterte Messwerte', c='r')
plt.xlabel(r'$2\theta$ in °')
plt.ylabel('Intensität (Zählrate)')
plt.xlim(6,118)


wavelen =  1.540593

theta_new = np.array([27.46, 31.85, 45.78, 54.18, 56.85, 66.66, 73.75, 75.743, 84.58,
                  89.52, 101.86, 110.80])
theta_half = theta_new / 2

# hkl-Werte
hkl = np.array(['(1 1 1)', '(2 0 0)', '(2 2 0)', '(1 1 3)', '(2 2 2)', '(0 0 4)',
                '(3 1 3)', '(2 2 4)', '(3 3 3)', '(4 0 4)', '(3 1 5)', '(4 2 4)'])


hkl_val = np.array([3, 4, 8, 12, 16, 20, 24, 32, 36])

offset = max(intens) * 0.05  # Adjust this value as needed

# for th, hkl_str in zip(theta_new, hkl):
#     plt.text(th, intens[np.argmin(np.abs(theta - th))] + offset, hkl_str, ha='center', va='bottom')

plt.legend()
plt.savefig('spektrum_roh.png', dpi=300)
plt.show()

# Berechnung der Gitterkonstanten
# a = (wavelen * np.sqrt(hkl_val)) / (2 * np.sin(np.radians(theta_half)))

# # Berechnung der Abstände
# d = a / np.sqrt(hkl_val)

# # Berechnung sin^2
# sin = (np.sin(np.radians(theta_half)))**2

# # Berechnung Konstante
# konst = wavelen ** 2  /(4 * a**2)

# df = pd.DataFrame({'hkl': hkl_val, 'd': d, 'sin^2': sin, 'Konstante': konst, 'a': a})

# # print(df)
# a = np.mean(a)
# theta_2 = np.array([54.18, 73.75, 89.52, 108.51])
# theta_2_half = theta_2 / 2

# hkl_val_2 = np.array([11, 19, 27, 35])

# d = a / np.sqrt(hkl_val_2)

# sin = (np.sin(np.radians(theta_2_half)))**2

# df = pd.DataFrame({'hkl': hkl_val_2, 'd': d, 'sin^2': sin, 'theta': theta_2})
# # print(df)
