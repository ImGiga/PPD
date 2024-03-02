import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Einlesen der Daten
theta, intens = np.loadtxt('/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Röntgenbeugung/Daten/Messung 2/G1NACL.DAT', unpack=True)

# plt.plot(theta, intens)
# plt.show()

wavelen =  1.540593

theta = np.array([27.5183, 31.8805, 45.7090, 56.8075, 66.6339, 75.7735, 84.5540,
                  101.9343, 110.9553])
theta_half = theta / 2

hkl_val = np.array([3, 4, 8, 12, 16, 20, 24, 32, 36])

# Berechnung der Gitterkonstanten
a = (wavelen * np.sqrt(hkl_val)) / (2 * np.sin(np.radians(theta_half)))

# Berechnung der Abstände
d = a / np.sqrt(hkl_val)

# Berechnung sin^2
sin = (np.sin(np.radians(theta_half)))**2

# Berechnung Konstante
konst = wavelen ** 2  /(4 * a**2)

df = pd.DataFrame({'hkl': hkl_val, 'd': d, 'sin^2': sin, 'Konstante': konst, 'a': a})

print(df)
a = np.mean(a)
theta_2 = np.array([54.18, 73.75, 89.52, 108.51])
theta_2_half = theta_2 / 2

hkl_val_2 = np.array([11, 19, 27, 35])

d = a / np.sqrt(hkl_val_2)

sin = (np.sin(np.radians(theta_2_half)))**2

df = pd.DataFrame({'hkl': hkl_val_2, 'd': d, 'sin^2': sin, 'theta': theta_2})
print(df)
