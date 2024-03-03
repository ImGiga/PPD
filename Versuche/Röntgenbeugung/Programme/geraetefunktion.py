import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def function(x, U, V, W):
    return np.sqrt(U * np.tan(np.radians(x))**2 + V * np.tan(np.radians(x)) + W)

two_theta = np.array([27.518, 31.881, 45.709, 56.808, 66.634, 75.774, 84.554, 101.934, 110.955])
fwhm = np.array([0.308, 0.311, 0.324, 0.338, 0.355, 0.375, 0.399, 0.467, 0.517])
theta = two_theta / 2

x_range = np.linspace(5, 60)

popt, pcov = curve_fit(function, theta, fwhm)

plt.plot(theta, fwhm, 'o', label='Messwerte', c='r')
plt.plot(x_range, function(x_range, *popt), label='Fit', c='b')
plt.xlabel(r'Winkel $\theta$ in °')
plt.ylabel(r'FWHM')
plt.legend()
plt.show()
# plt.savefig('fwhm_fit.png', dpi=300)
print(popt*1e3)
print(np.sqrt(np.diag(pcov))*1e3)