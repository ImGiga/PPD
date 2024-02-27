import numpy as np 
import matplotlib.pyplot as plt

teta = np.array([-0.8, -0.825, -0.85])
y = np.array([7000, 4500, 2800])

teta = np.array([-0.3, -0.35, -0.4])
y = np.array([1000, 3650, 7300])

teta = np.array([0, 0.1, 0.2])
y = np.array([9200, 6500, 3800])

teta = np.array([-0.9, -0.8, -0.7])
y = np.array([3200, 6300, 9700])

m, b = np.polyfit(teta, y, 1)

max_y = 9700
max_y2 = 17700

# plt.plot(teta, y, 'yo', teta, m*teta+b, '--k')
# plt.hlines(y=max_y2/2, xmin=-0.9, xmax=-0.7)
# plt.show()

FWHM_vorne = -0.3637556
FWHM_hinten = -0.8240091

FWHM_omega1 = 0.0129626
FWHM_omega2 = -0.7246159
print(abs(FWHM_hinten - FWHM_vorne)*0.5 + FWHM_hinten)
print(abs(FWHM_omega1 - FWHM_omega2)*0.5 + FWHM_omega2)

print(-2.5+0.594)
