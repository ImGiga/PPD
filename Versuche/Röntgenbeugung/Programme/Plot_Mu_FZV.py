import numpy as np 
import matplotlib.pyplot as plt

rho = 11.34
E = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.5, 2, 4, 6, 10, 15, 20, 30, 50, 100])
muM = np.array([130.6, 111.6, 86.36, 30.32, 14.36, 8.041, 5.020, 2.419, 5.55, 2.014, 0.999, 0.403, 0.232, 0.125, 0.0887, 0.071, 0.0522, 0.0461,
                0.0420, 0.0439, 0.0497, 0.0566, 0.0621, 0.07, 0.0807, 0.0937])
mu = rho*muM
h = 4.135669e-15 # eV
c = 299792458 # m / s
lamda = np.divide(1,E)*h*c*10e-6*10e10

figsize = (10, 4)
fig = plt.figure(figsize=figsize)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
fig.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.99, wspace=0.05, hspace=0)

ax1.loglog(E, mu, c="k")
ax2.loglog(lamda, mu, c="k")
ax2.set_yticklabels([])
ax1.set_xlim(np.min(E), np.max(E))
ax2.set_xlim(np.min(lamda), np.max(lamda))
ax1.set_ylabel(r"$\mu\,/\,cm^{2}g^{-1}$")
ax2.set_xlabel(r"$\lambda\,/\,\AA$")
ax1.set_xlabel(r"$E\,/\,MeV$")

plt.show()
#plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/Röntgenbeugung/Bilder/skizze.pdf", bbox_inches="tight")