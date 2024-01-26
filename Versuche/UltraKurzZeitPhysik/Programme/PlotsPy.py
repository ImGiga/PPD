import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib as mpl
from cycler import cycler
from matplotlib.ticker import MultipleLocator
from scipy.signal import windows
from scipy.optimize import minimize, curve_fit
from scipy.fftpack import rfft, irfft, rfftfreq, fft, fftfreq, ifft

c_cycle = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])
mpl.rcParams.update({'font.size': 18, 'axes.labelpad': 0, 'axes.linewidth': 1.2, 'axes.titlepad': 8.0,
                    'legend.framealpha': 0.4, 'legend.borderaxespad': 0.3, 'legend.borderpad': 0.2,
                    'legend.labelspacing': 0, 'legend.handletextpad': 0.2, 'legend.handlelength': 1.0,
                    'legend.loc': 'best', 'xtick.labelsize': 'small', 'xtick.major.pad': 2, 
                    'xtick.major.size': 3, 'xtick.major.width': 1.2, 'ytick.labelsize': 'small',
                    'ytick.major.pad': 2, 'ytick.major.size': 3, 'ytick.major.width': 1.2,
                     'axes.prop_cycle': cycler(color=c_cycle)})

data = np.load("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/auto.npy")
x = data[:,0]
AKF = data[:,1]/np.max(data[:,1])
filtered = data[:,2]/np.max(data[:,1])
data2 = np.load("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/FT_auto.npy")
f = data2[:,0]
FT = data2[:,1]/np.max(data2[:,1])
data3 = np.load("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/Gauss_auto.npy")
x = data3[:,0]
Fit = data3[:,1]/np.max(data[:,1])
Gauss = data3[:,2]/np.max(data[:,1])

# plt.plot(x, AKF, c="turquoise", linewidth=0.6, label="IAKF")
# plt.plot(x, filtered, c="darkblue", label="IAKF gefiltert") 
# plt.xlim(min(x), max(x))
# plt.ylim(-0.025, 1.025)

# plt.ylabel("normierte IAKF")
# plt.xlabel(r"$\Delta\tau\,/\,ps$")
# plt.gca().xaxis.set_major_locator(MultipleLocator(0.4))
# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=15)

# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Bilder/IAKF.pdf", bbox_inches="tight")
# plt.show()


# plt.plot(f, FT, lw=0.5, c="darkorange", label=r"$\mathcal{F}\,(IAKF)$")

# plt.xlim(-1, 380)
# plt.ylim(-0.025, 1.025)
# plt.vlines(10, -0.025, 1.025, color="orange", alpha=0.8, ls=":", label=r"$\omega_{\mathrm{cut}}=10\,$THz")
# plt.ylabel(r"$\mathcal{F}\,(IAKF)$ normiert")
# plt.xlabel(r"$\omega\,/\,$THz")

# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=15)

# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Bilder/FTIAKF.pdf", bbox_inches="tight")
# plt.show()


# plt.plot(x, Fit, c="darkblue", label="gefilterte IAKF")
# plt.plot(x, Gauss, c="springgreen", ls="--", label="Gaussian fit")
# plt.xlim(min(x), max(x))
# plt.ylim(0.14, 0.41)
# plt.gca().xaxis.set_major_locator(MultipleLocator(0.4))
# # plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=15)
# plt.ylabel("gefilterte IAKF")
# plt.xlabel(r"$\Delta\tau\,/\,ps$")
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Bilder/Gauss.pdf", bbox_inches="tight")
# plt.show()