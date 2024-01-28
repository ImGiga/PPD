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


### A2
# REFDATA = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/REF.dat")
# HRDATA = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/HR_SIL.dat")
# tRef = REFDATA[:,0]
# tHr = HRDATA[:,0]
# Ref = REFDATA[:,1]
# Hr = HRDATA[:,1]

# plt.plot(tRef, Ref, c="turquoise", lw=1.2, label="Referenz")
# plt.plot(tHr, Hr, c="darkblue", lw=1.2, label="HR-Silizium")
# plt.xlim(0, max(tRef))
# plt.ylim(np.min(Ref)-0.01, np.max(Ref)+0.01)

# plt.vlines(4.8867, ymax=Hr[np.argwhere(np.isclose(tHr, 4.8867))] ,ymin=-3, lw=1, ls=":", color="k", alpha=0.7, zorder=-1)
# plt.vlines(11.0667, ymax=Hr[np.argwhere(np.isclose(tHr, 11.0667))] ,ymin=-3, lw=1, ls=":", color="k", alpha=0.7, zorder=-1)
# plt.vlines(2.653, ymax=Ref[np.argmin(abs(tRef-2.653))] ,ymin=-3, lw=1, ls=":", color="k", alpha=0.7, zorder=-1)

# plt.xlabel(r"$\Delta\tau /$ ps")
# plt.ylabel(r"Signal-Amplitude / AU")
# plt.plot([],[], lw=1, ls=":", color="k", alpha=0.7, label=r"$t_{1}, t_{2}, t_{3}$")

# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=15)
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Bilder/HRPuls.pdf", bbox_inches="tight")
# plt.show()



# Theo_HR = 4*3.4175/((1 + 3.4175)**2)
# data = np.load("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/HRtrans.npy")
# fplot = data[:,0]
# trans = data[:,1]

# plt.plot(fplot, trans, c="orangered", label=r"$T(\omega)_{\mathrm{HR-Si}}$")
# plt.xlim(0.4, 2.25)
# plt.ylim(0.49, 1.01)
# plt.hlines(y=Theo_HR, xmin=0.4, xmax=2.25, color="cornflowerblue", zorder=-1, label=r"$T(\omega)_{\mathrm{theo}}$")
# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=15)
# plt.xlabel(r"$\omega\,/\,$THz")
# plt.ylabel(r"Transmission")
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Bilder/HRTrans.pdf", bbox_inches="tight")
# plt.show()


# REFDATA = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/REF.dat")
# pDATA = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/pSil.dat")
# tRef = REFDATA[:,0]
# tp = pDATA[:,0]
# Ref = REFDATA[:,1]
# p = pDATA[:,1]

# plt.plot(tRef, Ref, c="turquoise", lw=1.2, label="Referenz")
# plt.plot(tp, p, c="darkblue", lw=1.2, label="p-dotiertes Si")
# plt.xlim(0, max(tRef))
# plt.ylim(np.min(Ref)-0.01, np.max(Ref)+0.01)
# plt.xlim(0, 5)

# plt.xlabel(r"$\Delta\tau /$ ps")
# plt.ylabel(r"Signal-Amplitude / AU")

# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=15)
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Bilder/pDotPuls.pdf", bbox_inches="tight")
# plt.show()


# d = np.load("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/pSi.npy")
# f = d[:,0]
# trans = d[:,1]
# fit = d[:,2]

# plt.plot(f, trans, c="orangered", label=r"$T(\omega)_{\mathrm{p-Si}}$")
# plt.plot(f, fit, color="cornflowerblue", zorder=-1, label=r"$T(\omega)_{\mathrm{Drude-Fit}}$")

# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=15)
# plt.ylim(0.39, 1.01)
# plt.xlim(0.4, 2.25)
# plt.xlabel(r"$\omega\,/\,$THz")
# plt.ylabel(r"Transmission")
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Bilder/pDotTrans.pdf", bbox_inches="tight")
# plt.show()


# t = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/Lack_REF.dat")[40:,0]
# ref = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/Lack_REF.dat")[40:,1]
# rot = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/ROT.dat")[40:,1]
# silber = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/SILBER.dat")[38:-2,1]*0.9-0.09
# mag_dunn = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/MAG_dunn.dat")[40:,1]
# mag_dick = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/MAG_dick.dat")[40:,1]

# plt.plot(t, ref, label="Referenz", c="k")
# plt.plot(t, rot, label="Rot", c="r")
# plt.plot(t+0.0156, silber, label="Silber", c="grey")
# plt.plot(t, mag_dunn, label="Magnetisch dünn", c="lightblue")
# plt.plot(t, mag_dick, label="Magnetisch dick", c="darkblue")
# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=15)
# plt.xlim(1.36, max(t))
# plt.ylim(-1.12, -0.73)
# print(np.argmax(ref), np.max(ref))
# plt.vlines(x=t[np.argmax(ref)], ymin=-1.2, ymax=np.max(ref), color="k", alpha=0.6, lw=1, ls="dashed", zorder=-1)
# plt.vlines(x=t[np.argmax(rot)], ymin=-1.2, ymax=np.max(rot), color="r", alpha=0.6, lw=1, ls="dashed", zorder=-1)
# plt.vlines(x=t[np.argmax(silber)]+0.0156, ymin=-1.2, ymax=np.max(silber), color="grey", alpha=0.6, lw=1, ls="dashed", zorder=-1)
# plt.vlines(x=t[np.argmax(mag_dunn)], ymin=-1.2, ymax=np.max(mag_dunn), color="lightblue", alpha=0.6, lw=1, ls="dashed", zorder=-1)
# plt.vlines(x=t[np.argmax(mag_dick)], ymin=-1.2, ymax=np.max(mag_dick), color="darkblue", alpha=0.6, lw=1, ls="dashed", zorder=-1)
# plt.xlabel(r"$\Delta\tau /$ ps")
# plt.ylabel(r"Signal-Amplitude / AU")
# plt.show()