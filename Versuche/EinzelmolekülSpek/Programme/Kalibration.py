import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter
from icecream import ic
from cycler import cycler
import locale
import matplotlib.image as mpimg

locale.setlocale(locale.LC_NUMERIC, "de_DE")
c_cycle = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])
mpl.rcParams.update({'font.size': 20, 'axes.labelpad': 0, 'axes.linewidth': 1.2, 'axes.titlepad': 8.0,
                    'legend.framealpha': 0.4, 'legend.borderaxespad': 0.3, 'legend.borderpad': 0.2,
                    'legend.labelspacing': 0, 'legend.handletextpad': 0.2, 'legend.handlelength': 1.0,
                    'legend.loc': 'best', 'xtick.labelsize': 'small', 'xtick.major.pad': 2, 
                    'xtick.major.size': 3, 'xtick.major.width': 1.2, 'ytick.labelsize': 'small',
                    'ytick.major.pad': 2, 'ytick.major.size': 3, 'ytick.major.width': 1.2,
                     'axes.prop_cycle': cycler(color=c_cycle), 'axes.formatter.use_locale': True})

path = "C:/Users/toni-/OneDrive/Alt/Desktop/SMS/Kalibration/Kalibration-Breit.csv"
px, I = np.loadtxt(path, skiprows=1, unpack=True, delimiter=",")
pixels = np.array([234, 242, 302, 306, 369])
pInt = np.array([240.82, 461.64, 46.59, 48.20, 395.07])
lams = np.array([542.5, 546.2, 577, 579.1, 611])
stringis = [r"$\lambda=542,5\,\mathrm{nm}$", r"$\lambda=546,2\,\mathrm{nm}$", 
            r"$\lambda=577,0\,\mathrm{nm}$", r"$\lambda=579,1\,\mathrm{nm}$",
            r"$\lambda=611,0\,\mathrm{nm}$"]


vals, cov = np.polyfit(x=pixels, y=lams, deg=1, cov=True)
m, b  = vals[0], vals[1]
sm, sb = np.sqrt(np.diag(cov))
ic( m, sm, b, sb)

# Plot
# plotx = np.linspace(min(pixels)-4, max(pixels)+4, 100)
# plt.plot(pixels, lams, ls="none", marker="+", markersize=12, color="red", label=r"$Messpunkte$") 
# plt.plot(plotx, m*plotx+b, color="cadetblue", zorder=-1, lw=1.2, label="Fit-Gerade")
# plt.xlim(min(plotx), max(plotx))
# plt.ylim(539, 613)
# plt.xlabel("Pixel-Nummer")
# plt.ylabel(r"$\lambda\,/\,$nm")
# plt.legend(loc="upper left", facecolor="wheat", framealpha=0.5, fontsize=18)
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Bilder/Fit.pdf", bbox_inches="tight")

def PixToLam(pix):
    return m*pix+b
def Inverse(lam):
    return (lam-b)/m

img = mpimg.imread('C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Bilder/KalibImg.png')

fig, (ax2, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 5]}, figsize=(15, 5))
fig.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.99, wspace=0, hspace=-0.08)
ax2.imshow(img[250:270, :, :])
ax2.yaxis.set_ticks([], [])

lam_plot = PixToLam(px)
pixels = PixToLam(pixels)
ax1.plot(lam_plot, I, lw=1.3, color="darkcyan")
ax1.set_xlim(min(lam_plot), max(lam_plot))
ax1.set_ylim(0, max(I)+50)
ax1.set_xlabel(r"$\lambda\,/\,$nm")
ax1.set_ylabel("Intensität / a.u.")
ax2.xaxis.tick_top()
ax2.set_xlabel("Pixel-Nummer")
ax2.xaxis.set_label_position('top') 
#secax = ax1.secondary_xaxis('top', functions=(PixToLam, Inverse))
#secax.xaxis.set_ticks([], [])
#secax.set_xlabel(r"$\lambda\,/\,$nm", labelpad=10)
space = np.array([25,25,20,40,25])
ax1.vlines(x=pixels, ymin=pInt+3, ymax=pInt+space, color="k", lw=1, alpha=0.8)
for i, w in enumerate(pixels):
    ax1.text(x=w, y=pInt[i]+space[i]-1, s=str(i+1), horizontalalignment='center',
    verticalalignment='bottom', fontsize=15)
    ax1.plot([], [], ls="none", label=str(i+1) + ":  " + stringis[i])
ax1.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=18)
ax2.text(x=0.01, y=0.5, s="A", color="white", fontsize=25, horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
ax1.text(x=0.01, y=0.97, s="B", color="k", fontsize=25, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
plt.show()
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Bilder/KalibNeu.pdf", 
#             bbox_inches="tight")