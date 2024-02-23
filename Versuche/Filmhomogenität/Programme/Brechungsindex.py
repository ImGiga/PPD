import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from cycler import cycler
import locale
from matplotlib.ticker import MultipleLocator

locale.setlocale(locale.LC_NUMERIC, "de_DE")
c_cycle = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])
mpl.rcParams.update({'font.size': 22, 'axes.labelpad': 0, 'axes.linewidth': 1.2, 'axes.titlepad': 8.0,
                    'axes.formatter.use_locale': True, 'legend.framealpha': 0.4, 'legend.borderaxespad': 0.3, 
                    'legend.borderpad': 0.2,
                    'legend.labelspacing': 0, 'legend.handletextpad': 0.2, 'legend.handlelength': 1.0,
                    'legend.loc': 'best', 'xtick.labelsize': 'small', 'xtick.major.pad': 2, 
                    'xtick.major.size': 10, 'xtick.major.width': 2, 'ytick.labelsize': 'small',
                    'ytick.major.pad': 5, 'ytick.major.size': 8, 'ytick.major.width': 1.5,
                    'axes.prop_cycle': cycler(color=c_cycle),
                    'xtick.direction' : 'inout', 'ytick.direction' : 'inout', 
                    'text.usetex': True, 'pgf.texsystem': 'pdflatex', 'font.family': 'sans-serif',
                    'font.sans-serif': ['Helvetica'], 
                    'axes.axisbelow': False })

def Real(f, f0, d):
    return 1 + (f0**2 - f**2) / ( (f0**2 - f**2)**2 + (2*d*f)**2 )

def Imag(f, f0, d):
    return 2*d*f / ( (f0**2 - f**2)**2 + (2*d*f)**2 )

f = np.linspace(0, 10, 1000)
f0 = 5
d = 0.6

RealTeil = Real(f, f0, d)-1
ImagTeil = Imag(f, f0, d)
ArgMaxReal = np.argmax(RealTeil) 
ArgMinReal = np.argmin(RealTeil) 
ax = plt.gca()

plt.plot(f, RealTeil, c="g", label=r"$n - 1$", zorder=0)
plt.plot(f, ImagTeil, c="b", label=r"$\kappa$", zorder=0)
plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=22)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.xticks([f0], [r"$\omega_{0}$"])
plt.yticks([0], [r"$0$"])
plt.xlim(-0.01, 10.5)
plt.ylim(min(RealTeil)*1.1, max(ImagTeil)*1.05) 
plt.text(-0.05, 0.9, s=r"$n(\omega)$"+"\n"+r"$\kappa(\omega)$", horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)
plt.xlabel(r"$\omega$", loc="right", labelpad=-20)


ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)
plt.show()