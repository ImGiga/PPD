import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.signal import savgol_filter
from icecream import ic
from cycler import cycler
import locale
import matplotlib.image as mpimg
from scipy.optimize import curve_fit

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

def fit_cos(x, a, f, c, p):
    return a*np.cos(2*np.pi*(f*x - p)) + c

path = "C:/Users/toni-/OneDrive/Alt/Desktop/"

t1, I1 = np.loadtxt(path+"Pol1.csv", skiprows=1, unpack=True, delimiter=",")
t2, I2 = np.loadtxt(path+"Pol2.csv", skiprows=1, unpack=True, delimiter=",")
t3, I3 = np.loadtxt(path+"Pol3.csv", skiprows=1, unpack=True, delimiter=",")

x_plot = np.linspace(0, 80, 1000)
popt1, pcov1 = curve_fit(fit_cos, t1/2, I1, [160, 0.3, 200, 0.1], bounds=([0, 0.2, 50, -1] , [300, 0.4, 400, 1]))
popt2, pcov2 = curve_fit(fit_cos, t2/2, I2, [160, 0.3, 200, 0.1], bounds=([0, 0.2, 50, -1] , [300, 0.4, 400, 1]))
popt3, pcov3 = curve_fit(fit_cos, t3/2, I3, [160, 0.3, 200, 0.1], bounds=([0, 0.2, 50, -1] , [300, 0.4, 400, 1]))
ic(popt1, np.sqrt(np.diag(pcov1)))
ic(popt2, np.sqrt(np.diag(pcov2)))
ic(popt3, np.sqrt(np.diag(pcov3)))
err = 0
for i in np.arange(1, 4):
    print(f"a = ({eval('popt'+str(i))[0]} \pm {np.sqrt(np.diag( eval( 'pcov'+str(i) ) ))[0]})")
    print(f"f = ({eval('popt'+str(i))[1]} \pm {np.sqrt(np.diag( eval( 'pcov'+str(i) ) ))[1]})")
    print(f"\delta = ({eval('popt'+str(i))[2]} \pm {np.sqrt(np.diag( eval( 'pcov'+str(i) ) ))[2]})")
    print(f"b = ({eval('popt'+str(i))[3]} \pm {np.sqrt(np.diag( eval( 'pcov'+str(i) ) ))[3]})")
    print("")
    err += np.sqrt(np.diag( eval( 'pcov'+str(i) ) ))[1]**2
print(np.sqrt(err)/3)


bx = 104
img1 = mpimg.imread(path+"Pol1.png")[bx-30:2*bx-30,:,:]
img2 = mpimg.imread(path+"Pol2.png")[bx:2*bx,:,:]
img3 = mpimg.imread(path+"Pol3.png")[2*bx-50:3*bx-50,:,:]
ic(img1.shape)

def MyTicks(x, pos):
    return str(int(x/2))

fig = plt.figure(figsize=(15, 4))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 4], height_ratios=[1,1,1])
ax1 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[1, 0])
ax13 = fig.add_subplot(gs[2, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])
fig.subplots_adjust(left=-0.02, bottom=0.15, right=0.94, top=0.97, wspace=-0.07, hspace=0.08)
ax1.imshow(img1)
ax12.imshow(img2)
ax13.imshow(img3)
ax2.plot(t1/2, I1, color="olivedrab")
ax3.plot(t2/2, I2, color="olivedrab")
ax4.plot(t3/2, I3, color="olivedrab")
ax2.plot(x_plot, fit_cos(x_plot, *popt1), color="orange", alpha=0.5)
ax3.plot(x_plot, fit_cos(x_plot, *popt2), color="orange", alpha=0.5)
ax4.plot(x_plot, fit_cos(x_plot, *popt3), color="orange", alpha=0.5)

for j, ax in enumerate([ax2, ax3, ax4]):
    ax.set_xlim(0, 74)
    ax.set_ylim(-5, max(eval("I"+str(j+1)))+30)
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right') 

ax3.set_ylabel("Intensität / a.u.", rotation=270, labelpad=20)
ax4.set_xlabel("Zeit / s")
ax13.set_xlabel("Zeit / s")
ax13.xaxis.set_major_formatter(FuncFormatter(MyTicks))
ax2.xaxis.set_ticklabels([])
ax3.xaxis.set_ticklabels([])
ax12.xaxis.set_ticklabels([])
ax1.xaxis.set_ticklabels([])
ax13.set_yticks([],[])
ax1.set_yticks([],[])
ax12.set_yticks([],[])
ax13.xaxis.set_major_locator(MultipleLocator(60))
ax13.xaxis.set_minor_locator(MultipleLocator(20))
ax12.xaxis.set_major_locator(MultipleLocator(60))
ax12.xaxis.set_minor_locator(MultipleLocator(20))
ax1.xaxis.set_major_locator(MultipleLocator(60))
ax1.xaxis.set_minor_locator(MultipleLocator(20))
ax1.text(x=0.8, y=0.8, s="A", color="white", fontsize=25, horizontalalignment='center', 
         verticalalignment='center', transform=ax1.transAxes)
ax12.text(x=0.8, y=0.8, s="B", color="white", fontsize=25, horizontalalignment='center', 
         verticalalignment='center', transform=ax12.transAxes)
ax13.text(x=0.8, y=0.8, s="C", color="white", fontsize=25, horizontalalignment='center', 
         verticalalignment='center', transform=ax13.transAxes)

plt.show()
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Bilder/Polarisation.pdf", 
#             bbox_inches="tight")