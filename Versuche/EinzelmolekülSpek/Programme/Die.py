import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter
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

path = "C:/Users/toni-/OneDrive/Alt/Desktop/"

t1, I1 = np.loadtxt(path+"DieNeu1.csv", skiprows=1, unpack=True, delimiter=",")
t2, I2 = np.loadtxt(path+"DieNeu2.csv", skiprows=1, unpack=True, delimiter=",")
t3, I3 = np.loadtxt(path+"DieNeu3.csv", skiprows=1, unpack=True, delimiter=",")

bx = 104
img1 = mpimg.imread(path+"DieNeu1.png")[bx:2*bx,:,:]
img2 = mpimg.imread(path+"DieNeu2.png")[2*bx:3*bx,:,:]
img3 = mpimg.imread(path+"DieNeu3.png")[2*bx:3*bx,:,:]
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
fig.subplots_adjust(left=-0.02, bottom=0.15, right=0.94, top=0.97, wspace=-0.07, hspace=0.15)
ax1.imshow(img1)
ax12.imshow(img2)
ax13.imshow(img3)
ax2.plot(t1/2, I1, color="olivedrab")
ax3.plot(t2/2, I2, color="olivedrab")
ax4.plot(t3/2, I3, color="olivedrab")

for j, ax in enumerate([ax2, ax3, ax4]):
    ax.set_xlim(0, 74)
    ax.set_ylim(-20, max(eval("I"+str(j+1)))+30)
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

#plt.show()
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Bilder/Sterben.pdf", 
#             bbox_inches="tight")