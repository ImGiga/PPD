import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib as mpl
from cycler import cycler
from matplotlib.ticker import MultipleLocator
from scipy.signal import windows, savgol_filter
from scipy.optimize import minimize, curve_fit
from scipy.fftpack import rfft, irfft, rfftfreq, fft, fftfreq, ifft
from scipy.stats import linregress

def fit_Gauss(x, a, b, c):
    return a*np.exp( -0.5*(x-b)**2/(c**2))

c_cycle = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])
mpl.rcParams.update({'font.size': 18, 'axes.labelpad': 0, 'axes.linewidth': 1.2, 'axes.titlepad': 8.0,
                    'legend.framealpha': 0.4, 'legend.borderaxespad': 0.3, 'legend.borderpad': 0.2,
                    'legend.labelspacing': 0, 'legend.handletextpad': 0.2, 'legend.handlelength': 1.0,
                    'legend.loc': 'best', 'xtick.labelsize': 'small', 'xtick.major.pad': 2, 
                    'xtick.major.size': 3, 'xtick.major.width': 1.2, 'ytick.labelsize': 'small',
                    'ytick.major.pad': 2, 'ytick.major.size': 3, 'ytick.major.width': 1.2,
                     'axes.prop_cycle': cycler(color=c_cycle)})

path = os.path.dirname(__file__) + "/../Daten/"
x, Ilamp0 = np.loadtxt(path + "Lampe_Signal_0_at_679.99nm_cut_at_1286.00Y_01.dat", unpack=True, skiprows=6)
Ilamp90 = np.loadtxt(path + "Lampe_Signal_90_at_679.99nm_cut_at_1286.00Y_01.dat", unpack=False, skiprows=6)[:,1]
IlampUN= np.loadtxt(path + "Lampe_Signal_unpol_at_679.99nm_cut_at_1286.00Y_01.dat", unpack=False, skiprows=6)[:,1]
Idark = np.loadtxt(path + "Rausch_Signal_at_679.99nm_cut_at_1286.00Y_01.dat", unpack=False, skiprows=6)[:,1]
for file in os.listdir(path):
    if file[:6].count("-") == 2:
        a = 5
    else:
        a = 6

    if (file.count("-0_at") == 1) and (file[:2] == "90"):
        vars()["I_0_90_"+file[3:a]] = np.loadtxt(path+file, unpack=False, skiprows=6)[:,1]
    elif (file.count("-90_at") == 1) and (file[:2] == "90"):
        vars()["I_90_90_"+file[3:a]] = np.loadtxt(path+file, unpack=False, skiprows=6)[:,1]
    elif (file.count("-unpol_at") == 1) and (file[:2] == "90"):
        vars()["I_UN_90_"+file[3:a]] = np.loadtxt(path+file, unpack=False, skiprows=6)[:,1]
    if (file.count("-0_at") == 1) and (file[:2] == "70"):
        vars()["I_0_70_"+file[3:a]] = np.loadtxt(path+file, unpack=False, skiprows=6)[:,1]
    elif (file.count("-90_at") == 1) and (file[:2] == "70"):
        vars()["I_90_70_"+file[3:a]] = np.loadtxt(path+file, unpack=False, skiprows=6)[:,1]
    elif (file.count("-unpol_at") == 1) and (file[:2] == "70"):
        vars()["I_UN_70_"+file[3:a]] = np.loadtxt(path+file, unpack=False, skiprows=6)[:,1]

add = 0.6
pol = ["0", "90", "UN"][0]
leng = ["70", "90"][0] 

ber = 50
for n, leng in enumerate(["70", "90"]):
    plt_name = "I_" + pol + "_" + leng + "_"
    lambdas = []
    errs = []
    for j, i in enumerate(["70", "80", "90", "100", "110", "120", "130", "140"]):
        Imess = eval(plt_name+i)
        Ilampe = eval("Ilamp"+pol)
        pltI_fil = (savgol_filter(Imess,500,3)-savgol_filter(Idark,500,3)+add)/(savgol_filter(Ilampe,500,3)-savgol_filter(Idark,500,3)+add)
        pltI = (Imess-Idark+add)/(Ilampe-Idark+add)
        max_I = np.argmax(pltI_fil)
        fitx = x[max_I-ber:max_I+ber]
        fitI = pltI_fil[max_I-ber:max_I+ber]
        popt, pcov = curve_fit(fit_Gauss, fitx, fitI, [10, x[max_I], 5])
        lambdas.append(popt[1])
        errs.append(np.sqrt(np.diag(pcov))[1])
            
    chos = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    L_0 = np.arange(70, 150, 10)
    Lx = np.linspace(65, 145, 100)
    res = linregress(L_0[chos], np.asarray(lambdas)[chos])
    print(res.slope/2, res.intercept/2)
    print(res.stderr/2, res.intercept_stderr/2)
    print(lambdas)
    print(np.asarray(errs)*200)
    plt.plot(L_0, lambdas, ls="none", marker=["D", "o"][n], c=["steelblue", "darkkhaki"][n], markersize=[6, 6][n])
    plt.plot(Lx, res.slope*Lx + res.intercept, c=["darkcyan", "olive"][n], zorder=-1)
    plt.plot([140], lambdas[-1], marker="x", c="red", zorder=10, markersize=8)
    plt.errorbar(x=L_0, y=lambdas, xerr=None, yerr=np.asarray(errs)*200, fmt="none", capsize=5, color='k', alpha=0.5, 
                 zorder=-10, lw=1.1)
plt.xlabel(r"$L\,/$ nm")
plt.ylabel(r"$\lambda\,/$ nm")
plt.gca().yaxis.set_major_locator(MultipleLocator(50))
plt.gca().xaxis.set_major_locator(MultipleLocator(20))
plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
plt.xlim(68, 142)
plt.ylim(610, 870)
plt.plot([],[], label=r"Measurement for $W=70\,\mathrm{nm}$", c="steelblue", ls="none", marker="D")
plt.plot([],[], label=r"Fit for $W=70\,\mathrm{nm}$", c="darkcyan")
plt.plot([],[], label=r"Measurement for $W=90\,\mathrm{nm}$", c="darkkhaki", ls="none", marker="o")
plt.plot([],[], label=r"Fit for $W=90\,\mathrm{nm}$", c="olive")
plt.errorbar([200], [10], xerr=None, yerr=[10], fmt="none", capsize=5, color='k', alpha=0.5, 
                 zorder=-10, lw=1.1, label="Error bars")
plt.plot([], [], marker="x", c="red", ls="none", label="Excluded values")

plt.legend(loc="upper left", facecolor="wheat", framealpha=0.5, fontsize=17)
plt.show()


# zeilen = 3
# spalten = 2
# faktor = 4
# figsize = (spalten*faktor, zeilen*faktor)
# fig = plt.figure(figsize=figsize)
# #fig.subplots_adjust(left=0.07, bottom=0.1, right=0.97, top=0.92, wspace=0.35, hspace=0.35)
# ax1 = fig.add_subplot(zeilen, spalten, 1)
# ax2 = fig.add_subplot(zeilen, spalten, 2, sharey=ax1)
# ax3 = fig.add_subplot(zeilen, spalten, 3)
# ax4 = fig.add_subplot(zeilen, spalten, 4, sharey=ax3)
# ax5 = fig.add_subplot(zeilen, spalten, 5)
# ax6 = fig.add_subplot(zeilen, spalten, 6, sharey=ax5)

# eval("ax1").set_xticklabels([])
# eval("ax2").set_xticklabels([])
# eval("ax3").set_xticklabels([])
# eval("ax4").set_xticklabels([])
# plt.setp(ax2.get_yticklabels(), visible=False)
# plt.setp(ax4.get_yticklabels(), visible=False)
# plt.setp(ax6.get_yticklabels(), visible=False)
# ax1.yaxis.set_major_locator(MultipleLocator(0.2))
# ax3.yaxis.set_major_locator(MultipleLocator(0.2))
# ax5.yaxis.set_major_locator(MultipleLocator(0.2))
# ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax5.yaxis.set_minor_locator(MultipleLocator(0.1))
# # ax1.set_ylim(0.06, 1.1)
# # ax3.set_ylim(0.06, 0.53)
# # ax5.set_ylim(0.06, 0.735)
# ax1.set_ylabel(r"$I_{\mathrm{C}}\,/\,I_{\mathrm{C, max}}$")
# ax3.set_ylabel(r"$I_{\mathrm{C}}\,/\,I_{\mathrm{C, max}}$")
# ax5.set_ylabel(r"$I_{\mathrm{C}}\,/\,I_{\mathrm{C, max}}$")
# ax5.set_xlabel(r"$\lambda\,/$ nm")
# ax6.set_xlabel(r"$\lambda\,/$ nm")

# fig.subplots_adjust(wspace=0, hspace=0)

# add = Idark
# ax_pol = ["0", "0", "90", "90", "UN", "UN"]
# ax_leng = ["70", "90", "70", "90", "70", "90"]
# Imax = 8.8
# #np.max( ((savgol_filter(eval("I_0_90_110"),500,3) - savgol_filter(Idark,500,3) + add) /(savgol_filter(eval("Ilamp0"),500,3)-savgol_filter(Idark,500,3)+add)) )
# for a in np.arange(1, 7):
#     ax = eval("ax"+str(a))
#     ax.set_xlim(min(x), max(x))
#     plt_name = "I_" + ax_pol[a-1] + "_" + ax_leng[a-1] + "_"
#     for j, i in enumerate(["70", "80", "90", "100", "110", "120", "130", "140"]):
#         Imess = eval(plt_name+i)
#         Ilampe = eval("Ilamp"+ax_pol[a-1])
#         pltI_fil = (savgol_filter(Imess,500,3)-savgol_filter(Idark,500,3)+add)/(savgol_filter(Ilampe,500,3)-savgol_filter(Idark,500,3)+add)
#         pltI = (Imess-Idark+add)/(Ilampe-Idark+add)
#         ax.plot(x, pltI/Imax, c=c_cycle[j], alpha=0.5, lw=0.5)
#         ax.plot(x, pltI_fil/Imax, c=c_cycle[j], label=r"$L=$"+i+r"$\,\mathrm{nm}$")

# ax1.legend(loc="upper left", facecolor="wheat", framealpha=0.5, fontsize=14)
# ax1.text(x=0.87, y=0.83, s='W=70nm\npol=0°', color='black', fontsize=10, 
#         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.1'), transform=ax1.transAxes)
# ax2.text(x=0.87, y=0.83, s='W=90nm\npol=0°', color='black', fontsize=10, 
#         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.1'), transform=ax2.transAxes)
# ax3.text(x=0.87, y=0.83, s='W=70nm\npol=90°', color='black', fontsize=10, 
#         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.1'), transform=ax3.transAxes)
# ax4.text(x=0.87, y=0.83, s='W=90nm\npol=90°', color='black', fontsize=10, 
#         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.1'), transform=ax4.transAxes)
# ax5.text(x=0.87, y=0.83, s='W=70nm\nunpol', color='black', fontsize=10, 
#         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.1'), transform=ax5.transAxes)
# ax6.text(x=0.87, y=0.83, s='W=90nm\nunpol', color='black', fontsize=10, 
#         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.1'), transform=ax6.transAxes)
# plt.show()

# plt.plot(x, Ilamp0/np.max(IlampUN), label=r"$I_{\mathrm{L},\,0°}$", c="darkmagenta", lw=1)
# plt.plot(x, Ilamp90/np.max(IlampUN), label=r"$I_{\mathrm{L},\,90°}$", c="darkturquoise", lw=1)
# plt.plot(x, IlampUN/np.max(IlampUN), label=r"$I_{\mathrm{L,\,unpol}}$", c="forestgreen", lw=1)
# plt.plot(x, (Idark-add)/np.max(IlampUN), label=r"$I_{\mathrm{D}}$", c="darkslategrey", lw=1)
# plt.xlabel(r"$\lambda\,/$ nm")
# plt.ylabel(r"$I\,/\,I_{\mathrm{max}}$")
# plt.xlim(min(x)+10, max(x))
# plt.ylim(0.08, 1.02)
# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=20)
# plt.show()