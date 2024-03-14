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

path = "C:/Users/toni-/OneDrive/Alt/Desktop/SMS/Kalibration/Kalibration-Breit.csv"
px, I = np.loadtxt(path, skiprows=1, unpack=True, delimiter=",")
pixels = np.array([234, 242, 302, 306, 369])
lams = np.array([542.5, 546.2, 577, 579.1, 611])
vals, cov = np.polyfit(x=pixels, y=lams, deg=1, cov=True)
m, b  = vals[0], vals[1]
sm, sb = np.sqrt(np.diag(cov))

def PixToLam(pix):
    return m*pix+b

def fit_gaus(x,mu,sig):
    return np.exp(-(x-mu)**2/(2*sig**2))
def fit_multi_gaus(x, a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3):
    return a1*fit_gaus(x, mu1, sig1) + a2*fit_gaus(x, mu2, sig2) * a3*fit_gaus(x, mu3, sig3)

def voigt(x, eta, mu, sig, a, b):
    return a* (eta * (1 / ( 1 + ((x-mu) / sig)**2 ) ) + (1 - eta) * np.exp(-np.log(2)* ((x-mu) / sig)**2)) + b 
def multi_voigt(x, eta1, mu1, sig1, a1, b1, eta2, mu2, sig2, a2, b2, eta3, mu3, sig3, a3, b3):
    return voigt(x, eta1, mu1, sig1, a1, b1) + voigt(x, eta2, mu2, sig2, a2, b2) + voigt(x, eta3, mu3, sig3, a3, b3)

def Erg_Voigt(vals, errs, pars)-> None:
    # eta, mu, sig, a, b
    pl = ["eta", "mu", "sig", "a", "b"]
    print("Voigt")
    for j in range(3):
        print("Ergebnisse "+str(j+1)+":")
        for i in range(pars):
            print(pl[i]+f"\t=\t{vals[i+pars*j]:.4f}  +/-  {errs[i+pars*j]:.4f}")

def Erg_Gauss(vals, errs, pars)-> None:
    # a,mu,sig,b
    pl = ["a", "mu", "sig", "b"][:pars]
    print("Gauss")
    for j in range(3):
        print("Ergebnisse "+str(j+1)+":")
        for i in range(pars):
            print(pl[i]+f"\t=\t{vals[i+pars*j]:.4f}  +/-  {errs[i+pars*j]:.4f}")

#guess, lower, upper
g_eta1, l_eta1, u_eta1 = 0.5, 0.001, 0.999  
g_eta2, l_eta2, u_eta2 = 0.5, 0.001, 0.999 
g_eta3, l_eta3, u_eta3 = 0.5, 0.001, 0.999 

g_a1, l_a1, u_a1 = 1, 0.1, 1.5
g_b1, l_b1, u_b1 = 0.01, 0, 0.1
g_a2, l_a2, u_a2 = 0.4, 0.1, 1.5
g_b2, l_b2, u_b2 = 0.01, 0, 0.1
g_a3, l_a3, u_a3 = 0.05, 0.01, 0.2
g_b3, l_b3, u_b3 = 0.01, 0, 0.1

g_mu1, l_mu1, u_mu1 = 536, 500, 550  
g_sig1, l_sig1, u_sig1 = 10, 0, 1000  
g_mu2, l_mu2, u_mu2 = 578, 550, 600  
g_sig2, l_sig2, u_sig2 = 15, 0, 1000  
g_mu3, l_mu3, u_mu3 = 630, 610, 650  
g_sig3, l_sig3, u_sig3 = 15, 0, 1000  

lam_plot = np.linspace(300, 900, 3000)

lRef, IRef = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Literatur/Emmisionspektrum_PBI.txt", 
                        unpack=True)
poptRefV, pcovRefV = curve_fit(multi_voigt, lRef, IRef, p0=[g_eta1, g_mu1, g_sig1, g_a1, g_b1, g_eta2, g_mu2, g_sig2, g_a2, g_b2, g_eta3, g_mu3, g_sig3, g_a3, g_b3],
                               bounds=([l_eta1, l_mu1, l_sig1, l_a1, l_b1, l_eta2, l_mu2, l_sig2, l_a2, l_b2, l_eta3, l_mu3, l_sig3, l_a3, l_b3], 
                                       [u_eta1, u_mu1, u_sig1, u_a1, u_b1, u_eta2, u_mu2, u_sig2, u_a2, u_b2, u_eta3, u_mu3, u_sig3, u_a3, u_b3]))

data = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Programme/daten.txt", skiprows=1, unpack=True)

lam1 = np.mean(data[1,:])
lam2 = np.mean(data[2,:])
lam3 = np.mean(data[3,:])
slam1 = np.sqrt(np.sum(np.square(50*data[4,:]))) / 30
slam2 = np.sqrt(np.sum(np.square(15*data[5,:]))) / 30
slam3 = np.sqrt(np.sum(np.square(10*data[6,:]))) / 30

ic(lam1, slam1)
ic(lam2, slam2)
ic(lam3, slam3)
d1 = data[1,:]
d2 = data[2,:]
d3 = data[3,:]


print( ((1 / lam1) - (1 / lam2))*1e7 )
print( np.sqrt( ( slam1 / lam1 / lam1 )**2 + ( slam2 / lam2 / lam2 )**2 )*1e7)
print( ((1 / lam2) - (1 / lam3))*1e7 )
print( np.sqrt( ( slam3 / lam3 / lam3 )**2 + ( slam2 / lam2 / lam2 )**2 )*1e7)

for bx in [4]: 
    print(bx)
    # bx = 3
    hist, bin_edges = np.histogram(np.concatenate((d1,d2,d3)), bins=np.arange(520, 650, bx))
    plt.bar(bin_edges[:-1], hist, width = bx, edgecolor='black', facecolor="darkcyan", label="Anzahl", alpha=0.5)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.plot(lRef, IRef*np.max(hist), c="k", zorder=10, label="Ensemble-Spektrum", lw=2)
    plt.ylabel("Häufigkeit")
    plt.xlabel(r"$\lambda\,/\,$nm")
    plt.xlim(min(lRef), 680)
    plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=16)
    plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Bilder/histo.pdf", bbox_inches="tight")
    #plt.show()  

# plt.hist(data[1,:], bins=10)
# plt.hist(data[2,:], bins=10)
# plt.hist(data[3,:], bins=10)
# plt.show()
