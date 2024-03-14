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

path = "C:/Users/toni-/OneDrive/Alt/Desktop/SMS/Spektren/"
lam_dict = {}
for i in np.arange(1, 31):
    vars()["t"+str(i)], vars()["I"+str(i)] = np.loadtxt(path+"S"+str(i)+".csv", skiprows=1, unpack=True, delimiter=",")
    vars()["img"+str(i)] = mpimg.imread(path+"S"+str(i)+".png")
    vars()["lam"+str(i)] = PixToLam(eval("t"+str(i)))
    vars()["normI"+str(i)] = eval("I"+str(i))/max(eval("I"+str(i)))
    vars()["filtI"+str(i)] = savgol_filter(eval("normI"+str(i)), 10, 3)

    lam_max = eval("lam"+str(i))[np.argmax(eval("normI"+str(i)))]
    g_mu1, l_mu1, u_mu1 = lam_max, lam_max-4, lam_max+4  
    g_mu2, l_mu2, u_mu2 = lam_max+42.3, lam_max-10+42.3, lam_max+10+42.3
    g_mu3, l_mu3, u_mu3 = lam_max+90, lam_max-25+90, lam_max+25+90

    vars()["popt"+str(i)], vars()["pcov"+str(i)] = curve_fit(multi_voigt, eval("lam"+str(i)), eval("filtI"+str(i)), 
                                                    p0=[g_eta1, g_mu1, g_sig1, g_a1, g_b1, g_eta2, g_mu2, g_sig2, g_a2, g_b2, g_eta3, g_mu3, g_sig3, g_a3, g_b3],
                                                    bounds=([l_eta1, l_mu1, l_sig1, l_a1, l_b1, l_eta2, l_mu2, l_sig2, l_a2, l_b2, l_eta3, l_mu3, l_sig3, l_a3, l_b3], 
                                                            [u_eta1, u_mu1, u_sig1, u_a1, u_b1, u_eta2, u_mu2, u_sig2, u_a2, u_b2, u_eta3, u_mu3, u_sig3, u_a3, u_b3]))
    errs = np.sqrt(np.diag(eval("pcov"+str(i))))
    lam_dict[str(i)] = [eval("popt"+str(i))[1], eval("popt"+str(i))[6], eval("popt"+str(i))[11], errs[1], errs[6], errs[11]]
    # fig = plt.figure(figsize=(15, 7)) #15, 7
    # fig.subplots_adjust(left=0.13, bottom=0.11, right=0.99, top=1.05, wspace=0, hspace=-0.04)
    # gs = fig.add_gridspec(2, 1, height_ratios=[1,5])
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.imshow(eval("img"+str(i)))
    # ax1.set_yticks([])
    # ax1.set_xticks([]) 
    # ax2 = fig.add_subplot(gs[1, 0])
    # ax2.plot(eval("lam"+str(i)), eval("normI"+str(i)), color="darkcyan", lw=1, label="Moelkül-Spektrum")
    # ax2.plot(lam_plot, multi_voigt(lam_plot, *eval("popt"+str(i))), color="crimson", label="Voigt-Fit")
    # ax2.plot(lRef, IRef, lw=2, color="grey", alpha=0.5, label="Ensemble-Spektrum")
    # ax2.set_xlim(min(eval("lam"+str(i))), max(eval("lam"+str(i))))
    # ax2.set_ylim(-0.02, 1.02)
    # ax2.set_ylabel("normierte Intensität")
    # ax2.set_xlabel(r"$\lambda\,/\,$nm")
    # ax2.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=20)
    # ax2.xaxis.set_major_locator(MultipleLocator(100))
    # ax2.xaxis.set_minor_locator(MultipleLocator(50))
    # ax1.text(x=0.02, y=0.5, s="A", color="white", fontsize=25, horizontalalignment='center', 
    #      verticalalignment='center', transform=ax1.transAxes)
    # ax2.text(x=0.02, y=0.95, s="B", color="k", fontsize=25, horizontalalignment='center', 
    #      verticalalignment='center', transform=ax2.transAxes)

    #plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Bilder/Spektren/SP"+str(i)+"-"+str(int(max(eval("I"+str(i)))))+".pdf", bbox_inches="tight")
    #plt.show()  

    #Erg_Voigt(eval("popt"+str(i)), np.sqrt(np.diag(eval("pcov"+str(i)))), 5)

# with open("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/EinzelmolekülSpek/Programme/daten.txt", "w") as f:
#     f.write("num \t lam1 \t lam2 \t lam3 \t s1 \t s2 \t s3 \n")
#     for key, va in lam_dict.items():
#         f.write("%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n" % (key, va[0], va[1], va[2], va[3], va[4], va[5]))





