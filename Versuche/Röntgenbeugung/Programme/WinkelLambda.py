import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter
from icecream import ic
from cycler import cycler
import locale

locale.setlocale(locale.LC_NUMERIC, "de_DE")
c_cycle = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])
mpl.rcParams.update({'font.size': 18, 'axes.labelpad': 0, 'axes.linewidth': 1.2, 'axes.titlepad': 8.0,
                    'legend.framealpha': 0.4, 'legend.borderaxespad': 0.3, 'legend.borderpad': 0.2,
                    'legend.labelspacing': 0, 'legend.handletextpad': 0.2, 'legend.handlelength': 1.0,
                    'legend.loc': 'best', 'xtick.labelsize': 'small', 'xtick.major.pad': 2, 
                    'xtick.major.size': 3, 'xtick.major.width': 1.2, 'ytick.labelsize': 'small',
                    'ytick.major.pad': 2, 'ytick.major.size': 3, 'ytick.major.width': 1.2,
                     'axes.prop_cycle': cycler(color=c_cycle)})

rel_path = "C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/Röntgenbeugung/Daten/Messung 1/"
h = 4.135136*1e-18    # keV s
c = 299792458   # m / s
d = 1.93146 # A
lambda_Theo = np.array([0.17892, 0.17892, 0.17942, 0.17960, 0.18310, 0.18327, 0.18438, 0.18518, 0.20901, 0.21383,
                       1.06146, 1.06786, 1.09857, 1.24430, 1.26271, 1.28181, 1.30164, 1.42112, 1.47631, 1.48745]) # A

def calc_lambda(degree, n=1):
    return 2*d*np.sin(np.radians(degree))/n

def calc_theta(lam, n=1):
    return np.arcsin(n*lam / 2.0 / d) * 180 / np.pi

def correct_Theta(theta,m,b):
    return m*theta + b

def calc_mu(I, I0, lam):
    return -1.0 / lam * np.log(I / I0)

def error_theta(theta, sm, sb):
    return np.sqrt( (theta*sm)**2 + sb**2 )
def error_lambda(theta, st, n=1, d=d):
    return np.abs( 2*d/n *np.cos( theta ) * st )


theo_lam_1 = np.array([1.06786, 1.09857, 1.24430, 1.26271, 1.28181, 1.30164, 1.47631]) # 1.06146, 
theo_lam_2 = np.array([2*1.06786, 2*1.09857, 2*1.24430, 2*1.26271, 2*1.28181]) # 2*1.06146, 
theta_Theo = np.concatenate((calc_theta(theo_lam_1), calc_theta(theo_lam_2, 1)))
omega_Mess = np.array([15.927, 16.447, 18.733, 19.012, 19.324, 19.650, 22.435, # 15.346, 
                      33.333, 34.634, 40.093, 40.807, 41.581]) # 32.130, 
ic.disable()
ic( calc_theta(lambda_Theo) )
ic( omega_Mess )
ic( theta_Theo )

vals, cov = np.polyfit(x=omega_Mess, y=theta_Theo, deg=1, cov=True)
m, b  = vals[0], vals[1]
sm, sb = np.sqrt(np.diag(cov))

ic( m, sm, b, sb)
ic.enable()
# Plot
# px = np.linspace(min(omega_Mess)-2, max(omega_Mess)+2, 100)
# plt.plot(omega_Mess, theta_Theo, ls="none", marker="+", markersize=12, color="red", label=r"$\Theta_{\mathrm{theo}}$") 
# plt.plot(px, m*px+b, color="cadetblue", zorder=-1, lw=1.2)
# plt.xlim(min(px), max(px))
# plt.ylim(min(px), max(px))
# plt.xlabel(r"$\Theta_{\mathrm{mess}} [^{\circ}]$")
# plt.ylabel(r"$\Theta_{\mathrm{kali}} [^{\circ}]$")
# plt.legend(loc="upper left", facecolor="wheat", framealpha=0.5, fontsize=18)
# plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/Röntgenbeugung/Bilder/fit.pdf", bbox_inches="tight")


def load_data(path):
    w, I, non = np.loadtxt(rel_path + path, skiprows=22, unpack=True)
    return w, I

T_ref_back, I_ref_back = load_data("REFHINT.STN")
T_ref_front, I_ref_front = load_data("REFVORN.STN")
T_met_back, I_met_back = load_data("METHINT.STN")
T_met_front, I_met_front = load_data("METVORN.STN")

T_ref_ges = np.concatenate((T_ref_front[:np.argmin(abs(T_ref_front-22.5))+1], T_ref_back))
T_ref_corr = correct_Theta(theta=T_ref_ges, m=m, b=b)
lambda_ref = calc_lambda(T_ref_corr, 1)
I_ref_ges = np.concatenate((I_ref_front[:np.argmin(abs(T_ref_front-22.5))+1], I_ref_back*1.5/5))
I_ref_filt = savgol_filter(I_ref_ges, 15, 3)
T_met_ges = np.concatenate((T_met_front[:np.argmin(abs(T_met_front-22.5))+1], T_met_back))
T_met_corr = correct_Theta(theta=T_met_ges, m=m, b=b)
lambda_met = calc_lambda(T_met_corr, 1)
I_met_ges = np.concatenate((I_met_front[:np.argmin(abs(T_met_front-22.5))+1], I_met_back*1.5/5))
I_met_filt = savgol_filter(I_met_ges, 15, 3)

ic( (lambda_ref == lambda_met).all() )
mu = calc_mu(I_met_filt, I_ref_filt, lambda_ref)

# Plot
# plt.plot(T_ref_ges, I_ref_ges/max(I_ref_ges), color="mediumseagreen", lw=1)
# # secax = plt.gca().secondary_xaxis('top')
# # secax.xaxis.set_ticks(omega_Mess, [str(i) for i in range(len(omega_Mess))], rotation=90)
# ymaxs = [0.8, 0.9, 0.8, 0.85, 0.9, 0.95, 0.8, 0.5, 0.6, 0.5, 0.6, 0.5]
# stringis = [r"$L\gamma_{2}$", r"$L\gamma_{1}$",
#             r"$L\beta_{2}$", r"$L\beta_{3}$", r"$L\beta_{1}$",
#             r"$L\beta_{4}$", r"$L\alpha_{1}$",
#             r"$n=2 \rightarrow L\gamma_{2}$", r"$n=2 \rightarrow L\gamma_{1}$",
#             r"$n=2 \rightarrow L\beta_{2}$", r"$n=2 \rightarrow L\beta_{3}$", r"$n=2 \rightarrow L\beta_{1}$"]
# plt.vlines(x=omega_Mess, 
#            ymin=[I_ref_ges[np.argmin(abs(T_ref_ges - j))]/max(I_ref_ges) + 0.02 for j in omega_Mess], 
#            ymax=ymaxs, color="k", lw=0.7)
# for i, w in enumerate(omega_Mess):
#     plt.text(x=w, y=ymaxs[i], s=str(i+1), horizontalalignment='center',
#      verticalalignment='bottom', fontsize=12)
#     plt.plot([], [], ls="none", label=str(i+1) + ":  " + stringis[i])
# plt.xlim(min(T_ref_ges), max(T_ref_ges))
# plt.ylim(0, 1.01)
# plt.ylabel("normierte Intensität")
# plt.xlabel(r"$\Theta [^{\circ}]$")
# plt.legend(loc="upper right", facecolor="wheat", framealpha=0.5, fontsize=14, ncol=4)
# plt.show()

plt.plot(lambda_ref, I_ref_ges/max(I_ref_ges))
plt.plot(lambda_met, I_met_ges/max(I_ref_ges))
plt.show()

# plt.plot(lambda_ref, mu)
# plt.show()

#plt.plot(T_met_ges, I_met_ges/max(I_met_ges), T_met_ges, I_met_filt/max(I_met_ges))