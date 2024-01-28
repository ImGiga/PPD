import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import windows
from scipy.optimize import minimize, curve_fit

def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n

data = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/pSil.dat")
data2 = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/REF.dat")
Timestep = (data[2,0] - data[1,0])
# plt.plot(data[:,0], data[:,1]-data[0,1], data2[:,0], data2[:,1]-data2[0,1])
# plt.hlines(0, 0, 16)
# plt.show()

# HR: 3.441 - 7.527   - REF: 4.638
cutlow_Sample = np.argmin(abs(data[:,0] - 0))
cuthigh_Sample = np.argmin(abs(data[:,0] - 4.369))
cutlow_Ref = np.argmin(abs(data2[:,0] - 0))
cuthigh_Ref = np.argmin(abs(data2[:,0] - 5.396))

Sample = data[cutlow_Sample:cuthigh_Sample,1] - data[cutlow_Sample, 1]
tS = data[cutlow_Sample:cuthigh_Sample,0] - data[cutlow_Sample, 0]
Ref = data2[cutlow_Ref:cuthigh_Ref,1] - data2[cutlow_Ref, 1]
tR = data2[cutlow_Ref:cuthigh_Ref,0] - data2[cutlow_Ref, 0]

def symmetric(values):
    maxVal = np.argmax(values)
    length = len(values)
    dif_end = length - maxVal
    if dif_end > maxVal:
        sym_vals = np.concatenate((np.zeros(int(dif_end-maxVal)), values))
    else:
        sym_vals = np.concatenate((values, np.zeros(int(maxVal-dif_end))))
    return sym_vals

def window(values):
    length = len(values)
    #windows.flattop
    wind = windows.hann(length)
    return values*wind

def FFT(values, N2):
    N1 = len(values)
    Nfft = nextpow2(max(N1, N2))*32
    freq = fftfreq(Nfft, d=Timestep)
    return freq, np.abs(fft(values, Nfft))

Sym_Sample = symmetric(Sample)
Wind_Sample = window(Sym_Sample)
Sym_Ref = symmetric(Ref)
Wind_Ref = window(Sym_Ref)
t_Sample = range(len(Wind_Sample))*Timestep
t_Ref = range(len(Wind_Ref))*Timestep

f, ft_Sample = FFT(Wind_Sample, len(Wind_Ref))
f2, ft_Ref = FFT(Wind_Ref, len(Wind_Sample))
grenzfmin = np.argmin(abs(f - 0.4))
grenzfmax = np.argmin(abs(f - 2.25))
transmission = (ft_Sample/ft_Ref)[grenzfmin:grenzfmax]
f_plot = f[grenzfmin:grenzfmax]

Theo_HR = 4*3.4175/((1 + 3.4175)**2)
# plt.plot(tS, Sample)
# plt.plot(tR, Ref)
# plt.show()
# plt.plot(t_Sample, Sym_Sample, t_Sample, Wind_Sample)
# plt.show()
# plt.plot(t_Ref, Sym_Ref, t_Ref, Wind_Ref)
# plt.show()

# plt.plot(f_plot, transmission)
# plt.xlim(0.4, 2.25)
# plt.ylim(0.5, 0.9)
# plt.hlines(y=Theo_HR, xmin=0.4, xmax=2.25)
# plt.show()

# d5 = np.zeros((len(f_plot), 2))
# d5[:,0] = f_plot
# d5[:,1] = transmission-0.04
# np.save("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/HRtrans.npy", d5)

def n(omega, omega_p, gamma):
    return np.sqrt( 3.4175**2 - (omega_p**2 / (omega**2 - 1j*gamma*omega)) )
def Drude(omega, params):
    omega_p, gamma = params
    d = 270 #mum
    c = 3 # 10**8
    ind = n(omega, omega_p, gamma)
    fak = 4*ind/((1 + ind)**2)
    exponent = (ind-1)*omega*d/c/100
    return np.abs( fak * np.exp( -1*1j*exponent ) )
def fit_Drude(params):
    omega_p, gamma = params
    return np.sum(np.square(transmission - Drude(f_plot, (omega_p, gamma)) )) 

Res = minimize(fit_Drude, [1.5,1.5])
print(Res)

# plt.plot(f_plot, Drude(f_plot, Res.x), f_plot, transmission)
# plt.ylim(0.49, 1.01)
# plt.xlim(0.4, 2.25)
# plt.show()


# d6 = np.zeros((len(f_plot),3))
# d6[:,0] = f_plot
# d6[:,1] = transmission
# d6[:,2] = Drude(f_plot, Res.x)
# np.save("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/pSi.npy", d6)
print(Res.x)
print(np.sqrt(np.diag(Res.hess_inv)))

e = 1.6022 #e-19
m = 9.11 #e-31
eps_inf = 3.4175**2
eps_0 = 8.854 #e-12

mu = e/m/Res.x[1]  # m**2 / Vs   *e4 cm**2 /Vs
N = Res.x[0]**2*m*eps_inf*eps_0/e/e  #1/m**3 e19 = 1/cm**3 e13
sig = e*mu*N # 1/ Ohm m
rho = 1/sig

s_mu = e/m/Res.x[1]/Res.x[1] * 0.22 # * np.sqrt(np.diag(Res.hess_inv))[1]
s_N = 2*Res.x[0]*m*eps_inf*eps_0/e/e * 0.17 # * np.sqrt(np.diag(Res.hess_inv))[0]
s_sig = np.sqrt( (e*N*s_mu)**2 + (e*mu*s_N)**2 )
s_rho = 1.0/sig/sig*s_sig

print("mu = {}\ts = {}".format(mu*10000, s_mu*1000))
print("N = {}\ts = {}".format(N/1000, s_N/1000))#e16
print("sig = {}\ts = {}".format(sig/100, s_sig/100))
print("rho = {}\ts = {}".format(rho*100, s_rho*100))