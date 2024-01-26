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

# HR: 7.527 - Ref: 7.045
cutlow_Sample = np.argmin(abs(data[:,0] - 0))
cuthigh_Sample = np.argmin(abs(data[:,0] - data[-1,0]))
cutlow_Ref = np.argmin(abs(data2[:,0] - 0))
cuthigh_Ref = np.argmin(abs(data2[:,0] - 7.045))

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
    wind = windows.hamming(length)
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
# plt.show()
# plt.plot(tR, Ref)
# plt.show()
# plt.plot(t_Sample, Sym_Sample, t_Sample, Wind_Sample)
# plt.show()
# plt.plot(t_Ref, Sym_Ref, t_Ref, Wind_Ref)
# plt.show()

# plt.plot(f_plot, transmission)
# plt.xlim(0.4, 2.25)
# plt.ylim(0.5, 0.9)
# plt.show()

def n(omega, omega_p, gamma):
    return np.sqrt( 3.4175**2 - (omega_p**2 / (omega**2 - 1j*gamma*omega)) )
def Drude(omega, params):
    omega_p, gamma = params
    d = 380 #mum
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

plt.plot(f_plot, Drude(f_plot, Res.x), f_plot, transmission)
plt.show()
