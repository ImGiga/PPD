import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.optimize import minimize, curve_fit
from scipy.fftpack import rfft, irfft, rfftfreq, fft, fftfreq, ifft

def fft_filter(prof, max_freq, freq):
    f_signal = rfft(prof)    
    cut_f_signal = f_signal.copy()
    cut_f_signal[(freq>max_freq)] = 0
    cut_signal = irfft(cut_f_signal)
    return f_signal, cut_signal

daten = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/C4Auto00000.txt", skiprows=6)
print(daten.shape)

x = np.linspace(-0.822647, 0.822647, len(daten[:,0]))
Timestep = x[3] - x[2]

freq = rfftfreq(len(x), d=Timestep)

ft, filtered = fft_filter(-1*daten[:,1], 6, freq)


# plt.plot(daten[:,0], -1*daten[:,1], daten[:,0], filtered)
# plt.show()


def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n

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

def FFT(values):
    N1 = len(values)
    Nfft = nextpow2(N1)*4
    freq = rfftfreq(Nfft, d=Timestep)
    return freq, np.abs(rfft(values, Nfft))


Sym_Sample = symmetric(-1*daten[:,1])
Wind_Sample = window(Sym_Sample)
f, ft = FFT(Wind_Sample)

# d2 = np.zeros((len(f),2))
# d2[:, 0] = f
# d2[:, 1] = ft
# np.save("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/FT_auto.npy", d2)

# plt.plot(f, ft)
# plt.xlim(0, 50)
# plt.show()

xF = np.max(filtered)
xS = np.max(-1*daten[:,1])
print(xS/xF, 8/3)

def Gauss(x, params):
    a,b,c,d = params
    return a*np.exp( -0.5*(x-b)**2/(c**2)) + d

def fit_Gauss(params):
    a,b,c,d = params
    return np.sum(np.square(filtered/np.max(-1*daten[:,1]) - Gauss(x, (a,b,c,d)) )) 

Res = minimize(fit_Gauss, [0.01,-0.01,0.05,0.04])
print(Res)
plt.plot(x, filtered/np.max(-1*daten[:,1]), x, Gauss(x, Res.x))
#plt.hlines(0.5*(np.max(Gauss(x, Res.x))-Res.x[3])+Res.x[3], -1, 1)
plt.show()

print(2*np.sqrt(2*np.log(2))*Res.x[2]/np.sqrt(2))

print(np.abs(Res.x))
print(np.abs(np.diag(Res.hess_inv)))
print(np.sqrt(np.diag(Res.hess_inv)))
print(np.sqrt(Res.jac))

# d3 = np.zeros((len(x), 3))
# d3[:,0] = x
# d3[:,1] = filtered
# d3[:,2] = Gauss(x, Res.x)
# np.save("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/Gauss_auto.npy", d3)