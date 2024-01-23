import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n

data = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/REF.dat")
data2 = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/pSil.dat")

plt.plot(data[:,0], data[:,1], c="g")
plt.plot(data2[:,0], data2[:,1], c="k")
plt.show()

N1=data.shape[0]
N2=data2.shape[0]
Nfft = nextpow2(N1)*8
Timestep = (data[2,0] - data[1,0])*10e-12
freq = fftfreq(Nfft, d=Timestep)

FT_1 = fft(data[:,1], Nfft)/N1
FT_2 = fft(data2[:,1], Nfft)/N2

trans = abs(np.divide(FT_2[:Nfft//2], FT_1[:Nfft//2]))

plt.plot(freq[:Nfft//2], trans)

plt.xlim(0.4e12, 2.25e12)
plt.show()

plt.plot(freq, abs(FT_2)/abs(FT_1))
plt.show()
