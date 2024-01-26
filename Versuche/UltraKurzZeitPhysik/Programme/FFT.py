import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import windows

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
Timestep = (data[2,0] - data[1,0])
freq = fftfreq(Nfft, d=Timestep)

FT_1 = fft(data[:,1], Nfft)/N1
FT_2 = fft(data2[:,1], Nfft)/N2


FT_2R = np.reshape(FT_2, (32, int(Nfft/32))).mean(axis=-1)
FT_1R = np.reshape(FT_1, (32, int(Nfft/32))).mean(axis=-1)
trans = abs(np.divide(FT_2R[:Nfft//2], FT_1R[:Nfft//2]))

plt.plot(range(32), trans)

#plt.xlim(0.4e12, 2.25e12)
plt.show()

# plt.plot(freq, abs(FT_2)/abs(FT_1))
# plt.show()
data[:,1] -= data[0,1]
data2[:,1] -= data2[0,1]

diff1 = round(2 * (abs(len(data[:,1]) * 0.5 - np.argmin(data[:,1]))))
diff2 = round(2 * (abs(len(data2[:,1]) * 0.5 - np.argmin(data2[:,1]))))
exRef = np.pad(data[:,1], (diff1, 5000))
exSam = np.pad(data2[:,1], (diff2, 5000))
FT_1_1 = np.abs(fft(exRef*windows.hann(len(exRef)), 16000)/16000)
FT_2_2 = np.abs(fft(exSam*windows.hann(len(exSam)), 16000)/16000)
plt.plot(freq[:Nfft//2], (FT_1_1/FT_2_2)[:Nfft//2])
plt.show()

new1 = np.reshape(FT_1_1, (1000, int(16000/1000))).mean(axis=-1)
new2 = np.reshape(FT_2_2, (1000, int(16000/1000))).mean(axis=-1)

plt.plot(range(1000), new1/new2)
plt.show()
