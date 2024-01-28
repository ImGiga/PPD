import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.optimize import minimize, curve_fit
from scipy.fftpack import rfft, irfft, rfftfreq, fft, fftfreq, ifft

REFDATA = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/REF.dat")
HRDATA = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/HR_SIL.dat")
tRef = REFDATA[:,0]
tHr = HRDATA[:,0]
Ref = REFDATA[:,1]
Hr = HRDATA[:,1]

plt.plot(tHr, Hr)
plt.plot(t)
plt.show()




max1 = 4.8867
max2 = 11.0667
print(tHr[1])

