import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

w_mess, I_mess, non = np.loadtxt('/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Röntgenbeugung/Daten/Messung 1/REFVORN.STN', skiprows=22, unpack=True)

plt.plot(w_mess, I_mess, label='Messung')
plt.show()