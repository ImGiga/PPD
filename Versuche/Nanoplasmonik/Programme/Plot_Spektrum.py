import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.optimize import minimize

def Gauss(x, params):
    a,b,c,d = params
    return a*np.exp( -0.5*(x-b)**2/(c**2)) + d


path = os.path.dirname(__file__) + "/../Daten/"
x, Ilamp0 = np.loadtxt(path + "Lampe_Signal_0_at_679.99nm_cut_at_1286.00Y_01.dat", unpack=True, skiprows=6)
x, Idark = np.loadtxt(path + "Rausch_Signal_at_679.99nm_cut_at_1286.00Y_01.dat", unpack=True, skiprows=6)
maxs = []
for file in os.listdir(path):
    if (file.count("-0_at") == 1) and (file[:2] == "90"):
        print(file)
        x, I = np.loadtxt(path+file, unpack=True, skiprows=6)
        name = file[2:6]
        pltI = I / Ilamp0

        def fit_Gauss(params):
            a,b,c,d = params
            return np.sum(np.square(pltI - Gauss(x, (a,b,c,d)) )) 

        Res = minimize(fit_Gauss, [1,680,1,1])
       
        max_ind = x[np.argmax(I)]
        maxs.append(max_ind)
        plt.plot(x, pltI, label=name)
        plt.plot(x, Gauss(x, Res.x))
plt.legend()
plt.show()


x, I = np.loadtxt(path+file, unpack=True, skiprows=6)






