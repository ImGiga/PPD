import numpy as np
import matplotlib.pyplot as plt
import os 

path = "/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Nanoplasmonik/Daten/"

x, Ilamp0 = np.loadtxt("/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Nanoplasmonik/Daten/Lampe_Signal_0_at_679.99nm_cut_at_1286.00Y_01.dat", unpack=True, skiprows=6)
x, Idark = np.loadtxt("/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Nanoplasmonik/Daten/Rausch_Signal_at_679.99nm_cut_at_1286.00Y_01.dat", unpack=True, skiprows=6)
maxs = []
for file in os.listdir(path):
    if (file.count("-0_at") == 1) and (file[:2] == "90"):
        print(file)
        x, I = np.loadtxt(path+file, unpack=True, skiprows=6)
        name = file[2:6]
        pltI = I/ Ilamp0
        max_ind = x[np.argmax(I)]
        print(file, max_ind)
        maxs.append(max_ind)
        plt.plot(x, pltI, label=name)
plt.legend()
plt.show()

