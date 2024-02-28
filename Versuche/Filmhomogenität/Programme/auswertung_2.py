import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

file_path_refl = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten_Variation_Rotationsgeschw/Reflektion'
os.chdir(file_path_refl)

files_refl = sorted(glob.glob('*.xy'))


chunks_refl = [files_refl[i:i+6] for i in range(0, len(files_refl), 6)]

for chunk in chunks_refl:
    print(chunk)
    plt.figure()
    for i, file in enumerate(chunk, start=1):
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        plt.plot(data[:,0], data[:,1], label='Messung ' + str(i))
        plt.xlabel('Wellenlänge [nm]')
        plt.ylabel('Reflektion [%]')
        plt.title('Reflektion ' + title)
    plt.legend()
    plt.show()

file_path_trans = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten_Variation_Rotationsgeschw/Transmission'
os.chdir(file_path_trans)

files_trans = sorted(glob.glob('*.xy'))

chunks_trans = [files_trans[i:i+6] for i in range(0, len(files_trans), 6)]  

for chunk in chunks_trans:
    print(chunk)
    plt.figure()    
    for i, file in enumerate(chunk, start=1):
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        plt.plot(data[:,0], data[:,1], label='Messung ' + str(i))
        plt.xlabel('Wellenlänge [nm]')
        plt.ylabel('Transmission [%]')
        plt.title('Transmission ' + title)
    plt.legend()
    plt.show()