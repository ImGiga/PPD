import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

file_path_rot_refl = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Daten_Variation_Rotationsgeschw/Reflektion'
os.chdir(file_path_rot_refl)

files_refl = sorted(glob.glob('*.xy'))


chunks_refl = [files_refl[i:i+6] for i in range(0, len(files_refl), 6)]

fig_counter = 1

for chunk in chunks_refl:
    print(chunk)
    fig = plt.figure()
    all_y_values = []  # List to store all y-values
    for i, file in enumerate(chunk, start=1):
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        plt.plot(data[:,0], data[:,1], label='Messung ' + str(i), alpha=0.5)
        all_y_values.append(data[:,1])  # Append y-values to the list
    mean_y_values = np.mean(all_y_values, axis=0)  # Calculate the mean y-value for each x-value
    plt.plot(data[:,0], mean_y_values, label='Mean', color='black')  # Plot the mean
    plt.xlabel('Wellenlänge [nm]')
    plt.ylabel('Reflektion [%]')
    plt.title('Reflektion ' + title)
    plt.legend()
    plt.show()

    globals()[f'fig{fig_counter}'] = fig  # Store the figure in a variable with a dynamic name
    fig_counter += 1

file_path_rot_trans = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Daten_Variation_Rotationsgeschw/Transmission'
os.chdir(file_path_rot_trans)

files_trans = sorted(glob.glob('*.xy'))

chunks_trans = [files_trans[i:i+6] for i in range(0, len(files_trans), 6)]  

for chunk in chunks_trans:
    print(chunk)
    fig = plt.figure()
    all_y_values = []  # List to store all y-values
    for i, file in enumerate(chunk, start=1):
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        plt.plot(data[:,0], data[:,1], label='Messung ' + str(i), alpha=0.5)
        all_y_values.append(data[:,1])  # Append y-values to the list
    mean_y_values = np.mean(all_y_values, axis=0)  # Calculate the mean y-value for each x-value
    plt.plot(data[:,0], mean_y_values, label='Mean', color='black')  # Plot the mean
    plt.xlabel('Wellenlänge [nm]')
    plt.ylabel('Transmission [%]')
    plt.title('Transmission ' + title)
    plt.legend()
    plt.show()
    globals()[f'fig{fig_counter}'] = fig  # Store the figure in a variable with a dynamic name
    fig_counter += 1

file_path_conc_refl = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Gruppe1/Reflexion'
os.chdir(file_path_conc_refl)

files_conc_refl = sorted(glob.glob('*.xy'))

chunks_conc_refl = [files_conc_refl[i:i+6] for i in range(0, len(files_conc_refl), 6)]

for chunk in chunks_conc_refl:
    fig = plt.figure()
    all_y_values = []  # List to store all y-values
    for i, file in enumerate(chunk, start=1):
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        plt.plot(data[:,0], data[:,1], label='Messung ' + str(i), alpha=0.5)
        all_y_values.append(data[:,1])  # Append y-values to the list
    mean_y_values = np.mean(all_y_values, axis=0)  # Calculate the mean y-value for each x-value
    plt.plot(data[:,0], mean_y_values, label='Mean', color='black')  # Plot the mean
    plt.xlabel('Wellenlänge [nm]')
    plt.ylabel('Reflektion [%]')
    plt.title('Reflektion ' + title + '/mL')
    plt.legend()
    plt.show()

    globals()[f'fig{fig_counter}'] = fig  # Store the figure in a variable with a dynamic name
    fig_counter += 1

file_path_conc_trans = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Gruppe1/Transmission'
os.chdir(file_path_conc_trans)

files_conc_trans = sorted(glob.glob('*.xy'))

chunks_conc_trans = [files_conc_refl[i:i+6] for i in range(0, len(files_conc_refl), 6)]

for chunk in chunks_conc_trans:
    fig = plt.figure()
    all_y_values = []  # List to store all y-values
    for i, file in enumerate(chunk, start=1):
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        plt.plot(data[:,0], data[:,1], label='Messung ' + str(i), alpha=0.5)
        all_y_values.append(data[:,1])  # Append y-values to the list
    mean_y_values = np.mean(all_y_values, axis=0)  # Calculate the mean y-value for each x-value
    plt.plot(data[:,0], mean_y_values, label='Mean', color='black')  # Plot the mean
    plt.xlabel('Wellenlänge [nm]')
    plt.ylabel('Transmission [%]')
    plt.title('Transmission ' + title + '/mL')
    plt.legend()
    plt.show()

    globals()[f'fig{fig_counter}'] = fig  # Store the figure in a variable with a dynamic name
    fig_counter += 1

