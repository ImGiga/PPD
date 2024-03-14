import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob

def sort_key_conc(file):
    # Split the filename by underscore and convert the first part to a float
    return int(file.split('_')[0].replace('mg', ''))

def sort_key_rot(file):
    # Split the filename by underscore and convert the first part to an integer
    return int(file.split('_')[0].replace('rpm', ''))

file_path_rot_refl = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Daten_Variation_Rotationsgeschw/Reflektion'
os.chdir(file_path_rot_refl)

files_refl = sorted(glob.glob('*rpm_refl*.xy'), key=sort_key_rot)


chunks_refl = [files_refl[i:i+6] for i in range(0, len(files_refl), 6)]

fig_counter = 1

fig, axs = plt.subplots(8, 1, figsize=(8,10))  # Create a 2x4 grid of subplots
axs = axs.ravel()  # Flatten the array of axes

for i, chunk in enumerate(chunks_refl, start=1):
    print(chunk)
    all_y_values = []  # List to store all y-values
    for file in chunk:
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        axs[i-1].plot(data[:,0], data[:,1], alpha=0.3)
        all_y_values.append(data[:,1])  # Append y-values to the list
    mean_y_values = np.mean(all_y_values, axis=0)  # Calculate the mean y-value for each x-value
    axs[i-1].plot(data[:,0], mean_y_values, color='black')  # Plot the mean on each subplot
    axs[i-1].set_xlabel('Wellenlänge [nm]')
    axs[i-1].set_ylabel('Reflektion [%]')
    # axs[i-1].legend()
    axs[i-1].set_title(title)

plt.tight_layout()
plt.show()
globals()[f'fig{fig_counter}'] = fig  # Store the figure in a variable with a dynamic name
fig_counter += 1

file_path_rot_trans = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Daten_Variation_Rotationsgeschw/Transmission'
os.chdir(file_path_rot_trans)

files_trans = sorted(glob.glob('*rpm_trans*.xy'), key=sort_key_rot)

chunks_trans = [files_trans[i:i+6] for i in range(0, len(files_trans), 6)]  

fig, axs = plt.subplots(8,1, figsize=(8, 10))  # Create a 2x4 grid of subplots
axs = axs.ravel()  # Flatten the array of axes

for i, chunk in enumerate(chunks_trans, start=1):
    print(chunk)
    all_y_values = []  # List to store all y-values
    for file in chunk:
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        axs[i-1].plot(data[:,0], data[:,1], alpha=0.3)
        all_y_values.append(data[:,1])  # Append y-values to the list
    mean_y_values = np.mean(all_y_values, axis=0)  # Calculate the mean y-value for each x-value
    axs[i-1].plot(data[:,0], mean_y_values, color='black')  # Plot the mean on each subplot
    axs[i-1].set_xlabel('Wellenlänge [nm]')
    axs[i-1].set_ylabel('Transmission [%]')
    # axs[i-1].legend()
    axs[i-1].set_title(title)

plt.tight_layout()
plt.show()
globals()[f'fig{fig_counter}'] = fig  # Store the figure in a variable with a dynamic name
fig_counter += 1

file_path_conc_refl = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Gruppe1/Reflexion'
os.chdir(file_path_conc_refl)

files_conc_refl = sorted(glob.glob('*mg_ml*.xy'), key=sort_key_conc)

chunks_conc_refl = [files_conc_refl[i:i+6] for i in range(0, len(files_conc_refl), 6)]

fig, axs = plt.subplots(8, 1, figsize=(8, 10))  # Create a 2x4 grid of subplots
axs = axs.ravel()  # Flatten the array of axes

for i, chunk in enumerate(chunks_conc_refl, start=1):
    print(chunk)
    all_y_values = []  # List to store all y-values
    for file in chunk:
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        axs[i-1].plot(data[:,0], data[:,1], alpha=0.3)
        all_y_values.append(data[:,1])  # Append y-values to the list
    mean_y_values = np.mean(all_y_values, axis=0)  # Calculate the mean y-value for each x-value
    axs[i-1].plot(data[:,0], mean_y_values, color='black')  # Plot the mean on each subplot
    axs[i-1].set_xlabel('Wellenlänge [nm]')
    axs[i-1].set_ylabel('Reflektion [%]')
    # axs[i-1].legend()
    axs[i-1].set_title(title + '/mL')

plt.tight_layout()
plt.show()
globals()[f'fig{fig_counter}'] = fig  # Store the figure in a variable with a dynamic name
fig_counter += 1

file_path_conc_trans = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Gruppe1/Transmission'
os.chdir(file_path_conc_trans)

files_conc_trans = sorted(glob.glob('*mg_ml.xy'), key=sort_key_conc)

chunks_conc_trans = [files_conc_refl[i:i+6] for i in range(0, len(files_conc_refl), 6)]

fig, axs =  plt.subplots(8, 1, figsize=(8, 10))  # Create a 2x4 grid of subplots
axs = axs.ravel()  # Flatten the array of axes

for i, chunk in enumerate(chunks_conc_trans, start=1):
    print(chunk)
    all_y_values = []  # List to store all y-values
    for file in chunk:
        data = np.loadtxt(file, delimiter=',')
        title = file.split('_')[0] 
        axs[i-1].plot(data[:,0], data[:,1], alpha=0.3)
        all_y_values.append(data[:,1])  # Append y-values to the list
    mean_y_values = np.mean(all_y_values, axis=0)  # Calculate the mean y-value for each x-value
    axs[i-1].plot(data[:,0], mean_y_values, color='black')  # Plot the mean on each subplot
    axs[i-1].set_xlabel('Wellenlänge [nm]')
    axs[i-1].set_ylabel('Transmission [%]')
    # axs[i-1].legend()
    axs[i-1].set_title(title + '/mL')

plt.tight_layout()
plt.show()
globals()[f'fig{fig_counter}'] = fig  # Store the figure in a variable with a dynamic name
fig_counter += 1