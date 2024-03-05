import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def read_csv_files_and_calculate_stats(path):
    all_files = glob.glob(path + "/*.csv")
    stats = []

    for filename in all_files:
        df = pd.read_csv(filename, delimiter=';', usecols=[1, 2], names=['thickness', 'fit'])
        mean_thickness = df['thickness'].mean()
        std_thickness = df['thickness'].std()
        mean_fit = df['fit'].mean()
        std_fit = df['fit'].std()
        mean_div_std = mean_thickness / std_thickness if std_thickness != 0 else None
        base_filename = os.path.basename(filename)
        filename_without_ext = os.path.splitext(base_filename)[0]
        stats.append((filename_without_ext, mean_thickness, std_thickness, mean_div_std, mean_fit, std_fit))

    stats_df = pd.DataFrame(stats, columns=['filename', 'mean', 'std_dev', 'mean_div_std', 'mean_fit', 'std_fit'])
    return stats_df

file_path_refl_rpm = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Daten_Variation_Rotationsgeschw/Reflektion'
file_path_trans_rpm = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Daten_Variation_Rotationsgeschw/Transmission'
file_path_refl_conc = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Gruppe1/Reflexion'
file_path_trans_conc = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Gruppe1/Transmission'

stats_refl_rpm = read_csv_files_and_calculate_stats(file_path_refl_rpm)
stats_trans_rpm = read_csv_files_and_calculate_stats(file_path_trans_rpm)
stats_refl_conc = read_csv_files_and_calculate_stats(file_path_refl_conc)
stats_trans_conc = read_csv_files_and_calculate_stats(file_path_trans_conc)

print(stats_refl_rpm)
print(stats_trans_rpm)
print(stats_refl_conc)
print(stats_trans_conc)
