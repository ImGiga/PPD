import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from scipy.optimize import curve_fit

def read_csv_files_and_calculate_stats(path):
    all_files = glob.glob(path + "/*.csv")
    stats = []

    for filename in all_files:
        df = pd.read_csv(filename, delimiter=';', usecols=[1, 2], names=['thickness', 'fit'])
        mean_thickness = df['thickness'].mean()
        std_thickness = df['thickness'].std()
        mean_div_std = mean_thickness / std_thickness if std_thickness != 0 else None
        base_filename = os.path.basename(filename)
        filename_without_ext = os.path.splitext(base_filename)[0]
        fit = df['fit'].mean()
        stats.append((mean_thickness, std_thickness, mean_div_std, fit, filename_without_ext))

    stats_df = pd.DataFrame(stats, columns=['$<d>$ in nm', '$std(d)$ in nm', '$h$', '$X$', 'filename'])
    stats_df.insert(0, 'Drehzahl in $rpm$', stats_df['filename'].str.extract('(\d+)rpm').astype(float))
    stats_df.insert(0, 'Konzentration in $\si{\milli\gram\per\milli\litre}$', stats_df['filename'].str.extract('(\d+)mg').astype(float))
    return stats_df

def schubert(w, A):
    return A * (1950/w)**(1/2) * (100/20) * (35/100)**(1/4)

def schubertconst(c, A):
    return A * (1950/1000)**(1/2) * (c/20) * (35/100)**(1/4)


file_path_refl_rpm = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Daten_Variation_Rotationsgeschw/Reflektion'
file_path_trans_rpm = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Daten_Variation_Rotationsgeschw/Transmission'
file_path_refl_conc = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Gruppe1/Reflexion'
file_path_trans_conc = r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Daten/Gruppe1/Transmission'

stats_refl_rpm = read_csv_files_and_calculate_stats(file_path_refl_rpm)
stats_trans_rpm = read_csv_files_and_calculate_stats(file_path_trans_rpm)
stats_refl_conc = read_csv_files_and_calculate_stats(file_path_refl_conc)
stats_trans_conc = read_csv_files_and_calculate_stats(file_path_trans_conc)

stats_refl_rpm.sort_values(by='Drehzahl in $rpm$', inplace=True)
stats_trans_rpm.sort_values(by='Drehzahl in $rpm$', inplace=True)
stats_refl_conc.sort_values(by='Konzentration in $\si{\milli\gram\per\milli\litre}$', inplace=True)
stats_trans_conc.sort_values(by='Konzentration in $\si{\milli\gram\per\milli\litre}$', inplace=True)
print(stats_trans_rpm)

geschw = np.array([stats_refl_rpm['Drehzahl in $rpm$']]).flatten()
dicke_trans_rpm = np.array([stats_trans_rpm['$<d>$ in nm']]).flatten()
dicke_refl_rpm = np.array([stats_refl_rpm['$<d>$ in nm']]).flatten()
guete_trans_rpm = np.array([stats_trans_rpm['$X$']]).flatten()
guete_refl_rpm = np.array([stats_refl_rpm['$X$']]).flatten()

conc = np.array([stats_refl_conc['Konzentration in $\si{\milli\gram\per\milli\litre}$']]).flatten()
dicke_trans_conc = np.array([stats_trans_conc['$<d>$ in nm']]).flatten()
dicke_refl_conc = np.array([stats_refl_conc['$<d>$ in nm']]).flatten()
guete_trans_conc = np.array([stats_trans_conc['$X$']]).flatten()
guete_refl_conc = np.array([stats_refl_conc['$X$']]).flatten()

# plt.plot(dicke_trans_rpm, guete_trans_rpm, 'x', c='r', label='Transmission')
# plt.plot(dicke_refl_rpm, guete_refl_rpm, 'x', c='b', label='Reflektion')
# plt.xlabel('Dicke in nm')
# plt.ylabel(r'Fitgüte $\chi$')
# plt.legend()
# fig = plt.gcf()
# fig.set_size_inches(7,5)
# # plt.show()
# plt.savefig(r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Bilder/guete_dicke_rpm.png', transparent=True, dpi=300)

stats_refl_rpm.drop('filename', axis=1, inplace=True)
stats_trans_rpm.drop('filename', axis=1, inplace=True)
stats_refl_conc.drop('filename', axis=1, inplace=True)
stats_trans_conc.drop('filename', axis=1, inplace=True)

stats_refl_rpm.drop('Konzentration in $\si{\milli\gram\per\milli\litre}$', axis=1, inplace=True)
stats_trans_rpm.drop('Konzentration in $\si{\milli\gram\per\milli\litre}$', axis=1, inplace=True)
stats_refl_conc.drop('Drehzahl in $rpm$', axis=1, inplace=True)
stats_trans_conc.drop('Drehzahl in $rpm$', axis=1, inplace=True)

stats_rpm = stats_refl_rpm
stats_rpm.insert(2, '$<d_T>$ in nm', stats_trans_rpm['$<d>$ in nm'])
stats_rpm.insert(4, '$std(d_T)$ in nm', stats_trans_rpm['$std(d)$ in nm'])
stats_rpm.insert(6, '$h_T$', stats_trans_rpm['$h$'])
stats_rpm.insert(8, '$X_T$', stats_trans_rpm['$X$'])
stats_rpm.rename(columns={'$<d>$ in nm': '$<d_R>$ in nm', '$std(d)$ in nm': '$std(d_R)$ in nm', '$h$': '$h_R$', '$X$': '$X_R$'}, inplace=True)
stats_rpm.sort_values(by='Drehzahl in $rpm$', inplace=True) 

stats_conc = stats_refl_conc
stats_conc.insert(2, '$<d_T>$ in nm', stats_trans_conc['$<d>$ in nm'])
stats_conc.insert(4, '$std(d_T)$ in nm', stats_trans_conc['$std(d)$ in nm'])
stats_conc.insert(6, '$h_T$', stats_trans_conc['$h$'])
stats_conc.insert(8, '$X_T$', stats_trans_conc['$X$'])
stats_conc.rename(columns={'$<d>$ in nm': '$<d_R>$ in nm', '$std(d)$ in nm': '$std(d_R)$ in nm', '$h$': '$h_R$', '$X$': '$X_R$'}, inplace=True)
stats_conc.sort_values(by='Konzentration in $\si{\milli\gram\per\milli\litre}$', inplace=True)

# print(stats_rpm)
# print(stats_conc)



latex_table_rpm = stats_rpm.style.format({
    'Drehzahl in $rpm$': "{:.0f}",
    '$<d_R>$ in nm': "{:.2f}",
    '$<d_T>$ in nm': "{:.2f}",
    '$std(d_R)$ in nm': "{:.2f}",
    '$std(d_T)$ in nm': "{:.2f}",
    '$h_R$': "{:.2f}",
    '$h_T$': "{:.2f}",
    '$X_R$': "{:.6f}",
    '$X_T$': "{:.6f}"
}).to_latex()

latex_table_conc = stats_conc.style.format({
    'Konzentration in $\si{\milli\gram\per\milli\litre}$': "{:.0f}",
    '$<d_R>$ in nm': "{:.2f}",
    '$<d_T>$ in nm': "{:.2f}",
    '$std(d_R)$ in nm': "{:.2f}",
    '$std(d_T)$ in nm': "{:.2f}",
    '$h_R$': "{:.2f}",
    '$h_T$': "{:.2f}",
    '$X_R$': "{:.6f}",
    '$X_T$': "{:.6f}"
}).to_latex()

# print(latex_table_conc)
print(latex_table_rpm)
# print(stats_trans_rpm)
# print(stats_refl_conc)
# print(stats_trans_conc)
fit_refl = curve_fit(schubert, geschw, dicke_refl_rpm)
fit_trans = curve_fit(schubert, geschw, dicke_trans_rpm)
xrange = np.linspace(0,7000)

# plt.plot(geschw, dicke_trans_rpm, 'x', c='r', label='Transmissionsmessung')
# plt.plot(geschw, dicke_refl_rpm, 'x', c='b', label='Reflektionsmessung')
# # plt.plot(xrange, schubert(xrange, *fit_refl[0]), c='b', label='Fit Reflektionsmessung')
# plt.plot(xrange, schubert(xrange, *fit_trans[0]), c='g', label='Schubert-Fit')
# plt.xlabel('Rotationsgeschwindigkeit in rpm', fontsize=24)
# plt.ylabel('Schichtdicke in nm', fontsize=24)
# plt.legend(fontsize=22)
# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
# print(fit_trans)
# # plt.show()
# plt.savefig(r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Bilder/schubertfit.png', transparent=True, dpi=300)
# perr = np.sqrt(np.diag(fit_trans[1]))
# print(perr)
conc_1 = conc[:4+1]
conc_2 = conc[4:]
dicke_trans_1 = dicke_trans_conc[:4+1]
dicke_trans_2 = dicke_trans_conc[4:]
dicke_refl_1 = dicke_refl_conc[:4+1]
dicke_refl_2 = dicke_refl_conc[4:]
polyfit = np.polyfit(conc_2, dicke_refl_2, 1)
schubert_fit_1_trans = curve_fit(schubertconst, conc_1, dicke_trans_1)
schubert_fit_2_trans = curve_fit(schubertconst, conc_2, dicke_trans_2)
schubert_fit_1_refl = curve_fit(schubertconst, conc_1, dicke_refl_1)
schubert_fit_2_refl = curve_fit(schubertconst, conc_2, dicke_refl_2)

print(conc_2, dicke_refl_2)
xrange_1 = np.linspace(0,200)
xrange_2 = np.linspace(130,310)
print(conc_1, conc_2, conc)
plt.plot(conc, dicke_trans_conc, 'x', c='r', label='Transmissionsmessung')
plt.plot(conc, dicke_refl_conc, 'x', c='b', label='Reflektionsmessung')
plt.plot(xrange_1, schubertconst(xrange_1, *schubert_fit_1_trans[0]), c='g', label='Schubert-Fits')
# plt.plot(xrange_2, schubertconst(xrange_2, *schubert_fit_2_trans[0]), c='g')
# plt.plot(xrange_1, schubertconst(xrange_1, *schubert_fit_1_refl[0]), c='y')
# plt.plot(xrange_2, schubertconst(xrange_2, *schubert_fit_2_refl[0]), c='y')
plt.plot(xrange_2, polyfit[0] * xrange_2 + polyfit[1], c='g')
plt.xlabel('Konzentration in mg/ml', fontsize=24)
plt.ylabel('Schichtdicke in nm', fontsize=24)
plt.legend(fontsize=22)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
# plt.savefig(r'/Users/max/Documents/Uni Bayreuth/Master/Semester 1/PPD/Versuche/Filmhomogenität/Bilder/schubert_conc.png', transparent=True, dpi=300)