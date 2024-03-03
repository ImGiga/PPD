import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lorentzfaktor(two_theta):
    return 1 / (np.sin(np.radians(two_theta / 2))*np.cos(np.radians(two_theta / 2)))

def polarisation(two_theta):
    return (1 + np.cos(np.radians(two_theta))**2) / 2

def atomformfaktor(two_theta, const_a, const_b, const_c):
    results = []
    for t in two_theta:
        sum_result = np.sum([a * np.exp(-b * (np.sin(np.radians(t) / wavelen))**2) for a, b in zip(const_a, const_b)])
        results.append(sum_result + const_c)
    return np.array(results)


def strukturamplitude(hkl, f_cl, f_na, pos_cl, pos_na):
    results = []
    for i, h in enumerate(hkl):
        faktor_1_cos = np.cos(2 * np.pi * np.dot(h, pos_cl))
        faktor_1_sin = np.sin(2 * np.pi * np.dot(h, pos_cl))
        faktor_2_cos = np.cos(2 * np.pi * np.dot(h, pos_na))
        faktor_2_sin = np.sin(2 * np.pi * np.dot(h, pos_na))
        faktor_1 = faktor_1_cos + 1j * faktor_1_sin
        faktor_2 = faktor_2_cos + 1j * faktor_2_sin
        results.append(faktor_1 * f_cl[i] + faktor_2 * f_na[i])
    return np.array(results)

def strukturfaktor(ampl):
    return np.real(ampl * np.conj(ampl))

def intensity(lorfak, polfak, H, strukfak):
    return lorfak * polfak * H * strukfak

def norm(intensity, norm):
    return intensity / norm

wavelen = 1.540593

const_Na_a = np.array([4.76260, 3.17360, 1.26740, 1.11280])
const_Na_b = np.array([3.28500, 8.84220, 0.313600, 129.424])
const_Na_c = 0.676000

const_Cl_a = np.array([11.4604, 7.19650, 6.25560, 1.64550])
const_Cl_b = np.array([0.010400, 1.16620, 18.5194, 47.7784])
const_Cl_c = -9.5574

model_a_pos_cl = np.array([0.5, 0.5, 0.5])
model_b_pos_cl = np.array([0.25, 0.25, 0.25])
pos_na = np.array([0, 0, 0])

peak_nr = np.array([1,2,3,5,6,8,9,11,13])
two_theta = np.array([27.518, 31.881, 45.709, 56.808, 66.634, 75.774, 84.554, 101.934, 110.955])
hkl = np.array([(1,1,1), (2,0,0), (2,2,0), (2,2,2), (4,0,0), (2,0,4), (2,2,4), (4,0,4), (4,2,4)])
flaechenhaeuf = np.array([1,3,3,1,3,6,3,3,3])


lorfak = lorentzfaktor(two_theta)
polfak = polarisation(two_theta)

f_na = atomformfaktor(two_theta, const_Na_a, const_Na_b, const_Na_c)
f_cl = atomformfaktor(two_theta, const_Cl_a, const_Cl_b, const_Cl_c)

ampl_a = strukturamplitude(hkl, f_cl, f_na, model_a_pos_cl, pos_na)
ampl_b = strukturamplitude(hkl, f_cl, f_na, model_b_pos_cl, pos_na)

strukt_a = strukturfaktor(ampl_a)
strukt_b = strukturfaktor(ampl_b)

df = pd.DataFrame({'Peak_Nr.': peak_nr, 
                   '2Theta': two_theta, 
                   'L': lorfak, 
                   'P': polfak, 
                   'H': flaechenhaeuf, 
                   'f_Na': f_na,
                   'f_Cl': f_cl,
                   'F_hkl, A': ampl_a,
                   'F_hkl, B': ampl_b,
                   'F_a^2': strukt_a, 
                   'F_b^2': strukt_b})

latex_table = df.style.format({
    '2Theta': "{:.3f}",
    'L': "{:.2f}",
    'P': "{:.2f}",
    'H': "{:.0f}",
    'f_Na': "{:.2f}",
    'f_Cl': "{:.2f}",
    'F_hkl, A': "{:.2f}",
    'F_hkl, B': "{:.2f}",
    'F_a^2': "{:.2f}",
    'F_b^2': "{:.2f}"
    # Add more columns as needed
}).to_latex()

intensity_a = intensity(lorfak, polfak, flaechenhaeuf, strukt_a)
intensity_b = intensity(lorfak, polfak, flaechenhaeuf, strukt_b)

norm_a = norm(intensity_a, intensity_a[1])
norm_b = norm(intensity_b, intensity_b[1])

df2 = pd.DataFrame({'Peak-Nr.': peak_nr,
                    'I_A': intensity_a,
                    'I_B': intensity_b,
                    'I_{norm,A}': norm_a,
                    'I_{norm,B}': norm_b})

latex_table2 = df2.style.format({
    'I_A': "{:.1f}",
    'I_B': "{:.1f}",
    'I_{norm,A}': "{:.2f}",
    'I_{norm,B}': "{:.2f}"
}).to_latex()

print(latex_table2)

