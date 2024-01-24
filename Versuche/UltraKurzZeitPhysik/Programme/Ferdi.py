import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.signal import windows
from scipy.fft import fft

import locale
locale.setlocale(locale.LC_NUMERIC, "de_DE")

plt.rcdefaults()
plt.rcParams['axes.formatter.use_locale'] = True
plt.rcParams.update({'font.size': 17})
# mpl.use('pgf')
plt.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "pgf.preamble": '\n'.join([ # plots will use this preamble
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{amsmath,amssymb,amstext}',
        r'\usepackage[locale=DE]{siunitx}', 
        r'\setlength{\fboxrule}{0.0pt}',#detect-all,
        ])
    })

dirPath = os.path.dirname(__file__)

# Get all file names then get names of dimensions
pathToFiles = dirPath+'\..\Daten\\'

# tHr, hrAmplitude = np.loadtxt(pathToFiles + 'HRSilizium.dat', delimiter='\t', unpack=True)
# t, refAmplitude = np.loadtxt(pathToFiles + 'Referenzmessung.dat', delimiter='\t', unpack=True)
tHr, hrAmplitude = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/HR_SIL.dat", unpack=True)
t, refAmplitude = np.loadtxt("C:/Users/toni-/OneDrive/Alt/Desktop/Uni/Master/PPD/Versuche/UltraKurzZeitPhysik/Daten/REF.dat", unpack=True)

# Set mean to zero
hrAmplitude -= hrAmplitude[0]
refAmplitude -= refAmplitude[0]

# Cut before reflex occurs
indexCutHr = np.where(tHr==8.3)[0][0]
indexCutRef = np.where(t==6.2)[0][0]

hrAmplitude = hrAmplitude[:indexCutHr]
tHr = tHr[:indexCutHr]

refAmplitude = refAmplitude[:indexCutRef]
t = t[:indexCutRef]

def window(sample_content, t):
    b = 0
    time_window = np.zeros_like(sample_content)

    for kk in range(len(sample_content)):
        b += t[2] - t[1]
        time_window[kk] = b

    a = np.where(sample_content == np.min(sample_content))[0][0]
    diff = round(2 * (abs(len(sample_content) * 0.5 - a)))

    # print(diff)
    extended_content = np.zeros(diff)
    # print(extended_content)
    extended_content = np.append(extended_content, sample_content)

    hannWindow = windows.hann(len(extended_content))
    content = extended_content* hannWindow

    content = np.append(content, np.zeros(5000))

    b = 0
    time_window_zw = time_window
    time_window = np.zeros_like(content)

    for kk in range(len(content)):
        b += time_window_zw[2] - time_window_zw[1]
        time_window[kk] = b
    
    return time_window, content
    
def FFT(window_ampl, t):
    Fs = 1 / (t[2]-t[1])
    N = len(window_ampl)
    Nfft = int(2 ** np.ceil(np.log2(N)))
    f  = Fs / 2 * np.linspace(0, 1, int(1 + Nfft / 2))

    fft_data = fft(window_ampl, Nfft)/N
    fft_data = fft_data[:int(Nfft/2)+1]

    print(len(fft_data))
    print(len(f))
    # plt.plot(f, np.abs(fft_data))
    return f, np.abs(fft_data)


t, refAmplWindow = window(refAmplitude,t)
f, fftRef = FFT(refAmplWindow, t)

t2, hrAmplWindow = window(hrAmplitude, tHr)
f, fftHr = FFT(hrAmplWindow, t2)

plt.plot(f, fftHr/fftRef, label='HR-Silizium')
plt.hlines(0.7, 0.4,2.25, linestyles='--', colors='k', label='Theorie')

plt.legend()
plt.xlabel('Frequenz $f$ in THz')
plt.ylabel('Spektrale Transmission $\\frac{|E_{HR-Si}(\omega)|}{|E_{ref}(\omega)|}$')
plt.xlim([0.4,2.25])
plt.ylim([0,1.1])
# plt.savefig(dirPath+'\..\Bilder\\422HRSiliziumTransmission.pdf', bbox_inches='tight')

plt.show()