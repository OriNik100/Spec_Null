import numpy as np
import os
import matplotlib.pyplot as plt
import helper_functions as hlp

# ---- parameters ----
T = 60e-6           # duration (s)
B = 2e6             # bandwidth (Hz)
fs = 5 * B         # sampling rate- number of samples per second(1/s)
N = int(np.round(T * fs))  # (s/s - number)
t = np.linspace(0, T, N, endpoint=False) #array of time values from 0 to T spaced evenly with N points
N_FFT = 2**14

# LFM chirp phase (baseband) # center time optional, f0 = 0# amplitude (rect), replace with window if desired
b = B/T
psi = np.pi *b* t**2
a = np.ones_like(t)
s1 = a * np.exp(1j*psi)

filename_zero = 'optimal_phasor_zero.npy'
filename_random = 'optimal_phasor_random.npy'
filename_vouras  = 'optimal_phasor_vouras.npy'

if os.path.exists(filename_zero):
    zero_phasor = np.load(filename_zero)
    random_phasor = np.load(filename_random)
    vouras_phasor = np.load(filename_vouras)
    print(f"Loaded phasors successfully.")
else:
    correction_phasor = 0
    print(f"Error: File '{filename_zero}' not found.")

s_zero = s1* zero_phasor
s_random = s1* random_phasor
s_vouras = s1* vouras_phasor

freqs_zero, S_zero = hlp.spectrum(s_zero, fs,N_FFT)
freqs_random, S_random = hlp.spectrum(s_random, fs,N_FFT)
freqs_vouras, S_vouras = hlp.spectrum(s_vouras, fs,N_FFT)

S_zero_db = 20*np.log10(np.abs(S_zero)/np.max(np.abs(S_zero) + 1e-40))
S_random_db = 20*np.log10(np.abs(S_random)/np.max(np.abs(S_random))+ 1e-40)
S_vouras_db = 20*np.log10(np.abs(S_vouras)/np.max(np.abs(S_vouras)) + 1e-40)

# גרף 1: מקורי (כחול)
plt.plot(freqs_zero/1e6, S_zero_db, label='zero', color='blue', alpha=0.3)

# גרף 2: אנליטי (ירוק)
plt.plot(freqs_random/1e6, S_random_db, label='random', color='green', linestyle='--', alpha=0.6 )

# גרף 3: אופטימלי (אדום)
plt.plot(freqs_vouras/1e6, S_vouras_db, label='vouras', color='red')

plt.title("Spectrum Comparison: zero vs vouras vs random")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Normalized Magnitude [dB]")
plt.legend()
plt.grid(True)
plt.xlim(0, B/1e6 + 0.2)
plt.ylim(-120, 5)
plt.tight_layout()
plt.show()