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

null_freqs = [0.2e6, 0.3e6, 0.4e6]


# LFM chirp phase (baseband) # center time optional, f0 = 0# amplitude (rect), replace with window if desired
b = B/T
psi = np.pi *b* t**2
a = np.ones_like(t)
s1 = a * np.exp(1j*psi)

filename_zero = 'optimal_phasor_zero.npy'
filename_random = 'optimal_phasor_random.npy'
filename_vouras  = 'optimal_phasor.npy'


if os.path.exists(filename_zero):
    zero_phasor = np.load(filename_zero)
    random_phasor = np.load(filename_random)
    vouras_phasor = np.load(filename_vouras)
    print(f"Loaded phasors successfully.")

else:
    correction_phasor = 0
    print(f"Error: File '{filename_zero}' not found.")   
    print("calculated phases in degrees.")

from version1 import getphi
phi_hat = getphi(null_freqs).flatten()
analytical_phasor = np.exp(1j * phi_hat)

zero_phase = np.angle(zero_phasor)
phi_zero_deg = np.degrees(zero_phase)
random_phase = np.angle(random_phasor)
phi_random_deg = np.degrees(random_phase)
vouras_phase = np.angle(vouras_phasor)
phi_vouras_deg = np.degrees(vouras_phase)

plt.figure(figsize=(12, 6))
# plt.plot(t*1e6, phi_zero_deg, label='Zero Phase', color='blue')
# plt.plot(t*1e6, phi_random_deg, label='Random Phase', color='green')
plt.plot(t*1e6, phi_vouras_deg, label='Optimal (Vouras) Phase', color='red')
plt.plot(t*1e6, np.degrees(phi_hat), label='Analytical Phase', color='blue', alpha=0.8, linestyle='--')
plt.title('Optimal Phase Comparison')
plt.xlabel('Time (µs)')
plt.ylabel('Phase (degrees)')
plt.legend()
plt.grid()

s_zero = s1* zero_phasor
s_random = s1* random_phasor
s_vouras = s1* vouras_phasor
s_analytical = s1* analytical_phasor

freqs_zero, S_zero = hlp.spectrum(s_zero, fs,N_FFT)
freqs_random, S_random = hlp.spectrum(s_random, fs,N_FFT)
freqs_vouras, S_vouras = hlp.spectrum(s_vouras, fs,N_FFT)
freqs_analytical, S_analytical = hlp.spectrum(s_analytical, fs,N_FFT)

S_zero_db = 20*np.log10(np.abs(S_zero)/np.max(np.abs(S_zero) + 1e-40))
S_random_db = 20*np.log10(np.abs(S_random)/np.max(np.abs(S_random))+ 1e-40)
S_vouras_db = 20*np.log10(np.abs(S_vouras)/np.max(np.abs(S_vouras)) + 1e-40)
S_analytical_db = 20*np.log10(np.abs(S_analytical)/np.max(np.abs(S_analytical)) + 1e-40)


plt.figure(figsize=(12, 6))
# plt.plot(freqs_zero/1e6, S_zero_db, label='zero', color='blue', alpha=0.3)
# plt.plot(freqs_random/1e6, S_random_db, label='random', color='green', linestyle='--', alpha=0.6 )
plt.plot(freqs_vouras/1e6, S_vouras_db, label='optimal (Vouras)', color='red')
plt.plot(freqs_analytical/1e6, S_analytical_db, label='analytical', color='blue', alpha=0.8, linestyle='--')

plt.title("Spectrum Comparison")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Normalized Magnitude [dB]")
plt.legend()
plt.grid(True)
# plt.xlim(0, B/1e6 + 0.2)
# plt.ylim(-120, 5)
plt.tight_layout()
plt.show()