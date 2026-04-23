import numpy as np
from matplotlib import pyplot as plt
import helper_functions as hlp

# Parameters

T = 60e-6
B = 2e6
fs = 5 * B
N = int(np.round(T * fs))
t = np.linspace(0, T, N, endpoint=False)

np.random.seed(42)

# OFDM modulation

def OFDM(data, t=t, num=16, magnitude=1, normalize=False):
    '''
    OFDM modulation of data with 'num' subcarriers.
    Retrurn OFDM in baseband.
    '''
    signal = np.zeros_like(t, dtype=complex)
    for i in range(num):
        signal += np.array(data[i] *np.exp(2*np.pi*1j*i*t/T))

    if normalize:
        signal *= 1/num
    return magnitude * signal


def OFDM_freqs(num=16, T=T):
    '''
    Get the frequencies of the OFDM subcarriers.
    '''
    return np.arange(num) / T


if __name__ == "__main__":
    data = np.random.randint(0, 2, 32)
    QPSK_data = 2*data - 1 + 1j*(2*data - 1)
    signal = OFDM(QPSK_data)

    plt.figure()
    plt.plot(t, signal.real, label='Real part', color='blue')
    plt.plot(t, signal.imag, label='Imaginary part', color='red')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('OFDM Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

    # frequency spectrum
    frequencies, SIGNAL = hlp.spectrum(signal, fs, 1024)

    plt.figure()
    plt.plot(frequencies, np.abs(SIGNAL))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('OFDM Frequency Spectrum')
    plt.grid(True)
    plt.show()

    # Nulled LFM and OFDM insertion

    from version1 import getphi

    b = B/T
    psi = np.pi *b* t**2
    a = np.ones_like(t)
    s1 = a * np.exp(1j*psi)
    freqs, S = hlp.spectrum(s1, fs, 2**14)

    nulls = OFDM_freqs(16, T)
    phi_hat = getphi(nulls)
    s_adapted = a * np.exp(1j*psi + 1j * phi_hat.flatten())
    freqs2, S_adapted = hlp.spectrum(s_adapted, fs, 2**14)

    s_isac = s_adapted + 0.07* signal
    freqs_isac, S_isac = hlp.spectrum(s_isac, fs, 2**14)



    # Plotting the spectra of the original and adapted signals
    plt.figure()
    plt.plot(freqs2/1e6, 20*np.log10(np.abs(S_adapted)/np.max(np.abs(S_adapted))))
    plt.plot(freqs/1e6, 20*np.log10(np.abs(S)/np.max(np.abs(S))), color ='red')
    plt.xlim(-B/1e6-3, B/1e6 +3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.title('Adapted vs Unadapted LFM spectrum')
    plt.grid()

    plt.figure()
    plt.plot(freqs_isac/1e6, 20*np.log10(np.abs(S_isac)/np.max(np.abs(S_isac))))
    plt.xlim(-B/1e6-3, B/1e6 +3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.title('ISAC Spectrum')
    plt.grid()
    plt.show()

    # plotting both unadapted and adapted spectra, along with the ISAC spectra
    plt.figure()
    plt.plot(freqs/1e6, 20*np.log10(np.abs(S)/np.max(np.abs(S))), label='Unadapted LFM', color='red')
    plt.plot(freqs2/1e6, 20*np.log10(np.abs(S_adapted)/np.max(np.abs(S_adapted))), label='Adapted LFM', color='blue')
    plt.plot(freqs_isac/1e6, 20*np.log10(np.abs(S_isac)/np.max(np.abs(S_isac))), label='ISAC', color='green')
    plt.xlim(-B/1e6-3, B/1e6 +3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.title('Comparison of Spectra')
    plt.legend()
    plt.grid()
    plt.show()








