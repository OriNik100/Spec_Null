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

num_subcarriers = 4

# OFDM modulation

def OFDM(data, t=t, num= num_subcarriers, magnitude=1, normalize=False):
    '''
    OFDM modulation of data with 'num' subcarriers.
    Retrurn OFDM in baseband.
    '''
    signal = np.zeros_like(t, dtype=complex)
    for i in range(num):
        signal += np.array(data[i] *np.exp(2*np.pi*1j*i*t/T))
                    
    if normalize:
        signal *= 1/np.sqrt(num)
    return magnitude * signal

def OFDM_demodulate (signal, num, T=T, t=t):
    t = np.array(t)
    signal = np.array(signal)
    symbols = []
    for i in range(num):
        symbols.append((1/T)*np.trapezoid(signal * np.exp(-2*np.pi*1j*i*t/T), x=t))
    return np.array(symbols).flatten()


def OFDM_freqs(num= num_subcarriers, T=T):
    '''
    Get the frequencies of the OFDM subcarriers.
    '''
    return np.arange(num) / T

def QPSK_modulation(bits):
    """ Modulate the input bits using QPSK modulation. """
    if len(bits) % 2 != 0:
        raise ValueError("Number of bits must be even for QPSK modulation.")
    
    # bits reshape
    bit_pairs = bits.reshape(-1, 2)

    # bits to QPSK symbols conversion
    symbols = np.zeros(len(bit_pairs), dtype=complex)
    for i, pair in enumerate(bit_pairs):
        if np.array_equal(pair, [0, 0]):
            symbols[i] = 1 + 1j
        elif np.array_equal(pair, [0, 1]):
            symbols[i] = 1 - 1j
        elif np.array_equal(pair, [1, 0]):
            symbols[i] = -1 + 1j
        elif np.array_equal(pair, [1, 1]):
            symbols[i] = -1 - 1j
    return symbols/np.sqrt(2) # return normalized bits

if __name__ == "__main__":
    
    data = np.random.randint(0, 2, num_subcarriers * 2)
    QPSK_data = QPSK_modulation(data)
    signal = OFDM(QPSK_data)
    
    print("The sent symbols: ", QPSK_data)
    

    
    plt.figure(figsize=(10, 6))
    
    # נייצר ונצייר כל תת-נושא בנפרד
    for i in range(num_subcarriers):
        # יצירת גל הנושא הבודד (בדיוק כמו בתוך פונקציית ה-OFDM)
        subcarrier_signal = np.array(QPSK_data[i] * np.exp(2*np.pi*1j*i*t/T))
        
        # אנחנו נצייר רק את החלק הממשי (Real) כדי שהגרף יהיה קריא
        plt.plot(t, subcarrier_signal.real)

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Individual OFDM Subcarriers (Real Part)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    
    for i in range(num_subcarriers):
        # 1. יצירת גל הנושא
        subcarrier_signal = np.array(QPSK_data[i] * np.exp(2*np.pi*1j*i*t/T))
        
        # 2. חישוב הספקטרום בעזרת פונקציית העזר שלך
        freqs_sub, S_sub = hlp.spectrum(subcarrier_signal, fs, 2**14)
        
        # 3. ציור בסקאלה *לינארית* (ללא Log10) כדי לראות את החיתוך באפס
        plt.plot(freqs_sub/1e6, np.abs(S_sub))

    # נגביל את ציר ה-X כדי להתמקד רק באזור של 4 תתי-הנושאים שלנו
    # התדרים הם בכפולות של 1/T
    plt.xlim(-2 / (T * 1e6), (num_subcarriers + 1) / (T * 1e6)) 
    
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Magnitude (Linear)')
    plt.title('OFDM Orthogonality in Frequency Domain (Sinc Pulses)')
    plt.grid(True)
    plt.show()
    # ---------------------------------------------------------
    


    received_symbols = OFDM_demodulate(signal, num_subcarriers)
    #print("Sent symbols:", QPSK_data)
    #print("Received symbols:", received_symbols)
    is_different = list(np.abs(QPSK_data - received_symbols) > 0.5)
    print("Difference:", is_different)
    
    # QPSK to bits
    received_bits = []
    for symbol in received_symbols:
        received_bits.append(0 if symbol.real > 0 else 1)
        received_bits.append(0 if symbol.imag > 0 else 1)
    print("The sent bits: ", data)
    print("Received bits: ", np.array(received_bits)) 

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
    plt.plot(frequencies/1e6, 20*np.log10(np.abs(SIGNAL)/np.max(np.abs(SIGNAL))+1e-40))
    #plt.xlim(-0.5, 0.5) # מציגים רק בין -500 ל-500 קילו-הרץ
    #plt.ylim(-60, 5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('OFDM Frequency Spectrum')
    plt.grid(True)
    plt.show()

    
    '''
    
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

    s_isac = s_adapted + signal
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




    '''



