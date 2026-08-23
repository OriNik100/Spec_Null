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

num_subcarriers = 64

# OFDM modulation
import numpy as np

def OFDM(data, t=t, num=num_subcarriers, magnitude=1, normalize=False, os=1E6, guard_num=11, pilot_num=4, pilot_idx=None, pilot_vals=None):
    '''
    OFDM modulation of data with 'num' subcarriers.
    Returns OFDM in baseband centered around 'os'.
    '''
    # Set default pilot indices and values if not provided externally
    if pilot_idx is None:
        pilot_idx = [-21, -7, 7, 21]
    if pilot_vals is None:
        # Generic complex pilot values if none are provided
        pilot_vals = [1+1j] * pilot_num 
        
    signal = np.zeros_like(t, dtype=complex)
    
    # Dynamically calculate guard boundaries (places e.g., 6 at the lower edge and 5 at the upper edge)
    guard_bottom = int(np.ceil(guard_num / 2))
    guard_top = int(np.floor(guard_num / 2))
    
    # Create lists of indices to zero out
    guard_idx = list(range(-num//2, -num//2 + guard_bottom)) + \
                list(range(num//2 - guard_top, num//2))
    dc_idx = [0]
    
    data_counter = 0
    pilot_counter = 0
    
    # Iterate over all orthogonal indices
    for k in range(-num//2, num//2):
        
        # 1. Zero out Guard and DC subcarriers
        if (k in dc_idx) or (k in guard_idx):
            symbol = 0
            
        # 2. Insert Pilots according to the defined indices
        elif k in pilot_idx:
            symbol = pilot_vals[pilot_counter]
            pilot_counter += 1
            
        # 3. Insert Data symbols
        else:
            if data_counter < len(data):
                symbol = data[data_counter]
                data_counter += 1
            else:
                symbol = 0
                
        # Build the signal: apply the orthogonal frequency shifted by the center frequency 'os'
        # (Assuming 'T' is defined in the outer scope as in your original code)
        signal += symbol * np.exp(2 * np.pi * 1j * (os + k/T) * t)
        
    if normalize:
        # Normalize energy according to the number of active subcarriers (num - guards - dc)
        active_subcarriers = num - guard_num - 1
        signal *= 1 / np.sqrt(active_subcarriers)
        
    return magnitude * signal



def OFDM_demodulate(signal, num, T=T, t=t, os=1E6, only_active=False):
    t = np.array(t)
    signal = np.array(signal)
    
    # הבאת התדרים המדויקים מהפונקציה
    freqs = OFDM_freqs(num=num, T=T, os=os, only_active=only_active)
    symbols = []
    
    # ריצה ישירה על התדרים
    for f_k in freqs:
        basis_function = np.exp(-2 * np.pi * 1j * f_k * t)
        symbols.append((1/T) * np.trapezoid(signal * basis_function, x=t))
        
    return np.array(symbols)


import numpy as np

def OFDM_freqs(num, T=T, guard_num=11, os=1E6, only_active=False):
    '''
    Get the frequencies of the OFDM subcarriers.
    '''
    g_bot, g_top = int(np.ceil(guard_num/2)), int(np.floor(guard_num/2))
    
    # List comprehension קצר שבודק את הגבולות ואת תדר ה-DC
    k_vals = [k for k in range(-num//2, num//2) 
              if not only_active or (k != 0 and -num//2 + g_bot <= k < num//2 - g_top)]
    
    return np.array(k_vals) / T + os


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
    
    # 1. Generate EXACTLY 48 data symbols (96 bits)
    num_data_subcarriers = 48
    data_bits = np.random.randint(0, 2, num_data_subcarriers * 2)
    QPSK_data = QPSK_modulation(data_bits)
    
    # 2. Modulate the signal
    signal = OFDM(QPSK_data, t=t, num=num_subcarriers, os=1e6)
    
    # 3. Demodulate the signal (returns 64 values)
    received_full_spectrum = OFDM_demodulate(signal, num_subcarriers, t=t, os=1e6)
    
    # 4. Extract ONLY the data subcarriers from the received spectrum
    guard_bottom = 6 
    guard_top = 5
    guard_idx = list(range(-32, -32 + guard_bottom)) + list(range(32 - guard_top, 32))
    dc_idx = [0]
    pilot_idx = [-21, -7, 7, 21]
    
    k_indices = np.arange(-num_subcarriers//2, num_subcarriers//2)
    received_data_symbols = []
    
    for idx, k in enumerate(k_indices):
        # If it's a data subcarrier, save it
        if (k not in guard_idx) and (k not in dc_idx) and (k not in pilot_idx):
            received_data_symbols.append(received_full_spectrum[idx])
            
    received_data_symbols = np.array(received_data_symbols)
    
    # 5. Compare transmitted and received DATA symbols
    print("Difference magnitude:", np.abs(QPSK_data - received_data_symbols))
    is_different = list(np.abs(QPSK_data - received_data_symbols) > 0.5)
    print("Are there any errors > 0.5?", any(is_different))
    
    # 6. Corrected Plotting Loop for individual subcarriers
    plt.figure(figsize=(10, 6))
    freqs = OFDM_freqs(num=num_subcarriers, T=T, os=1e6)
    
    for k, f_k in zip(k_indices, freqs):
        # Example: plot only the data subcarriers to avoid clutter
        if (k not in guard_idx) and (k not in dc_idx) and (k not in pilot_idx):
            # Find the corresponding original QPSK symbol (simplified mapping for visual)
            subcarrier_signal = np.exp(2*np.pi*1j*f_k*t) # Using proper frequency f_k
            plt.plot(t, subcarrier_signal.real, alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Individual OFDM Subcarriers (Real Part - Correct Frequencies)')
    plt.grid(True)
    plt.show()

    # (The rest of your plotting code for signal, spectrum, etc. remains the same...)
    
    # Running the visual test function you included

    
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

def test_ofdm_visualization_full():
    # System parameters
    global T 
    T = 60e-6                 
    fs = 10e6                 
    os = 1e6                  
    num_subcarriers = 64
    
    t = np.arange(0, T, 1/fs)
    
    # Generate random QPSK data and define pilots
    qpsk_constellation = [1+1j, 1-1j, -1+1j, -1-1j]
    data = np.random.choice(qpsk_constellation, 48)
    
    pilot_idx = [-21, -7, 7, 21]
    pilot_vals = [1+1j, 1+1j, 1+1j, 1+1j]
    
    # Call the OFDM function
    signal = OFDM(data, t=t, num=num_subcarriers, os=os, pilot_idx=pilot_idx, pilot_vals=pilot_vals)
    
    # Spectral analysis
    N = len(signal)
    frequencies = fftshift(fftfreq(N, 1/fs))
    spectrum = fftshift(fft(signal))
    magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
    
    # --- Build the Detailed Visual Plot ---
    plt.figure(figsize=(13, 6)) # Slightly wider for the legend
    
    # Plot continuous spectrum
    plt.plot(frequencies / 1e6, magnitude_db, color='#1f77b4', linewidth=1.5, zorder=1)
    
    # Map frequencies by type (Data, Pilots, Guards)
    k_indices = np.arange(-num_subcarriers//2, num_subcarriers//2)
    guard_bottom = 6 
    guard_top = 5
    guard_idx = list(range(-32, -32 + guard_bottom)) + list(range(32 - guard_top, 32))
    dc_idx = [0]
    
    data_freqs, pilot_freqs, guard_freqs = [], [], []
    
    for k in k_indices:
        f_k = (os + k / T) / 1e6
        if k in dc_idx:
            continue # DC is handled via the vertical line
        elif k in guard_idx:
            guard_freqs.append(f_k)
        elif k in pilot_idx:
            pilot_freqs.append(f_k)
        else:
            data_freqs.append(f_k)
            
    # Set marker height to the peak of the spectrum for alignment
    peak_val = np.max(magnitude_db)
    
    # 1. Plot Data (Blue Circles)
    plt.scatter(data_freqs, [peak_val]*len(data_freqs), color='blue', marker='o', s=40, label='Data', zorder=3)
    
    # 2. Plot Pilots (Green Stars)
    plt.scatter(pilot_freqs, [peak_val]*len(pilot_freqs), color='lime', marker='*', s=200, edgecolor='black', label='Pilots', zorder=4)
    
    # 3. Plot Guards (Red X marks)
    plt.scatter(guard_freqs, [peak_val]*len(guard_freqs), color='red', marker='X', s=80, label='Guards (Nulls)', zorder=4)
    
    # 4. Plot DC (Red Dashed Line)
    plt.axvline(os / 1e6, color='red', linestyle='--', linewidth=2, label='DC (Null)', zorder=2)
    
    # General Styling
    plt.title('Complete OFDM Spectrum - Data, Pilots & Guards', fontsize=14)
    plt.xlabel('Frequency [MHz]', fontsize=12)
    plt.ylabel('Magnitude [dB]', fontsize=12)
    
    plt.xlim((os / 1e6) - 1.2, (os / 1e6) + 1.2)
    plt.ylim(np.min(magnitude_db), peak_val + 5)
    
    plt.grid(True, alpha=0.3)
    
    # Arrange the legend neatly at the bottom
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4) 
    plt.tight_layout()
    plt.show()

# הרצת הטסט
test_ofdm_visualization_full()



