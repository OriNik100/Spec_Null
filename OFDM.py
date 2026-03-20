import numpy as np
import matplotlib.pyplot as plt

def QPSK_modulation (bits):
    """
    Modulate the input bits using QPSK modulation.

    Parameters:
    bits (numpy array): Input binary data to be modulated.

    Returns:
    numpy array: Modulated QPSK symbols (not normalized).
    """
    # Ensure the number of bits is even
    if len(bits) % 2 != 0:
        raise ValueError("Number of bits must be even for QPSK modulation.")
    
    # Reshape the bits into pairs
    bit_pairs = bits.reshape(-1, 2)
    
    # Map the bit pairs to QPSK symbols
    symbols = np.zeros(len(bit_pairs), dtype=complex)
    
    for i, pair in enumerate(bit_pairs):
        if np.array_equal(pair, [0, 0]):
            symbols[i] = 1 + 1j  # Symbol for '00'
        elif np.array_equal(pair, [0, 1]):
            symbols[i] = -1 + 1j  # Symbol for '01'
        elif np.array_equal(pair, [1, 0]):
            symbols[i] = 1 - 1j  # Symbol for '10'
        elif np.array_equal(pair, [1, 1]):
            symbols[i] = -1 - 1j  # Symbol for '11'
    
    return symbols

def nummber_of_subcarriers(N_FFT):
    """
    Calculate the number of subcarriers for OFDM modulation.

    Parameters:
    N_FFT (int): Size of the FFT used in OFDM modulation.
    fs (float): Sampling frequency.

    Returns:
    int: Number of subcarriers.
    """
    # The number of subcarriers(data) is typically N_FFT / 2 for real-valued signals
    return N_FFT // 2

def OFDM_modulation(symbols, N_FFT, fs, null_targets, is_spectrum=False):  
    df_ofdm = fs / N_FFT # diff between bins
    ofdm_bins = [int(np.round(f / df_ofdm)) for f in null_targets]# Number of bins for given null_freq
    num_subcarriers = len(ofdm_bins) # num of null_freq
    print(f"OFDM modulation: {num_subcarriers} subcarriers at bins {ofdm_bins}")

    S_ofdm = np.zeros(N_FFT, dtype=complex) #
    for i, bin_idx in enumerate(ofdm_bins):
        S_ofdm[bin_idx] = symbols[i]

    s_ofdm = np.fft.ifft(S_ofdm)

    if is_spectrum:
        return S_ofdm
    return s_ofdm


# Example usage
if __name__ == "__main__":
    # Parameters
    N_FFT = 1024  # FFT size
    fs = 1000  # Sampling frequency in Hz
    null_targets = [100, 200, 300]  # Frequencies to be nulled in Hz

    # Generate random bits and modulate using QPSK
    num_bits = len(null_targets) * 2  # Each symbol represents 2 bits
    bits = np.random.randint(0, 2, num_bits)
    symbols = QPSK_modulation(bits)

    # Perform OFDM modulation
    s_ofdm = OFDM_modulation(symbols, N_FFT, fs, null_targets)

    # Plot the time-domain OFDM signal
    plt.figure()
    plt.plot(np.real(s_ofdm), label='Real Part')
    plt.plot(np.imag(s_ofdm), label='Imaginary Part')
    plt.title('OFDM Time-Domain Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

    S_ofdm = OFDM_modulation(symbols, N_FFT, fs, null_targets, is_spectrum=True)
    # Plot the frequency-domain OFDM signal
    plt.figure()
    plt.plot(np.abs(S_ofdm), label='Magnitude')
    plt.title('OFDM Frequency-Domain Signal')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

