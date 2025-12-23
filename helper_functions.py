import numpy as np


##############################################
# Build z from a, psi, t, null_freq
##############################################

# z is a raw vector, each arg have N sampels so we expand z to be a matrix(N*K)
def build_z(a, psi, t, null_freqs):
    N = len(t)
    K = len(null_freqs)
    z = np.zeros((N, K), dtype=complex)
    base = a * np.exp(1j*psi)     # common factor
    for k, fk in enumerate(null_freqs):
        z[:, k] = base * np.exp(-1j * 2*np.pi * fk * t) 
    return z


##############################################
# Create c, s from z
##############################################
def create_c(z):
    c = np.zeros((len(z),len(z[0])))
    for k_idx, fk in enumerate(z[0]):
        c[:,k_idx]=np.real(z[:,k_idx])
    return c

def create_s(z):
    s=  np.zeros((len(z),len(z[0])))
    for k_idx, fk in enumerate(z[0]):
        s[:,k_idx]=np.imag(z[:,k_idx])
    return s

##############################################
# Inner Product of matrices g (), h().
# gives a matrix of 
##############################################
def inner_product_mat(g, h):
        N = len(g)        
        result = (g.T @ h) / N
        # dropped the 1/N factor just so we wil be consistent
        # restored the 1/N fzctor to be consistent with the article   
        return result

##############################################
# Matrix inversion
##############################################

def matrix_inverse(A):
    """
    Compute the inverse of a matrix (power of -1).
    
    Parameters:
    -----------
    A : numpy.ndarray or array-like
        Input square matrix
    
    Returns:
    --------
    numpy.ndarray
        Inverse of the matrix A (A^-1)
    
    Raises:
    -------
    LinAlgError
        If the matrix is singular (not invertible)
    """
    A = np.array(A)
    
    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square. Got shape {A.shape}")
    
    # Compute the inverse
    try:
        A_inv = np.linalg.inv(A)
        return A_inv
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Matrix is singular and cannot be inverted")

##########################################
# Calculate the baseband spectrum of a signal
# Returns the relevant frequencies and the Frequency-Domain signal
##########################################

def spectrum(x, fs):
    """
    Steps:
    1. np.fft.fft(x) computes the FFT (frequency content).
    2. np.fft.fftshift(...) reorders the FFT so that 0 Hz is centered,
       with negative frequencies on the left and positive on the right.
    3. np.fft.fftfreq(len(x), 1/fs) generates the frequency axis in Hz.
       - len(x) = number of samples
       - 1/fs   = time spacing between samples
       - This gives the actual frequency values for each FFT bin.
    4. np.fft.fftshift(...) is applied again to align the frequency axis
       with the shifted FFT output.

    Returns:
    freqs : frequency axis in Hz (from -fs/2 to +fs/2)
    X     : FFT of the signal, aligned with freqs
    """
    X = np.fft.fft(x)
    X = np.fft.fftshift(X)
    freqs=np.fft.fftfreq(len(x), 1/fs)
    freqs = np.fft.fftshift(freqs)
    return freqs, X

def apply_matched_filter(adapted, reference):
    """
    מממש את מוצא ה-Matched Filter כפי שמתואר במאמר (איור 8).
    
    Parameters:
    adapted_signal: האות לאחר הוספת ה-nulls (s_adapted)
    reference_signal: האות המקורי ללא ה-nulls (s1)
    
    Returns:
    mf_output: מוצא המסנן בערכים לוגריתמיים (dB) מנורמלים
    """
    #conjugating the reference to create the MF kernel
    mf_kernel = np.conj(reference[::-1])
    
    # Convolution implementation
    output = np.convolve(adapted, mf_kernel, mode='same')
    
    # dB normalization
    abs_output = np.abs(output)
    mf_output_db = 20 * np.log10(abs_output / np.max(abs_output))
    
    return mf_output_db

