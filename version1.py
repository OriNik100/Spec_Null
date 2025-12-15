import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import null_depth_control as dpth

# ---- parameters ----
T = 60e-6           # duration (s)
B = 2e6             # bandwidth (Hz)
fs = 30 * B         # sampling rate- number of samples per second (10 and not two for over sampling)(1/s)
N = int(np.round(T * fs))  # (s/s - number)
t = np.linspace(0, T, N, endpoint=False) #array of time values from 0 to T spaced evenly with N points

# LFM chirp phase (baseband) # center time optional, f0 = 0# amplitude (rect), replace with window if desired
b = B/T
psi = 1j * 2*np.pi * (b/2) * t**2 
a = np.ones_like(t)
s1 = a * np.exp(psi)


# compute baseband spectrum
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

freqs, S = spectrum(s1, fs)

# z is a raw vector, each arg have N sampels so we expand z to be a matrix(N*K)
def build_z(a, psi, t, null_freqs):
    N = len(t)
    K = len(null_freqs)
    z = np.zeros((N, K), dtype=complex)
    base = a * np.exp(psi)     # common factor
    for k, fk in enumerate(null_freqs):
        z[:, k] = base * np.exp(-1j * 2*np.pi * fk * t) # cannot explain the minus sign
    return z


def create_c(z):
    c=  np.zeros((len(z),len(z[0])))
    for k_idx, fk in enumerate(z[0]):
        c[:,k_idx]=np.real(z[:,k_idx])
    return c

def create_s(z):
    s=  np.zeros((len(z),len(z[0])))
    for k_idx, fk in enumerate(z[0]):
        s[:,k_idx]=np.imag(z[:,k_idx])
    return s

def create_b(c,s):
    b_base = np.hstack([-s,c]) 

    
    return 0
               
def inner_product_mat(g, h):
        N = len(g)        
        result = (g.T @ h) / N      
        return result


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


# Example: single null at 0.4 MHz
nulls = [0.4e6]
K=len(nulls)
z = build_z(a,psi,t,nulls)
c= create_c(z)
s=create_s(z)
ones = np.ones((N,1))
A = np.hstack([c,s])
A_inner = inner_product_mat(A , A)
y = inner_product_mat(np.hstack([-s,c]),ones)

gamma = matrix_inverse(A_inner)@y

phi_hat = (A @ gamma)

s_adapted = s1 * np.exp(1j * phi_hat.flatten())

freqs2, S_adapted = spectrum(s_adapted, fs)


'''
plt.figure()
plt.plot(t,np.real(s1), label ="Real part (I)")
plt.plot(t,np.imag(s1),color ='red', label ="Imag part (Q)")
plt.xlim(0, T)
plt.xlabel('Time')
plt.ylabel("Amplitude")
plt.title('LFM')
plt.legend()
plt.grid()


plt.figure()
plt.plot(freqs/1e6, 20*np.log10(np.abs(S)/np.max(np.abs(S))))
plt.xlim(-B/1e6, B/1e6 +1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB)')
plt.title('Unadapted LFM spectrum')
plt.grid()
'''
plt.figure()
plt.plot(freqs2/1e6, 20*np.log10(np.abs(S_adapted)/np.max(np.abs(S))))
plt.xlim(-B/1e6, B/1e6 +1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB)')
plt.title('adapted LFM spectrum')
plt.grid()
'''
plt.figure()
plt.plot(t*1e6, phi_hat*180/np.pi)
plt.xlabel("Time (µs)")
plt.ylabel("Phase offset φ(t) (rad)")
plt.title("Computed φ̂(t) from equation (8)")
plt.grid()
plt.show()
'''
print(f"shape of z :{np.shape(z)}")
print(f"shape of c :{np.shape(c)}")
print(f"shape of s :{np.shape(s)}")
print(f"shape of A :{np.shape(A)}")
print(f"shape of y :{np.shape(y)}")
print(f"shape of phi_hat :{np.shape(phi_hat)}")

phi__depth_control = dpth.solve_nulling_problem(
    A=A,
    y=y,
    phi_hat = phi_hat,
    beta=10000000,
    W=None,
    M=None,
    max_iter=50
)

s_depth_control = s1 * np.exp(1j * phi__depth_control.flatten())

freqs3, S_depth_control = spectrum(s_depth_control, fs)

plt.figure()
plt.plot(freqs3/1e6, 20*np.log10(np.abs(S_depth_control)/np.max(np.abs(S))))
plt.xlim(-B/1e6, B/1e6 +1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB)')
plt.title('adapted LFM spectrum')
plt.grid()

plt.figure()
plt.plot(t*1e6, phi__depth_control*180/np.pi)
plt.xlabel("Time (µs)")
plt.ylabel("Phase offset φ(t) (rad)")
plt.title("Computed φ̂(t) from equation (8)")
plt.grid()
plt.show()
