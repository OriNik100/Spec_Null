import numpy as np


##############################################
# Build z from a, psi, t, null_freq
##############################################

# z is a raw vector, each arg have N sampels so we expand z to be a matrix(N*K)
def build_z(a, psi, t, null_freqs):
    N = len(t)
    K = len(null_freqs)
    z = np.zeros((N, K), dtype=complex)
    base = a * np.exp(psi)     # common factor
    for k, fk in enumerate(null_freqs):
        z[:, k] = base * np.exp(-1j * 2*np.pi * fk * t) # cannot explain the minus sign
    return z


##############################################
# Create c, s from z
##############################################
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

##############################################
# Inner Product of matrices g (), h().
# gives a matrix of 
##############################################
def inner_product_mat(g, h):
        N = len(g)        
        result = (g.T @ h) / N      
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
