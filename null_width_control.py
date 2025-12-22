import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hlp
def matrix_inverse(A):
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

def build_z_hat(a, psi, t, null_freqs):
   
    N = len(t)
    K = len(null_freqs)
    
    z = np.zeros((N, K), dtype=complex)
    z1 = np.zeros((N, K), dtype=complex)
    z2 = np.zeros((N, K), dtype=complex)
    z_hat =np.zeros((N,3*K),dtype=complex)

    base = a * np.exp(1j*psi)     
    for k, fk in enumerate(null_freqs):
        z[:, k] = base * np.exp(-1j * 2*np.pi * fk * t) # cannot explain the minus sign
    for k, fk in enumerate(null_freqs):
        z1[:, k] = t*base * np.exp(-1j * 2*np.pi * fk * t) # cannot explain the minus sign
    for k, fk in enumerate(null_freqs):
        z2[:, k] = (t**2)*base * np.exp(-1j * 2*np.pi * fk * t) # cannot explain the minus sign
    
    z_hat = np.hstack((z,z1,z2)) 
    print(f"Shape of z_hat: {z_hat.shape}")
    return z_hat

def inner_product_mat(g, h):
        N = len(g) 
        print(f"Shape of g.T: {(g.T).shape}")   
        print(f"Shape of h: {(h).shape}")       
        result = (g.T @ h) / N      
        return result


def compute_phi_hat(a, psi, t, null_freqs):

    z_hat= build_z_hat(a, psi, t, null_freqs)
    
    c =np.real(z_hat)
    s = np.imag(z_hat)
    A = np.hstack((c, s))
    
    
    # Verify the shape
    print(f"Shape of c: {c.shape}")
    print(f"Shape of A: {A.shape}")
   
    A_inner = hlp.inner_product_mat(A,A)
    A_inner_inv = hlp.matrix_inverse(A_inner)
    N=len(t)
    ones = np.ones((N,1))
    y = hlp.inner_product_mat(np.hstack([-s,c]),ones)
    phi_hat_width = A@A_inner_inv@y
    print(f"Shape of phi_hat_width: {phi_hat_width.shape}")
    return phi_hat_width
