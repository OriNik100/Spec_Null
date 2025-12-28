import numpy as np
import helper_functions as hlp

def build_D(N):
    """Build D matrix of size (N-1) x N for first-order differences."""
    D = np.zeros((N - 1, N))
    for i in range(N - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    return D


def apply_W(W, v):
    if W is None:
        return v
    W_arr = np.asarray(W)
    if W_arr.ndim == 0:
        return W_arr * v
    elif W_arr.ndim == 1:
        return W_arr * v
    else:
        return W_arr @ v

def apply_M(M, v):
    if M is None:
        return v
    M_arr = np.asarray(M)
    if M_arr.ndim == 0:
        return M_arr * v
    else:
        return M_arr @ v


def solve_nulling_problem(A, y, phi_hat, beta=10000, W=None, M=None, max_iter=50):
    print("shape of ")
    # Get the A matrix (N x 2K)
    N, n = A.shape  # N = time samples, n = 2K (coefficients)

    D = build_D(N)  # shape ((n-1) x n)

    if M is None:
        M = np.eye(N)  # (1200 x 1200)
        
    f0 = phi_hat
    f = f0
    
    DH = D.conj().T
    K = len(A[1])

    # initial residual r = y - A f0
    # We multiply and divide by N (in 3 places) to ensure same "units"
    # r = y - A.T @ f0
    r = y - hlp.inner_product_mat(A, f0) # 2x1

    # initial gradient
    AH = A.conj().T
    g_new = N*(AH.T @ apply_W(W, r)) - beta * (DH @ (D @ f0))  # (1200x1) 
    d = + M @ g_new  # (1200x1)
    dH = d.conj().T
    g_old = None
    for i in range(max_iter):
        q = A.T @ d  # (2K x 1)
        qH = q.conj().T
        qWq = qH @ apply_W(W, q)  # (1x1)
        secTerm = beta * (dH @ (DH @ (D @ d)))  # (1x1)
        denom = qWq + secTerm
        num = dH @ g_new
        alpha = num / denom
        f = f + alpha * d
        r = r - (alpha * q)/N
        g_old = g_new
        g_oldH = g_old.conj().T
        g_new = N*(AH.T @ apply_W(W, r)) - beta * (DH @ (D @ f))
        g_newH = g_new.conj().T

        gamma = ((g_new.T @ apply_M(M, g_new))) / (g_old.T @ apply_M(M, g_old))
        d = M @ g_new + gamma * d

    return f