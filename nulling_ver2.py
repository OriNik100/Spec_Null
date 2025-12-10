import numpy as np

def build_D(N):
    """Build D matrix of size (N-1) x N for first-order differences."""
    D = np.zeros((N - 1, N))
    for i in range(N - 1):
        D[i, i] = 1
        D[i, i + 1] = -1
    return D


def solve_nulling_problem(A,y, phi_hat, beta=10000, W=None, M=None, max_iter=50):
    # Get the A matrix (N x 2K)
    N, n = A.shape  # N = time samples, n = 2K (coefficients)
    
    D = build_D(N)  # shape ((n-1) x n)
    W=4000*np.eye(n)    #(2Kx2K)
    M=np.eye(N)   #(1200 x 1200)
    f0=phi_hat
    Dt = D.T
    DH = D.conj().T
    res_norms = []
    K = len(A[1])
        
    f= f0
    # initial residual r = y - A f0
    r = y - A.T @ f
        
    # initial gradient
    AH = A.conj().T
    g_new = AH.T @ W @ r - beta * (Dt @ (D @ f))#(1200x1)
    d = M @ g_new   #(1200x1)
    dH = d.conj().T
    g_old = None
    for i in range(max_iter):
        q = A.T @ d #(2K x 1)
        qH = q.conj().T
        qWq = qH @ W @ q # (1x1)
        secTerm = beta*(dH @ DH @ D @ d)#(1x1)
        denom = qWq + secTerm
        num = dH @ g_new
        alpha = num / denom
        f = f + alpha * d
        r = r - alpha * q
        g_old = g_new
        g_oldH = g_old.conj().T
        g_new = AH.T @ W @ r - beta * (Dt @ (D @ f))
        g_newH = g_new.conj().T
        
        
        gamma = (g_newH @ M @ g_new) / (g_oldH @ M @ g_old)
        d = M @ g_new + gamma * d  

    

    return f