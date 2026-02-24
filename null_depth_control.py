import numpy as np

def build_D(N):
    D = np.zeros((N - 1, N))
    for i in range(N - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    return D

def solve_nulling_problem(A, b, phi_hat, beta=1e4, W=None, M=None, t=None, max_iter=50):
    """
    Solve conjugate-gradient exactly as MATLAB code:
    - A: (N x 2K)
    - b: (2K x 1) already computed via trapz (hlp.inner_product_mat(..., t))
    - phi0: (N x 1) initial phi (LS solution)
    - returns phi (N x 1)
    """
    N, ncols = A.shape
    if M is None:
        M = np.eye(N)
    if W is None:
        W = 4000 * np.eye(ncols)

    D = build_D(N)

    # initial residual and gradient as in MATLAB
    phi = phi_hat.copy().reshape(-1, 1)   # (N x 1)
    res = - A.T @ phi + b              # (2K x 1)
    g_old = A @ res - beta * (D.T @ (D @ phi))

    d = M @ g_old
    for iter in range(max_iter):
        # compute new gradient
        g_new = A @ res - beta * (D.T @ (D @ phi))

        if iter == 0:
            d = M @ g_new
        else:
            num = (g_new.T @ M @ g_new)
            den = (g_old.T @ M @ g_old)
            gam = num / den
            d = (M @ g_new) + gam * d

        q = A.T @ d
        denom = (q.T @ W @ q) + beta * (d.T @ (D.T @ (D @ d)))
        Alpha = (d.T @ g_new) / denom

        phi = phi + Alpha * d
        res = res - Alpha * q

        g_old = g_new

    return phi