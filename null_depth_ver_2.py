import numpy as np


def build_D(N):
    """Build D matrix of size (N-1) x N for first-order differences."""
    D = np.zeros((N - 1, N))
    for i in range(N - 1):
        D[i, i] = 1
        D[i, i + 1] = -1
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
    # A is (N x n)
    N, n = A.shape
    A=A.T

    # Regularization matrix should act on f (length n)
    D = build_D(n)           # (n-1 x n)
    DH = D.conj().T          # (n x n-1)

    # Preconditioner
    if M is None:
        M = np.eye(n)

    # Initial guess (same size as coefficients)
    f = np.array(phi_hat, dtype=np.float64)

    # A^H
    AH = A.conj()

    # initial residual r = y - A f
    r = y - A @ f

    # initial gradient
    g_new = AH @ apply_W(W, r) - beta * (DH @ (D @ f))

    # initial direction
    d = apply_M(M, g_new)

    g_old = None

    for i in range(max_iter):

        q = A @ d  # (N,)
        qWq = np.vdot(q, apply_W(W, q))

        Dd = D @ d
        secTerm = beta * np.vdot(Dd, Dd)

        denom = qWq + secTerm
        if abs(denom) < 1e-30:
            break

        num = np.vdot(d, g_new)
        alpha = num / denom

        f = f + alpha * d
        r = r - alpha * q

        g_old = g_new
        g_new = AH @ apply_W(W, r) - beta * (DH @ (D @ f))

        denom_gamma = np.vdot(g_old, apply_M(M, g_old))
        if abs(denom_gamma) < 1e-30:
            gamma = 0
        else:
            gamma = np.vdot(g_new, apply_M(M, g_new)) / denom_gamma

        d = apply_M(M, g_new) + gamma * d

    return f
