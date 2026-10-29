import numpy as np


def translate(u, mu_s, mu_t):
    return u + (mu_t - mu_s)


def project(u_tilde, beta, C, idx, max_iter=50):
    """min (u−ũ)ᵀC⁻¹(u−ũ)  s.t. u[idx_k] ≤ β_k；有效集法（方法论 §5.4）。"""
    idx = np.asarray(idx)
    K = len(idx)
    A = [k for k in range(K) if u_tilde[idx[k]] > beta[k]]
    for _ in range(max_iter):
        u = u_tilde.copy()
        lam_full = np.zeros(K)
        drop = []
        if A:
            ja = idx[A]
            lam = np.linalg.solve(C[np.ix_(ja, ja)], u_tilde[ja] - beta[A])
            u = u_tilde - C[:, ja] @ lam
            lam_full[A] = lam
            drop = [A[i] for i in range(len(A)) if lam[i] < -1e-10]
        add = [k for k in range(K) if k not in A and u[idx[k]] > beta[k] + 1e-9]
        if not add and not drop:
            active = np.zeros(K, bool)
            active[A] = True
            return u, active, lam_full
        A = sorted([k for k in A if k not in drop] + add)
    raise RuntimeError("有效集法未收敛")


def truncate(u_tilde, beta, idx):
    u = u_tilde.copy()
    u[idx] = np.minimum(u_tilde[idx], beta)
    return u


def shift_raw(x_filled_src, mean_src, mean_tgt):
    return x_filled_src + (mean_tgt - mean_src)
