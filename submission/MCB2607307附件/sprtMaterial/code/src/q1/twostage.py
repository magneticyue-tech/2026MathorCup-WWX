import numpy as np

from .data import grade
from .discretize import apply_cuts, apply_rules, quantile_cuts, qwk_cuts, qwk_score

NO_LOCAL = ("F_malt_all",)
C2_RULES = ("const", "round", "median", "quant", "qwk_cut", "median_knn")
M_SCALE = 3.0


def group_zeros(Z, panels, idx, dims) -> np.ndarray:
    idx = np.sort(np.asarray(idx))
    out = np.full(Z.shape, np.nan)
    for g in np.unique(panels[idx]):
        m = idx[panels[idx] == g]
        if len(m) < 2:
            continue
        v = np.round(Z[m], 9)
        mn = v.min(axis=0)
        for j in range(Z.shape[1]):
            if (v[:, j] == mn[j]).sum() >= 2 and dims[j] not in NO_LOCAL:
                out[m, j] = mn[j]
    return out


def local_scale(zeros, panels, idx, m=M_SCALE) -> np.ndarray:
    idx = np.sort(np.asarray(idx))
    out = np.full(zeros.shape[1], np.nan)
    for j in range(zeros.shape[1]):
        ok = ~np.isnan(zeros[idx, j])
        if not ok.any():
            continue
        _, first = np.unique(panels[idx][ok], return_index=True)
        r = -np.median(zeros[idx, j][ok][first])
        if r > 0:
            out[j] = m * r / (1 + r**2)
    return out


def linear_endpoints(Z, idx):
    return Z[idx].min(axis=0), Z[idx].max(axis=0)


def linear_map(z, lo, hi) -> np.ndarray:
    return 6 * (z - lo) / (hi - lo)


def mixed_reference(z, zb, sigma, lo, hi):
    L = grade(linear_map(z, lo, hi))
    use = ~np.isnan(zb) & ~np.isnan(sigma)[None, :]
    A = grade(np.where(use, sigma[None, :] * (z - np.where(use, zb, 0.0)), 0.0))
    return np.where(use, A, L), use


def c1_fold(Z, xbar, panels, dims, train_idx, test_idx, oof_z, inner_z) -> dict:
    tr, te = np.sort(train_idx), np.sort(test_idx)
    lo, hi = linear_endpoints(Z, tr)
    zb_tr = group_zeros(Z, panels, tr, dims)
    sigma = local_scale(zb_tr, panels, tr)
    R_tr, _ = mixed_reference(Z[tr], zb_tr[tr], sigma, lo, hi)

    zb_te = group_zeros(Z, panels, te, dims)
    R_te, I_te = mixed_reference(Z[te], zb_te[te], sigma, lo, hi)

    Tp, Ti = linear_map(oof_z, lo, hi), linear_map(inner_z, lo, hi)
    D = Z.shape[1]
    qwk_T = np.column_stack(
        [apply_cuts(Tp[:, d], qwk_cuts(Ti[:, d], R_tr[:, d])) for d in range(D)]
    )
    quant_T = np.column_stack(
        [apply_cuts(Tp[:, d], quantile_cuts(Ti[:, d], R_tr[:, d])) for d in range(D)]
    )
    return {
        "test_idx": te,
        "R_te": R_te,
        "I_te": I_te,
        "sigma": sigma,
        "round_T": grade(Tp),
        "qwk_T": qwk_T,
        "quant_T": quant_T,
    }


def select_c1_map(R_te, grades: dict):
    scores = {}
    for name in ("round_T", "qwk_T"):
        g = grades[name]
        per_rep = []
        for r in range(g.shape[0]):
            q = [qwk_score(R_te[r, :, d], g[r, :, d]) for d in range(g.shape[2])]
            per_rep.append(np.nanmean(q))
        scores[name] = float(np.mean(per_rep))
    best = "qwk_T" if scores["qwk_T"] > scores["round_T"] else "round_T"
    return best, scores


def c2_fold(xbar, Z, train_idx, test_idx, oof_z, inner_z) -> dict:
    tr, te = np.sort(train_idx), np.sort(test_idx)
    D = Z.shape[1]
    coef = np.array([np.polyfit(Z[tr, d], xbar[tr, d], 1) for d in range(D)])
    mu_te = oof_z * coef[:, 0] + coef[:, 1]
    mu_inner = inner_z * coef[:, 0] + coef[:, 1]
    p = np.empty((len(te), D, 7))
    grades = {rule: np.empty((len(te), D), dtype=int) for rule in C2_RULES}
    for d in range(D):
        out = apply_rules(mu_te[:, d], mu_inner[:, d], xbar[tr, d])
        p[:, d] = out["p"]
        for rule in C2_RULES:
            grades[rule][:, d] = out[rule]
    return {
        "test_idx": te,
        "mu": mu_te,
        "inner_mu": mu_inner,
        "coef": coef,
        "p": p,
        "grades": grades,
    }
