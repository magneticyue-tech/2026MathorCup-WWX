"""One-stage (chem -> raw panel mean) vs two-stage (chem -> attachment-4 z -> 0-6), scored on raw truth.

Two-stage mappings follow q1_methodology.md §4.5-4.6 (attachment-only): linear endpoint T_d, local-zero
reference A with m=3, mixed reference R, QWK cutpoints fitted on inner OOF predictions against R.
Seeds: outer GroupKFold shuffle random_state=rep (0,1,2); inner GroupKFold(4) shuffle random_state=100+rep;
HGB random_state=0.
"""

import os
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
from exp_targets import N_REP, X, Z, beers, dims, fit_pred, grp, xbar

warnings.filterwarnings("ignore")
ZV, XB, ND = Z.values, xbar.values, len(dims)
ZR = np.round(ZV, 9)
NO_LOCAL = {"F_malt_all"}
W = (np.arange(7)[:, None] - np.arange(7)[None]) ** 2


def grade(v):
    return np.floor(np.clip(v, 0, 6) + 0.5).astype(int)


def qwk(a, b):
    o = np.bincount(7 * a + b, minlength=49).reshape(7, 7).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    den = (W * e).sum()
    return 1 - (W * o).sum() / den if den > 0 else np.nan


def cut(v, c):
    return np.searchsorted(c, v, side="right")


def fit_cuts(p, y, n_pass=3):
    c = np.arange(0.5, 6.0, 1.0)
    score = lambda t: np.nan_to_num(qwk(y, cut(p, t)), nan=-1.0)
    best = score(c)
    grid = np.unique(p)
    grid = grid[:: max(1, len(grid) // 120)]
    for _ in range(n_pass):
        for i in range(6):
            lo = c[i - 1] if i else -np.inf
            hi = c[i + 1] if i < 5 else np.inf
            for v in grid[(grid > lo) & (grid < hi)]:
                t = c.copy()
                t[i] = v
                s = score(t)
                if s > best:
                    best, c = s, t
    return c


def quantile_cuts(p, y):
    cum = np.cumsum(np.bincount(y, minlength=7))[:-1] / len(y)
    c = np.quantile(p, np.clip(cum, 0, 1))
    c[cum >= 1 - 1e-12] = np.inf
    return c


def infer_panels(k=3):
    parent = np.arange(len(ZR))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(ZR)):
        for j in np.where((ZR[i + 1:] == ZR[i]).sum(1) >= k)[0] + i + 1:
            parent[find(j)] = find(i)
    return np.array([find(i) for i in range(len(ZR))])


PANEL = infer_panels()


def group_zeros(idx):
    """Per-beer panel zero from members in idx only (min z tied >= 2 times); NaN if none."""
    out = np.full((len(beers), ND), np.nan)
    for g in np.unique(PANEL[idx]):
        m = idx[PANEL[idx] == g]
        if len(m) < 2:
            continue
        v = ZR[m]
        mn = v.min(0)
        for j in np.where((v == mn).sum(0) >= 2)[0]:
            if dims[j] not in NO_LOCAL:
                out[m, j] = mn[j]
    return out


def local_scale(idx, zb, m=3.0):
    s = np.full(ND, np.nan)
    for j in range(ND):
        ok = ~np.isnan(zb[idx, j])
        if ok.any():
            _, first = np.unique(PANEL[idx][ok], return_index=True)
            r = -np.median(zb[idx, j][ok][first])
            if r > 0:
                s[j] = m * r / (1 + r**2)
    return s


def reference(z, zb, sig, T):
    L = grade(T(z))
    A = grade(sig * (z - zb))
    use = ~np.isnan(zb) & ~np.isnan(sig)[None]
    return np.where(use, A, L), use


def one_fold(rep, tr, te):
    in_z = np.zeros((len(tr), ND))
    in_r = np.zeros((len(tr), ND))
    for itr, ite in GroupKFold(4, shuffle=True, random_state=100 + rep).split(tr, groups=grp[tr]):
        for j in range(ND):
            in_z[ite, j] = fit_pred(ZV[:, j], tr[itr], tr[ite])
            in_r[ite, j] = fit_pred(XB[:, j], tr[itr], tr[ite])
    pz = np.column_stack([fit_pred(ZV[:, j], tr, te) for j in range(ND)])
    pr = np.column_stack([fit_pred(XB[:, j], tr, te) for j in range(ND)])
    lo, hi = ZV[tr].min(0), ZV[tr].max(0)
    T = lambda v: 6 * (v - lo) / (hi - lo)
    zb_tr, zb_te = group_zeros(tr), group_zeros(te)
    sig = local_scale(tr, zb_tr)
    R_tr, _ = reference(ZV[tr], zb_tr[tr], sig, T)
    R_te, use_te = reference(ZV[te], zb_te[te], sig, T)
    coef = np.array([np.polyfit(ZV[tr, j], XB[tr, j], 1) for j in range(ND)])
    M = lambda v: v * coef[:, 0] + coef[:, 1]
    G_tr = grade(XB[tr])
    cont = {"2SA": (T(pz), T(in_z), R_tr), "2SR": (M(pz), M(in_z), G_tr), "1S": (pr, in_r, G_tr)}
    out = {"R_te": R_te, "use_te": use_te}
    for name, (c_te, c_in, ref) in cont.items():
        out[f"{name}_cont"] = c_te
        out[f"{name}_round"] = grade(c_te)
        out[f"{name}_quant"] = np.column_stack(
            [cut(c_te[:, j], quantile_cuts(c_in[:, j], ref[:, j])) for j in range(ND)])
        out[f"{name}_qwk"] = np.column_stack(
            [cut(c_te[:, j], fit_cuts(c_in[:, j], ref[:, j])) for j in range(ND)])
    return rep, te, out


if __name__ == "__main__":
    multi = [g for g in np.unique(PANEL) if (PANEL == g).sum() > 1]
    print("inferred panels: multi-beer groups", len(multi), "covering", sum((PANEL == g).sum() for g in multi))
    jobs = [
        (rep, tr, te)
        for rep in range(N_REP)
        for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=grp)
    ]
    res = Parallel(n_jobs=8)(delayed(one_fold)(*j) for j in jobs)
    keys = list(res[0][2].keys())
    P = {k: np.zeros((N_REP, len(beers), ND), dtype=res[0][2][k].dtype) for k in keys}
    for rep, te, out in res:
        for k in keys:
            P[k][rep, te] = out[k]
    np.savez(os.path.join(OUT, "exp_twostage_oof.npz"), panel=PANEL, **P)
    print("saved")
