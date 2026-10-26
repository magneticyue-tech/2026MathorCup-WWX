"""Nested comparison of discretisation rules and sample weighting on the additive-adjusted target.

Seeds: outer GroupKFold shuffle random_state=rep (0,1,2); inner GroupKFold(4) shuffle random_state=100+rep;
HGB random_state=0.
"""

import os
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
from rawlib import additive
from exp_targets import N_REP, X, beers, dims, fit_pred, grp, nb, r

warnings.filterwarnings("ignore")
grade = lambda v: np.floor(np.clip(v, 0, 6) + 0.5).astype(int)


def cut(v, c):
    return np.searchsorted(np.sort(c), v, side="right")


def fit_cuts(p, y_grade, n_pass=3):
    """Coordinate search for 6 cutpoints maximising QWK, started from 0.5..5.5."""
    c = np.arange(0.5, 6.0, 1.0)
    best = cohen_kappa_score(y_grade, cut(p, c), weights="quadratic", labels=range(7))
    grid_all = np.unique(np.r_[p, c])
    for _ in range(n_pass):
        for i in range(6):
            lo = c[i - 1] if i > 0 else -np.inf
            hi = c[i + 1] if i < 5 else np.inf
            for v in grid_all[(grid_all > lo) & (grid_all < hi)][:: max(1, len(grid_all) // 120)]:
                t = c.copy(); t[i] = v
                s = cohen_kappa_score(y_grade, cut(p, t), weights="quadratic", labels=range(7))
                if s > best:
                    best, c = s, t
    return c


def quantile_cuts(p, y_grade):
    """Cutpoints so that predicted grade frequencies equal training grade frequencies."""
    cum = np.cumsum(np.bincount(y_grade, minlength=7))[:-1] / len(y_grade)
    return np.quantile(p, np.clip(cum, 0, 1))


def target(ratings, idx):
    alpha, _ = additive(ratings, dims, n_iter=150)
    return alpha.loc[beers[idx]].values


def one_fold(rep, tr, te):
    y_tr = target(r[r.beer_name.isin(set(beers[tr]))], tr)
    resid_var = None
    # inner OOF predictions on the training set
    inner = np.zeros((len(tr), len(dims)))
    for itr, ite in GroupKFold(4, shuffle=True, random_state=100 + rep).split(tr, groups=grp[tr]):
        y_in = target(r[r.beer_name.isin(set(beers[tr][itr]))], tr[itr])
        for j in range(len(dims)):
            y = np.full(len(beers), np.nan); y[tr[itr]] = y_in[:, j]
            inner[ite, j] = fit_pred(y, tr[itr], tr[ite])
    out = {k: np.zeros((len(te), len(dims))) for k in ["round", "qwk_cut", "quant", "round_w"]}
    for j in range(len(dims)):
        y = np.full(len(beers), np.nan); y[tr] = y_tr[:, j]
        p = fit_pred(y, tr, te)
        g_tr = grade(y_tr[:, j])
        out["round"][:, j] = grade(p)
        out["qwk_cut"][:, j] = cut(p, fit_cuts(inner[:, j], g_tr))
        out["quant"][:, j] = cut(p, quantile_cuts(inner[:, j], g_tr))
        # precision weights: w = n/(n+k), k = residual rating var / unexplained beer var
        tau2 = max(np.var(y_tr[:, j] - inner[:, j]) - 0, 1e-6)
        sig2 = r[r.beer_name.isin(set(beers[tr]))][dims[j]].var() * 0.75
        w = nb[tr] / (nb[tr] + sig2 / tau2)
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y_tr[:, j], sample_weight=w)
        out["round_w"][:, j] = grade(m.predict(X[te]))
    return rep, te, out


if __name__ == "__main__":
    jobs = [
        (rep, tr, te)
        for rep in range(N_REP)
        for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=grp)
    ]
    res = Parallel(n_jobs=15)(delayed(one_fold)(*j) for j in jobs)
    keys = ["round", "qwk_cut", "quant", "round_w"]
    P = {k: np.zeros((N_REP, len(beers), len(dims)), int) for k in keys}
    for rep, te, out in res:
        for k in keys:
            P[k][rep, te] = out[k]
    np.savez(os.path.join(OUT, "exp_discretize_oof.npz"), **P)
    print("saved")
