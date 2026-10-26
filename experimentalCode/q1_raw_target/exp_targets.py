"""Compare Q1 training targets on the raw 0-6 scale.

Seeds: GroupKFold shuffle random_state=rep (0,1,2); HGB random_state=0.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
from rawlib import additive, load, session_groups

warnings.filterwarnings("ignore")
r, a4, c1, dims = load()
feat = [c for c in c1.columns if c not in ("beer", "beer_id", "tasting_category_fine")]
X = c1[feat].values.astype(float)
beers = a4.index.values
xbar = r.groupby("beer_name")[dims].mean().loc[beers]
nb = r.groupby("beer_name").size().loc[beers].values
Z = a4[dims]
grp = session_groups(r).loc[beers].values
style = a4.tasting_category_fine.values
N_REP = 3


def fit_pred(y, tr, te):
    m = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0,
    )
    m.fit(X[tr], y[tr])
    return m.predict(X[te])


def one_fold(rep, tr, te):
    rt = r[r.beer_name.isin(set(beers[tr]))]
    alpha, _ = additive(rt, dims, n_iter=150)
    alpha = alpha.loc[beers[tr]]
    # representative taster: rating-weighted mean of per-taster mu, sd (training ratings only)
    g = rt.groupby("taster")[dims]
    w = g.size().values[:, None]
    mu = (g.mean().values * w).sum(0) / w.sum()
    sd = (g.std().fillna(0).values * w).sum(0) / w.sum()
    out = {}
    for j, d in enumerate(dims):
        yr = xbar[d].values
        ya = np.full(len(beers), np.nan)
        ya[tr] = alpha[d].values
        yz = Z[d].values
        p_raw, p_add, p_z = fit_pred(yr, tr, te), fit_pred(ya, tr, te), fit_pred(yz, tr, te)
        coef = np.linalg.lstsq(np.c_[np.ones(len(tr)), yz[tr]], yr[tr], rcond=None)[0]
        sm = pd.Series(yr[tr]).groupby(style[tr]).agg(["sum", "count"])
        gm = yr[tr].mean()
        p_b1 = np.array([
            (sm.loc[s, "sum"] + 5 * gm) / (sm.loc[s, "count"] + 5) if s in sm.index else gm
            for s in style[te]
        ])
        out[d] = dict(
            raw=p_raw, add=p_add, z_rep=mu[j] + sd[j] * p_z, z_ols=coef[0] + coef[1] * p_z,
            b0=np.full(len(te), gm), b1=p_b1,
        )
    return rep, te, out


if __name__ == "__main__":
    jobs = [
        (rep, tr, te)
        for rep in range(N_REP)
        for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=grp)
    ]
    res = Parallel(n_jobs=15)(delayed(one_fold)(*j) for j in jobs)
    methods = ["raw", "add", "z_rep", "z_ols", "b0", "b1"]
    P = {m: np.zeros((N_REP, len(beers), len(dims))) for m in methods}
    for rep, te, out in res:
        for j, d in enumerate(dims):
            for m in methods:
                P[m][rep, te, j] = out[d][m]
    np.savez(
        os.path.join(OUT, "exp_targets_oof.npz"), beers=beers, dims=np.array(dims), grp=grp,
        nb=nb, xbar=xbar.values, z=Z.values, **{f"pred_{m}": P[m] for m in methods},
    )
    print("saved")
