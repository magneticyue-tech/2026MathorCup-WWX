import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import ChemNAEncoder, grade
from .features import monotone_cst
from .paths import SEED

N_THRESH = 6


def cumulative_to_dist(q_raw) -> np.ndarray:
    q_raw = np.asarray(q_raw, float)
    Q = np.hstack([np.ones((len(q_raw), 1)), q_raw])
    Q = np.minimum.accumulate(Q, axis=1)
    Q = np.hstack([Q, np.zeros((len(q_raw), 1))])
    return Q[:, :-1] - Q[:, 1:]


def fit_predict_k2(X_tr, G_tr, X_te, params, cst, min_pos=2):
    q_raw = np.empty((len(X_te), N_THRESH))
    fallback = np.zeros(N_THRESH, dtype=bool)
    for t in range(1, N_THRESH + 1):
        y = (np.asarray(G_tr) >= t).astype(int)
        if min(y.sum(), len(y) - y.sum()) < min_pos:
            q_raw[:, t - 1] = y.mean()
            fallback[t - 1] = True
            continue
        clf = HistGradientBoostingClassifier(
            loss="log_loss", early_stopping=False, random_state=SEED, monotonic_cst=cst, **params
        ).fit(X_tr, y)
        q_raw[:, t - 1] = clf.predict_proba(X_te)[:, 1]
    return cumulative_to_dist(q_raw), fallback


def k2_fold(X_raw, xbar, dims, dim_groups, train_idx, test_idx, chosen) -> dict:
    tr, te = np.sort(train_idx), np.sort(test_idx)
    dims = list(dims)
    G_tr = grade(xbar[tr])
    enc = ChemNAEncoder("col_mean").fit(X_raw.iloc[tr])
    X_tr_df = enc.transform(X_raw.iloc[tr])
    names = list(X_tr_df.columns)
    X_tr = X_tr_df.to_numpy(np.float64)
    X_te = enc.transform(X_raw.iloc[te]).to_numpy(np.float64)
    group_of = {d: g for g, ds in dim_groups.items() for d in ds}
    p = np.empty((len(te), len(dims), 7))
    fb = np.zeros((len(dims), N_THRESH), dtype=bool)
    for j, d in enumerate(dims):
        best = chosen[group_of[d]]
        params = {k: v for k, v in best.items() if k != "mono"}
        cst = monotone_cst(names, d, best["mono"]) if best["mono"] else None
        p[:, j], fb[j] = fit_predict_k2(X_tr, G_tr[:, j], X_te, params, cst)
    return {"test_idx": te, "p": p, "fallback": fb}
