import warnings

import numpy as np
from scipy.stats import spearmanr

from .paths import SEED

N_BOOT = 2000
EPS_LOG = 0.01
CTRL_MODELS = ("b0", "b1", "lin10", "enet", "rf", "chemstyle")
ALL_RULES = ("const", "round", "median", "quant", "qwk_cut", "median_knn")
PAIR_STATS = ("mae", "exact", "within1", "qwk", "rps", "logloss", "cont_mae")

_K = 7
_I = np.arange(_K)
_ABS = np.abs(_I[:, None] - _I[None, :])
_W = _ABS**2


def conf_weighted(G, Gh, w=None):
    G, Gh = np.asarray(G), np.asarray(Gh)
    R, n, D = Gh.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    idx = ((np.arange(R)[:, None, None] * D + np.arange(D)[None, None, :]) * 49) + 7 * G[None] + Gh
    weights = np.broadcast_to(w[None, :, None], Gh.shape)
    conf = np.bincount(idx.ravel(), weights=weights.ravel(), minlength=R * D * 49)
    return conf.reshape(R, D, _K, _K)


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def mae_dim(conf):
    return _ratio((conf * _ABS).sum((-2, -1)), conf.sum((-2, -1)))


def exact_dim(conf):
    return _ratio(np.trace(conf, axis1=-2, axis2=-1), conf.sum((-2, -1)))


def within1_dim(conf):
    return _ratio((conf * (_ABS <= 1)).sum((-2, -1)), conf.sum((-2, -1)))


def qwk_dim(conf):
    o = conf
    tot = o.sum((-2, -1), keepdims=True)
    e = o.sum(-1, keepdims=True) * o.sum(-2, keepdims=True) / tot
    den = (_W * e).sum((-2, -1))
    num = (_W * o).sum((-2, -1))
    out = np.full(den.shape, np.nan)
    ok = den > 0
    out[ok] = 1 - num[ok] / den[ok]
    return out


def grade_mae(G, Gh):
    return float(mae_dim(conf_weighted(G, Gh)).mean())


def exact(G, Gh):
    return float(exact_dim(conf_weighted(G, Gh)).mean())


def within1(G, Gh):
    return float(within1_dim(conf_weighted(G, Gh)).mean())


def qwk_by_dim(G, Gh):
    return qwk_dim(conf_weighted(G, Gh))


def mean_qwk(q):
    q = np.asarray(q, float)
    n_nan = int(np.isnan(q).sum())
    if n_nan == q.size:
        return np.nan, n_nan
    return float(np.nanmean(q)), n_nan


def qwk_pair(qa, qb):
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    mask = ~np.isnan(qa) & ~np.isnan(qb)
    if not mask.any():
        return np.nan, np.nan, int(mask.size)
    return float(qa[mask].mean()), float(qb[mask].mean()), int((~mask).sum())


def confusion_pooled(G, Gh):
    return conf_weighted(G, Gh).sum(1).mean(0)


def recall_by_class(cm):
    rows = cm.sum(1)
    out = np.full(_K, np.nan)
    ok = rows > 0
    out[ok] = np.diag(cm)[ok] / rows[ok]
    return out


def balanced_accuracy(cm):
    return float(np.nanmean(recall_by_class(cm)))


def share_high(Gh):
    return float((np.asarray(Gh) >= 5).mean())


def grade_freq(g):
    return np.bincount(np.asarray(g).ravel(), minlength=_K) / np.asarray(g).size


def rps_cells(p, G):
    F = np.cumsum(p, -1)[..., :6]
    obs = np.asarray(G)[None, :, :, None] <= np.arange(6)
    return ((F - obs) ** 2).sum(-1)


def logloss_cells(p, G, eps=EPS_LOG):
    pp = (1 - eps) * p + eps / _K
    idx = np.broadcast_to(np.asarray(G)[None, :, :, None], pp.shape[:-1] + (1,))
    return -np.log(np.take_along_axis(pp, idx, -1)[..., 0])


def climatology_p(G, fold_id):
    R, n = fold_id.shape
    D = G.shape[1]
    out = np.full((R, n, D, _K), np.nan)
    for r in range(R):
        for k in np.unique(fold_id[r][fold_id[r] >= 0]):
            te, tr = fold_id[r] == k, fold_id[r] != k
            freq = np.stack([np.bincount(G[tr, d], minlength=_K) for d in range(D)]) / tr.sum()
            out[r, te] = freq[None]
    return out


def cont_abs_cells(mu, xbar):
    return np.abs(mu - xbar[None])


def r2_by_dim(mu, xbar):
    ss_res = ((xbar[None] - mu) ** 2).sum(1)
    ss_tot = ((xbar - xbar.mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def rmse_over_sd(mu, xbar):
    return np.sqrt(((xbar[None] - mu) ** 2).mean(1)) / np.std(xbar, axis=0, ddof=0)


def spearman_by_dim(mu, xbar):
    R, _, D = mu.shape
    out = np.full((R, D), np.nan)
    for r in range(R):
        for d in range(D):
            if np.ptp(mu[r, :, d]) == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[r, d] = spearmanr(mu[r, :, d], xbar[:, d])[0]
    return out


def _dim_err(G, Gh):
    return np.abs(np.asarray(Gh) - np.asarray(G)[None]).mean(axis=(0, 1))


def wins_by_dim(G, Gh_a, Gh_b):
    return int((_dim_err(G, Gh_a) < _dim_err(G, Gh_b)).sum())


def wins_by_dim_cont(mu_a, mu_b, xbar):
    ea = np.abs(mu_a - xbar[None]).mean(axis=(0, 1))
    eb = np.abs(mu_b - xbar[None]).mean(axis=(0, 1))
    return int((ea < eb).sum())


def coverage(G, lo, hi):
    return float(((G >= lo) & (G <= hi)).mean()), float((hi - lo).mean())


def subset_report(G, Gh, idx):
    idx = np.asarray(idx)
    conf = conf_weighted(G[:, idx], Gh[:, :, idx])
    q, n_nan = mean_qwk(qwk_dim(conf))
    return {
        "n_dims": int(len(idx)),
        "mae": float(mae_dim(conf).mean()),
        "exact": float(exact_dim(conf).mean()),
        "qwk": q,
        "qwk_undefined": n_nan,
    }


def reference_error(R, I, G):  # noqa: E741
    Gb = np.broadcast_to(np.asarray(G)[None], R.shape)
    out = {}
    for name, mask in (("all", np.ones(R.shape, bool)), ("A", I), ("L", ~I)):
        r, g = R[mask], Gb[mask]
        out[name] = {
            "cells_per_rep": float(mask.sum() / R.shape[0]),
            "exact": float((r == g).mean()),
            "within1": float((np.abs(r - g) <= 1).mean()),
            "mae": float(np.abs(r - g).mean()),
            "bias": float((r - g).mean()),
        }
    return out


def bootstrap_weights(groups, n_boot=N_BOOT, seed=SEED):
    _, gidx = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(gidx.max()) + 1
    rng = np.random.default_rng(seed)
    W = np.empty((n_boot, len(gidx)))
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        W[i] = counts[gidx]
    return W


def cluster_bootstrap(stat_fn, W):
    point = np.asarray(stat_fn(np.ones(W.shape[1])), float)
    boot = np.array([stat_fn(w) for w in W], float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
    return point, lo, hi


def pair_stat_fn(G, a, b, xbar=None):
    R, _, D = np.asarray(a["g"]).shape

    def per_beer(cells):
        return cells.sum(axis=(0, 2))

    both_p = "p" in a and "p" in b
    both_mu = "mu" in a and "mu" in b and xbar is not None
    if both_p:
        rps_a, rps_b = per_beer(rps_cells(a["p"], G)), per_beer(rps_cells(b["p"], G))
        ll_a, ll_b = per_beer(logloss_cells(a["p"], G)), per_beer(logloss_cells(b["p"], G))
    if both_mu:
        cm_a = per_beer(cont_abs_cells(a["mu"], xbar))
        cm_b = per_beer(cont_abs_cells(b["mu"], xbar))

    def stat_fn(w):
        ca, cb = conf_weighted(G, a["g"], w), conf_weighted(G, b["g"], w)
        qa, qb, _ = qwk_pair(qwk_dim(ca), qwk_dim(cb))
        den = w.sum() * R * D
        out = [
            mae_dim(ca).mean() - mae_dim(cb).mean(),
            exact_dim(ca).mean() - exact_dim(cb).mean(),
            within1_dim(ca).mean() - within1_dim(cb).mean(),
            qa - qb,
            ((rps_a - rps_b) * w).sum() / den if both_p else np.nan,
            ((ll_a - ll_b) * w).sum() / den if both_p else np.nan,
            ((cm_a - cm_b) * w).sum() / den if both_mu else np.nan,
        ]
        return np.array(out, float)

    return stat_fn
