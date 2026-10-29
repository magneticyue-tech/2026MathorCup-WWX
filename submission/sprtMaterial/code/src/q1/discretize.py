import numpy as np

from .data import grade

N_GRADES = 7
_W = (np.arange(N_GRADES)[:, None] - np.arange(N_GRADES)[None, :]) ** 2


def qwk_score(y, yhat) -> float:
    o = np.bincount(N_GRADES * np.asarray(y) + np.asarray(yhat), minlength=49).reshape(7, 7)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    den = (_W * e).sum()
    if den == 0:
        return np.nan
    return float(1 - (_W * o).sum() / den)


def predictive_dist(mu_te, resid, mu_ref=None, k=None) -> np.ndarray:
    if (mu_ref is None) != (k is None):
        raise ValueError("mu_ref 与 k 必须同时给出或同时省略")
    mu_te = np.asarray(mu_te, float)
    resid = np.asarray(resid, float)
    if mu_ref is None:
        g = grade(mu_te[:, None] + resid[None, :])
    else:
        mu_ref = np.asarray(mu_ref, float)
        sel = np.array([np.argsort(np.abs(mu_ref - m), kind="stable")[:k] for m in mu_te])
        g = grade(mu_te[:, None] + resid[sel])
    counts = (g[:, :, None] == np.arange(N_GRADES)).sum(1)
    return counts / g.shape[1]


def _first_reaching(p, threshold) -> np.ndarray:
    return np.argmax(np.cumsum(p, axis=-1) >= threshold, axis=-1)


def dist_median(p) -> np.ndarray:
    return _first_reaching(p, 0.5 - 1e-12)


def dist_interval(p, lo=0.05, hi=0.95):
    return _first_reaching(p, lo - 1e-12), _first_reaching(p, hi - 1e-12)


def const_mode(g_train) -> int:
    return int(np.argmax(np.bincount(g_train, minlength=N_GRADES)))


def quantile_cuts(mu_inner, g_train) -> np.ndarray:
    cum = np.cumsum(np.bincount(g_train, minlength=N_GRADES))[:-1] / len(g_train)
    cuts = np.quantile(mu_inner, np.clip(cum, 0, 1))
    cuts[cum >= 1 - 1e-12] = np.inf
    return cuts


def apply_cuts(mu, cuts) -> np.ndarray:
    return np.searchsorted(cuts, mu, side="right")


def qwk_cuts(mu_inner, g_train, n_pass=3) -> np.ndarray:
    c = np.arange(0.5, 6.0, 1.0)

    def score(t):
        return np.nan_to_num(qwk_score(g_train, apply_cuts(mu_inner, t)), nan=-1.0)

    best = score(c)
    grid = np.unique(mu_inner)
    grid = grid[:: max(1, len(grid) // 120)]
    for _ in range(n_pass):
        for i in range(6):
            lo = c[i - 1] if i > 0 else -np.inf
            hi = c[i + 1] if i < 5 else np.inf
            for v in grid[(grid > lo) & (grid < hi)]:
                t = c.copy()
                t[i] = v
                s = score(t)
                if s > best:
                    c, best = t, s
    return c


def apply_rules(mu_te, mu_inner, y_inner) -> dict:
    g_inner = grade(y_inner)
    resid = y_inner - mu_inner
    p = predictive_dist(mu_te, resid)
    p_knn = predictive_dist(mu_te, resid, mu_inner, 50)
    n = len(mu_te)
    return {
        "p": p,
        "p_knn": p_knn,
        "const": np.full(n, const_mode(g_inner), dtype=int),
        "round": grade(mu_te),
        "median": dist_median(p),
        "quant": apply_cuts(mu_te, quantile_cuts(mu_inner, g_inner)),
        "qwk_cut": apply_cuts(mu_te, qwk_cuts(mu_inner, g_inner)),
        "median_knn": dist_median(p_knn),
    }
