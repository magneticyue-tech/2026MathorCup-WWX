import numpy as np
from scipy.stats import spearmanr
from src.q1.data import load_q1
from src.q1.discretize import dist_median
from src.q1.ordinal import cumulative_to_dist, fit_predict_k2, k2_fold
from src.q1.splits import outer_splits


def test_cumulative_to_dist():
    p = cumulative_to_dist(np.array([[0.9, 0.5, 0.6, 0.2, 0.1, 0.0]]))
    np.testing.assert_allclose(p, [[0.1, 0.4, 0.0, 0.3, 0.1, 0.1, 0.0]], atol=1e-12)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
    assert dist_median(p).tolist() == [1]


PARAMS = {
    "max_depth": 2,
    "min_samples_leaf": 10,
    "learning_rate": 0.05,
    "l2_regularization": 0.0,
    "max_iter": 100,
}


def test_all_zero_fallback():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    p, fb = fit_predict_k2(X, np.zeros(40, int), X[:5], PARAMS, None)
    assert fb.all()
    np.testing.assert_array_equal(p, np.tile([1, 0, 0, 0, 0, 0, 0], (5, 1)))


def test_synthetic_ordering():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(150, 5))
    G_tr = np.clip(np.round(2 + 1.5 * X_tr[:, 0]), 0, 6).astype(int)
    X_te = rng.normal(size=(50, 5))
    p, _ = fit_predict_k2(X_tr, G_tr, X_te, PARAMS, None)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
    assert (p >= 0).all()
    assert spearmanr(p @ np.arange(7), X_te[:, 0])[0] > 0.5


def test_k2_fold_no_leakage():
    d = load_q1()
    splits = outer_splits(d.targets.date_group, d.style)
    tr, te = splits[0][0]
    dims = ["alcohol", "body"]
    xbar = d.targets.xbar[:, [d.dims.index(x) for x in dims]]
    groups = {"Basic": dims}
    chosen = {"Basic": {**PARAMS, "mono": False}}
    a = k2_fold(d.X, xbar, dims, groups, tr, te, chosen)
    x2 = xbar.copy()
    x2[te] = np.random.default_rng(1).uniform(0, 6, size=x2[te].shape)
    b = k2_fold(d.X, x2, dims, groups, tr, te, chosen)
    np.testing.assert_array_equal(a["p"], b["p"])
    x3 = xbar.copy()
    x3[tr] += 1.0
    c = k2_fold(d.X, x3, dims, groups, tr, te, chosen)
    assert not np.allclose(a["p"], c["p"])
    assert a["p"].shape == (len(te), 2, 7) and a["fallback"].shape == (2, 6)
