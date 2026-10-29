import inspect

import numpy as np
from sklearn.metrics import cohen_kappa_score
from src.q1.metrics import (
    PAIR_STATS,
    balanced_accuracy,
    bootstrap_weights,
    climatology_p,
    cluster_bootstrap,
    conf_weighted,
    coverage,
    exact,
    grade_mae,
    logloss_cells,
    mae_dim,
    pair_stat_fn,
    qwk_dim,
    qwk_pair,
    r2_by_dim,
    recall_by_class,
    reference_error,
    rps_cells,
    spearman_by_dim,
    wins_by_dim,
    within1,
)


def test_grade_mae_exact_within1():
    G = np.ones((250, 50), int)
    Gh = np.full((5, 250, 50), 3)
    assert grade_mae(G, Gh) == 2.0
    assert exact(G, Gh) == 0.0 and within1(G, Gh) == 0.0


def test_conf_weighted_and_qwk():
    rng = np.random.default_rng(0)
    G = rng.integers(0, 7, (40, 6))
    Gh = rng.integers(0, 7, (3, 40, 6))
    conf = conf_weighted(G, Gh)
    assert conf.shape == (3, 6, 7, 7)
    assert (conf.sum(axis=(2, 3)) == 40).all()
    assert abs(mae_dim(conf).mean() - np.abs(G[None] - Gh).mean()) < 1e-12
    for _ in range(20):
        a, b = rng.integers(0, 7, 50), rng.integers(0, 7, 50)
        q = qwk_dim(conf_weighted(a[:, None], b[None, :, None]))[0, 0]
        ref = cohen_kappa_score(a, b, weights="quadratic", labels=range(7))
        assert abs(q - ref) <= 1e-10
    G2 = np.full((10, 1), 2)
    assert np.isnan(qwk_dim(conf_weighted(G2, G2[None]))[0, 0])


def test_qwk_pair():
    a, b, n = qwk_pair(np.array([0.2, np.nan, 0.4]), np.array([0.1, 0.3, np.nan]))
    assert (round(a, 12), round(b, 12), n) == (0.2, 0.1, 2)


def test_balanced_accuracy():
    cm = np.diag([2, 2, 2, 2, 2, 0, 0]).astype(float)
    cm[0] = [1, 1, 0, 0, 0, 0, 0]
    rec = recall_by_class(cm)
    np.testing.assert_allclose(rec[:5], [0.5, 1, 1, 1, 1])
    assert np.isnan(rec[5:]).all()
    assert abs(balanced_accuracy(cm) - 0.9) < 1e-12


def test_rps_and_logloss():
    p = np.array([0, 0.5, 0.5, 0, 0, 0, 0])[None, None, None, :]
    for g, v in ((1, 0.25), (2, 0.25), (0, 1.25)):
        assert abs(rps_cells(p, np.array([[g]]))[0, 0, 0] - v) < 1e-12
    p = np.array([1.0, 0, 0, 0, 0, 0, 0])[None, None, None, :]
    assert abs(logloss_cells(p, np.array([[0]]))[0, 0, 0] + np.log(0.99 + 0.01 / 7)) < 1e-12
    assert abs(logloss_cells(p, np.array([[1]]))[0, 0, 0] + np.log(0.01 / 7)) < 1e-12


def test_climatology_uses_train_only():
    rng = np.random.default_rng(0)
    G = rng.integers(0, 7, (20, 3))
    fold_id = np.tile(np.arange(20) % 4, (2, 1))
    fold_id[1] = (np.arange(20) + 1) % 4
    a = climatology_p(G, fold_id)
    G2 = G.copy()
    G2[fold_id[0] == 0] = 6
    b = climatology_p(G2, fold_id)
    np.testing.assert_array_equal(a[0, fold_id[0] == 0], b[0, fold_id[0] == 0])
    np.testing.assert_allclose(a.sum(-1), 1.0)


def test_r2_and_spearman():
    rng = np.random.default_rng(0)
    xbar = rng.normal(size=(30, 3))
    assert np.allclose(r2_by_dim(xbar[None], xbar), 1.0)
    const = np.broadcast_to(xbar.mean(0), xbar.shape)[None]
    assert np.allclose(r2_by_dim(const, xbar), 0.0)
    mu = xbar[None].copy()
    mu[0, :, 1] = 1.0
    rho = spearman_by_dim(mu, xbar)
    assert np.isnan(rho[0, 1]) and np.allclose(rho[0, [0, 2]], 1.0)


def test_wins_by_dim():
    G = np.zeros((4, 3), int)
    a = np.zeros((1, 4, 3), int)
    b = a.copy()
    b[0, :, 0] = 1
    b[0, :, 1] = 2
    assert wins_by_dim(G, a, b) == 2


def test_coverage():
    G = np.array([[1, 2], [3, 0]])
    lo = np.array([[0, 3], [3, 0]])
    hi = np.array([[2, 4], [3, 1]])
    c, w = coverage(G, lo, hi)
    assert c == 0.75 and w == 1.0


def test_reference_error():
    rng = np.random.default_rng(0)
    G = rng.integers(0, 6, (10, 4))
    R = np.broadcast_to(G + 1, (5, 10, 4)).copy()
    local = np.zeros((5, 10, 4), bool)
    local[:, :5] = True
    out = reference_error(R, local, G)
    for k in ("all", "A", "L"):
        assert out[k]["bias"] == 1.0 and out[k]["mae"] == 1.0 and out[k]["exact"] == 0.0
    assert out["A"]["cells_per_rep"] == 20.0


def test_weighted_bootstrap_matches_resampling():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(5), 4)
    G = rng.integers(0, 7, (20, 3))
    ga = rng.integers(0, 7, (2, 20, 3))
    gb = rng.integers(0, 7, (2, 20, 3))
    W = bootstrap_weights(groups, n_boot=5)
    w = W[0]
    idx = np.repeat(np.arange(20), w.astype(int))
    stat = pair_stat_fn(G, {"g": ga}, {"g": gb})(w)
    Gr, ar, br = G[idx], ga[:, idx], gb[:, idx]
    mae = np.abs(ar - Gr[None]).mean() - np.abs(br - Gr[None]).mean()
    ex = (ar == Gr[None]).mean() - (br == Gr[None]).mean()

    def q(g):
        v = np.array(
            [
                [
                    cohen_kappa_score(Gr[:, d], g[r, :, d], weights="quadratic", labels=range(7))
                    for d in range(3)
                ]
                for r in range(2)
            ]
        )
        return v

    qa, qb = q(ar), q(br)
    m = ~np.isnan(qa) & ~np.isnan(qb)
    assert abs(stat[PAIR_STATS.index("mae")] - mae) < 1e-12
    assert abs(stat[PAIR_STATS.index("exact")] - ex) < 1e-12
    assert abs(stat[PAIR_STATS.index("qwk")] - (qa[m].mean() - qb[m].mean())) < 1e-12
    assert np.isnan(stat[PAIR_STATS.index("rps")])


def test_cluster_bootstrap_deterministic_and_zero():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(5), 4)
    G = rng.integers(0, 7, (20, 3))
    ga = rng.integers(0, 7, (2, 20, 3))
    W = bootstrap_weights(groups, n_boot=50)
    f = pair_stat_fn(G, {"g": ga}, {"g": ga})
    p1, lo1, hi1 = cluster_bootstrap(f, W)
    p2, lo2, hi2 = cluster_bootstrap(f, bootstrap_weights(groups, n_boot=50))
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(p1, f(np.ones(20)))
    i = PAIR_STATS.index("mae")
    assert lo1[i] == hi1[i] == 0.0


def test_pair_stat_fn_signature():
    assert list(inspect.signature(pair_stat_fn).parameters) == ["G", "a", "b", "xbar"]
