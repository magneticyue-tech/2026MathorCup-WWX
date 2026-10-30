import inspect

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score
from src.q1 import data
from src.q1.discretize import (
    apply_cuts,
    apply_rules,
    const_mode,
    dist_interval,
    dist_median,
    grade,
    predictive_dist,
    quantile_cuts,
    qwk_cuts,
    qwk_score,
)


def test_grade():
    v = np.array([-0.2, 0.49, 0.5, 5.5, 6.4])
    assert grade(v).tolist() == [0, 0, 1, 6, 6]
    assert grade is data.grade


def test_predictive_dist_and_summaries():
    p = predictive_dist(np.array([2.0]), np.array([-0.6, 0.0, 0.6]))
    np.testing.assert_allclose(p, [[0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0]], atol=1e-15)
    assert dist_median(p).tolist() == [2]
    lo, hi = dist_interval(p)
    assert (lo.tolist(), hi.tolist()) == ([1], [3])

    p = predictive_dist(np.array([0.1]), np.array([-0.3, -0.2, 0.0, 0.4, 0.9]))
    assert p[0, 0] == pytest.approx(0.6) and p[0, 1] == pytest.approx(0.4)
    assert dist_median(p).tolist() == [0]

    p = predictive_dist(np.array([5.9]), np.array([0.5]))
    assert p[0, 6] == 1.0


def test_const_mode_tie_smaller():
    assert const_mode(np.array([0, 0, 1, 1])) == 0


def test_quantile_cuts_and_apply():
    cuts = quantile_cuts(np.arange(6.0), np.array([0, 0, 0, 1, 1, 2]))
    np.testing.assert_allclose(cuts[:2], [2.5, 4.1666667], atol=1e-6)
    assert np.isinf(cuts[2:]).all()
    assert apply_cuts(np.array([2.4, 2.5, 4.2, 100]), cuts).tolist() == [0, 1, 2, 2]


def test_qwk_score():
    y = np.array([0, 1, 2, 3, 6])
    assert qwk_score(y, y) == 1.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.integers(0, 7, 50), rng.integers(0, 7, 50)
        ref = cohen_kappa_score(a, b, weights="quadratic", labels=range(7))
        assert abs(qwk_score(a, b) - ref) <= 1e-12
    assert np.isnan(qwk_score(np.full(10, 2), np.full(10, 2)))


def test_knn_selection():
    p = predictive_dist(np.array([0.0]), np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.1, 5.0]), 2)
    np.testing.assert_allclose(p, [[0, 0.5, 0.5, 0, 0, 0, 0]])
    p = predictive_dist(np.array([0.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, -1.0, 9.0]), 1)
    assert p[0, 1] == 1.0
    with pytest.raises(ValueError):
        predictive_dist(np.array([0.0]), np.array([1.0]), np.array([0.0]))


def test_qwk_cuts_improves():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0, 6, 300)
    g = grade(mu + rng.normal(0, 0.4, 300))
    c = qwk_cuts(mu, g)
    assert (np.diff(c) > 0).all()
    init = np.arange(0.5, 6.0, 1.0)
    assert qwk_score(g, apply_cuts(mu, c)) >= qwk_score(g, apply_cuts(mu, init))


def test_apply_rules_ranges_and_signature():
    rng = np.random.default_rng(0)
    mu_inner = rng.uniform(0, 4, 200)
    y_inner = np.clip(mu_inner + rng.normal(0, 0.5, 200), 0, 6)
    mu_te = rng.uniform(-0.5, 6.5, 40)
    out = apply_rules(mu_te, mu_inner, y_inner)
    for k in ("const", "round", "median", "quant", "qwk_cut", "median_knn"):
        assert out[k].dtype.kind == "i" and out[k].min() >= 0 and out[k].max() <= 6
        assert out[k].shape == (40,)
    for k in ("p", "p_knn"):
        np.testing.assert_allclose(out[k].sum(1), 1.0, atol=1e-12)
    assert len(inspect.signature(apply_rules).parameters) == 3
