import numpy as np
import pandas as pd
from src.q3.evaluate import IMPROVED, KEPT, NOT_IMPROVED, REACH, StyleDiscriminator, category
from src.q3.portrait import summarize
from src.q3.sens_search import StylePosterior, alpha_fit


def test_category_order():
    assert category(0.5, 0.1, 0.0, 0.3) == REACH
    assert category(0.2, 0.1, 0.0, 0.3) == IMPROVED
    assert category(0.12, 0.1, 0.0, 0.3) == KEPT
    assert category(0.12, 0.1, 0.5, 0.3) == NOT_IMPROVED


def test_summarize():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0, 6, size=(100, 4))
    G = np.clip(np.round(mu), 0, 6).astype(int)
    p = rng.dirichlet(np.ones(7), size=(100, 4))
    s = summarize(mu, G, p)
    assert np.allclose(s["pi"].sum(-1), 1) and np.allclose(s["pi_tilde"].sum(-1), 1)
    assert (s["mu_q10"] <= s["mu_q50"]).all() and (s["mu_q50"] <= s["mu_q90"]).all()


class _G:
    def __init__(self):
        self.L = np.eye(2)
        self.mu = {"a": np.zeros(2), "b": np.ones(2)}


def test_alpha_fit_ties_and_feasible():
    post = StylePosterior(_G(), ["a", "b"], np.log([0.5, 0.5]))
    U = np.array([[0, 0], [1, 1], [1, 1], [2, 2.0]])
    alphas = np.array([0, 0.5, 0.75, 1.0])
    a, _ = alpha_fit(post, np.array([True, True, True, False]), U, "b", alphas)
    assert a == 0.5


def test_discriminator_no_leak():
    rng = np.random.default_rng(0)
    s = np.array(["a"] * 20 + ["b"] * 20)
    X = rng.normal(size=(40, 3))
    d = StyleDiscriminator(X, s)
    for b in range(40):
        assert b not in d.train_idx(b)
    assert isinstance(pd.Series(d.p_real_oof("a")).mean(), float)
