import numpy as np
import pytest
from src.q3.generator import StyleGen, fit_grid
from src.q3.selection import outer_splits


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    s = np.array(["a"] * 25 + ["b"] * 25 + ["c"] * 10)
    X = rng.normal(size=(60, 30)) + (s == "b")[:, None] * 1.0
    X[:, :5] += X[:, [0]]
    X[3, 2] = np.nan
    return X, s, ["a", "b", "c", "d"]


def test_reproducible_and_psd():
    X, s, st = _toy()
    g = StyleGen("famod", 3, 2.0).fit(X, s, st)
    a = g.sample("a", 20, np.random.default_rng(5))
    b = g.sample("a", 20, np.random.default_rng(5))
    assert np.array_equal(a, b)
    assert np.allclose(g.C, g.C.T)
    assert np.linalg.eigvalsh(g.C).min() > 0
    assert np.abs(g.L @ g.L.T - (g.C + 1e-9 * np.eye(30))).max() <= 1e-8
    assert (a >= np.nanmin(X, 0) - 1e-12).all() and (a <= np.nanmax(X, 0) + 1e-12).all()


def test_with_kappa_and_grid():
    X, s, st = _toy()
    g = StyleGen("famod", 3, 2.0).fit(X, s, st)
    g5 = StyleGen("famod", 3, 5.0).fit(X, s, st)
    w = g.with_kappa(5.0)
    for k in ("a", "b", "c"):
        assert np.abs(w.mu[k] - g5.mu[k]).max() <= 1e-9
    assert np.abs(w.score(X, s) - g5.score(X, s)).max() <= 1e-9
    cfgs = [("fa", 3, 1.0), ("famod", 3, 2.0), ("lw", None, 0.0)]
    grid = fit_grid(X, s, st, cfgs)
    for kind, k, kap in cfgs:
        one = StyleGen(kind, k, kap).fit(X, s, st)
        assert np.abs(grid[(kind, k, kap)].C - one.C).max() <= 1e-9
        assert np.abs(grid[(kind, k, kap)].score(X, s) - one.score(X, s)).max() <= 1e-9


def test_missing_style():
    X, s, st = _toy()
    g = StyleGen("fa", 3, 2.0).fit(X, s, st)
    assert np.isnan(g.score(X[:2], ["d", "d"])).all()
    with pytest.raises(ValueError):
        g.sample_latent("d", 3, np.random.default_rng(0))


def test_outer_splits():
    s = np.array(["a"] * 30 + ["b"] * 20 + ["c"] * 3)
    sp = outer_splits(s)
    assert len(sp) == 25
    for r in range(5):
        cnt = np.zeros(len(s), int)
        for rep, _, _, te in sp:
            if rep == r:
                cnt[te] += 1
        assert (cnt == 1).all()
