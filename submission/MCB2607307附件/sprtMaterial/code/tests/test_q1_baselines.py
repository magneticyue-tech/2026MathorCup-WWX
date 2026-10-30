import numpy as np
import pandas as pd
import pytest
from src.q1.baselines import (
    EN_CONFIGS,
    RF_CONFIGS,
    b0_mean,
    b1_style_shrunk,
    b2_log_columns,
    b2_prepare,
    chemstyle_input,
    fit_outer_fold_generic,
    fit_predict_enet,
    fit_predict_lin10,
    fit_predict_rf,
    prep_linear,
    prep_plain,
    simple_fold,
)
from src.q1.data import FLOOR_VALUE, load_q1
from src.q1.splits import outer_splits

DIMS2 = ["alcohol", "body"]
GROUPS2 = {"Basic": DIMS2}


@pytest.fixture(scope="module")
def setup():
    d = load_q1()
    groups = d.targets.date_group
    splits = outer_splits(groups, d.style)
    return d, groups, splits


def test_b0_b1():
    y = np.array([1.0, 3.0, 5.0, 7.0])
    assert b0_mean(y, 2).tolist() == [4.0, 4.0]
    out = b1_style_shrunk(y, np.array(["a", "a", "b", "b"]), np.array(["a", "c"]))
    np.testing.assert_allclose(out, [24 / 7, 4.0])


@pytest.mark.parametrize("kind", ["b0", "b1"])
def test_simple_fold_inner_rows_and_no_leak(setup, kind):
    d, groups, splits = setup
    tr, te = splits[0][0]
    Y = d.targets.xbar
    a = simple_fold(kind, Y, d.style, groups, tr, te, 100)
    filled = ~np.isnan(a["inner"]).any(1)
    assert filled.all() and a["inner"].shape == (len(tr), 50)
    Y2 = Y.copy()
    Y2[te] = np.random.default_rng(1).uniform(0, 6, size=Y2[te].shape)
    b = simple_fold(kind, Y2, d.style, groups, tr, te, 100)
    np.testing.assert_array_equal(a["oof"], b["oof"])
    np.testing.assert_array_equal(a["inner"], b["inner"])


def test_prep_linear_floor_columns_from_fit_only():
    fit = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 0.6, np.nan, 0.8]})
    other = pd.DataFrame({"a": [FLOOR_VALUE, 2.5], "b": [0.1, 0.2]})
    X_fit, (X_o,), names = prep_linear(fit, [other])
    assert "a_floor" not in names and names == ["a", "b"]
    other2 = other.copy()
    other2.loc[0, "b"] = 99.0
    X_fit2, _, _ = prep_linear(fit, [other2])
    np.testing.assert_array_equal(X_fit, X_fit2)
    fit.loc[0, "a"] = FLOOR_VALUE
    _, _, names = prep_linear(fit, [other])
    assert names == ["a", "b", "a_floor"]


def test_lin10_selects_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 15))
    y = 2 * X[:, 3] + rng.normal(scale=0.05, size=100)
    names = [f"c{i}" for i in range(15)]
    (pred,), sel = fit_predict_lin10({}, X[:70], y[:70], [X[70:]], names, return_selected=True)
    assert 3 in sel
    r2 = 1 - ((y[70:] - pred) ** 2).sum() / ((y[70:] - y[70:].mean()) ** 2).sum()
    assert r2 > 0.9


def _generic(d, groups, Y, tr, te, configs, prep, fp):
    return fit_outer_fold_generic(
        d.X, Y, DIMS2, GROUPS2, groups, d.style, tr, te, 100, configs, prep, fp
    )


def test_generic_no_leakage(setup):
    d, groups, splits = setup
    tr, te = splits[0][0]
    cols = [d.dims.index(x) for x in DIMS2]
    Y = d.targets.xbar[:, cols]
    a = _generic(d, groups, Y, tr, te, EN_CONFIGS[:3], prep_linear, fit_predict_enet)
    Y2 = Y.copy()
    Y2[te] = np.random.default_rng(1).uniform(0, 6, size=Y2[te].shape)
    b = _generic(d, groups, Y2, tr, te, EN_CONFIGS[:3], prep_linear, fit_predict_enet)
    assert a["chosen"] == b["chosen"]
    assert a["scores"] == b["scores"]
    np.testing.assert_array_equal(a["inner"], b["inner"])
    np.testing.assert_array_equal(a["oof"], b["oof"])
    Y3 = Y.copy()
    Y3[tr] += 1.0
    c = _generic(d, groups, Y3, tr, te, EN_CONFIGS[:3], prep_linear, fit_predict_enet)
    assert not np.allclose(a["oof"], c["oof"])


def test_rf_small(setup):
    d, groups, splits = setup
    tr, te = splits[0][0]
    cols = [d.dims.index(x) for x in DIMS2]
    Y = d.targets.xbar[:, cols]
    cfg = [{**RF_CONFIGS[0], "n_estimators": 20}]
    u = _generic(d, groups, Y, tr, te, cfg, prep_plain, fit_predict_rf)
    assert u["oof"].shape == (len(te), 2) and u["inner"].shape == (len(tr), 2)
    assert np.isfinite(u["oof"]).all() and np.isfinite(u["inner"]).all()


def test_b2_log_columns():
    X = pd.DataFrame(
        {
            "skew": np.r_[np.exp(np.random.default_rng(0).normal(size=40) * 2)],
            "neg": np.r_[-1.0, np.exp(np.random.default_rng(1).normal(size=39) * 2)],
        }
    )
    sel = b2_log_columns(X)
    assert "neg" not in sel and "skew" in sel
    te = X.iloc[:5].copy()
    A1, _ = b2_prepare(X, te)
    te.iloc[:, :] = 7.0
    A2, _ = b2_prepare(X, te)
    np.testing.assert_array_equal(A1, A2)


def test_chemstyle_input(setup):
    d, _, _ = setup
    Xs = chemstyle_input(d.X, d.style)
    sty = [c for c in Xs.columns if c.startswith("style_")]
    assert len(sty) == d.style.nunique() == 22
    assert sty == sorted(sty)
    assert Xs.shape == (250, 231 + 22)
