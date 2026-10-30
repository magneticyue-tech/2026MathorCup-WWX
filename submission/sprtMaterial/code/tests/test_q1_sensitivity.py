import numpy as np
import pandas as pd
import pytest
from src.q1 import sensitivity
from src.q1.data import NA_COLS, ChemNAEncoder, load_q1
from src.q1.features import monotone_cst
from src.q1.model import CONFIGS, fit_outer_fold
from src.q1.sensitivity import (
    SUM_COLS,
    alpha_target,
    apply_att8,
    arm_kwargs,
    changed_share,
    fit_fold_fixed,
    merge_arm,
    mono_flag,
)
from src.q1.splits import outer_splits

SMALL = [
    c
    for c in CONFIGS
    if c["max_depth"] == 2 and c["min_samples_leaf"] == 10 and c["l2_regularization"] == 0.0
]
DIMS2 = ["alcohol", "body"]
GROUPS2 = {"Basic": DIMS2}


@pytest.fixture(scope="module")
def real():
    d = load_q1()
    splits = outer_splits(d.targets.date_group, d.style)
    return d, splits


def _toy_ratings():
    alpha = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 2.5, 6: 1.5}
    beta = {"t1": 0.5, "t2": -0.5}
    rows = [(b, t, a + beta[t]) for b, a in alpha.items() for t in beta]
    return pd.DataFrame(rows, columns=["row", "taster", "d"]), alpha


def test_alpha_target_train_only():
    r, alpha = _toy_ratings()
    tr = np.array([0, 1, 2, 3])
    out = alpha_target(r, tr, dims=["d"])
    np.testing.assert_allclose(out[tr, 0], [alpha[i + 1] for i in tr], atol=1e-8)
    assert np.isnan(out[[4, 5], 0]).all()
    r2 = r.copy()
    r2.loc[r2["row"].isin([5, 6]), "d"] += 3.0
    np.testing.assert_array_equal(alpha_target(r2, tr, dims=["d"])[tr], out[tr])


def test_apply_att8_toy(tmp_path, monkeypatch):
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    beer = pd.Series([10, 20, 30])
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "beer": [20, 30],
            "field": ["f1", "f2"],
            "csv1_literal": [2.0, 6.0],
            "calibrated_8": [9, 8],
        }
    ).to_csv(csv, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(sensitivity, "DISCORDANCE_CSV", csv)
    monkeypatch.setattr(sensitivity, "ATT8_EXPECTED", (2, 2))
    out, counts = apply_att8(X, beer)
    assert counts == {"n_cells": 2, "n_beers": 2}
    assert out.loc[1, "f1"] == 9 and out.loc[2, "f2"] == 8
    assert (out != X).sum().sum() == 2
    assert X.loc[1, "f1"] == 2.0
    pd.DataFrame(
        {"beer": [20], "field": ["f1"], "csv1_literal": [7.0], "calibrated_8": [9]}
    ).to_csv(csv, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(sensitivity, "ATT8_EXPECTED", (1, 1))
    with pytest.raises(AssertionError):
        apply_att8(X, beer)


def test_att8_real_and_arm_kwargs(real):
    d, _ = real
    out, counts = apply_att8(d.X, d.beer)
    assert counts == {"n_cells": 80, "n_beers": 24}
    Xa, _ = arm_kwargs("att8", d.X, d.beer)
    diff = ~((Xa == d.X) | (Xa.isna() & d.X.isna()))
    assert int(diff.values.sum()) == 80
    Xd, kw = arm_kwargs("drop_sum", d.X, d.beer)
    assert Xd.shape[1] == 231 and set(kw["drop_cols"]) == set(SUM_COLS)
    assert d.X.drop(columns=list(SUM_COLS)).shape[1] == 227
    with pytest.raises(ValueError):
        arm_kwargs("nope", d.X, d.beer)


def test_fit_fold_fixed_identity_and_no_leak(real):
    d, splits = real
    tr, te = splits[0][0]
    groups = d.targets.date_group
    Y = d.targets.xbar[:, [d.dims.index(x) for x in DIMS2]]
    ref = fit_outer_fold(d.X, Y, DIMS2, GROUPS2, groups, d.style, tr, te, 100, SMALL)
    a = fit_fold_fixed(d.X, Y, DIMS2, GROUPS2, groups, d.style, tr, te, 100, ref["chosen"])
    np.testing.assert_allclose(a["oof"], ref["oof"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(a["inner"], ref["inner"], atol=1e-10, rtol=0)
    Y2 = Y.copy()
    Y2[te] = np.random.default_rng(1).uniform(0, 6, size=Y2[te].shape)
    b = fit_fold_fixed(d.X, Y2, DIMS2, GROUPS2, groups, d.style, tr, te, 100, ref["chosen"])
    np.testing.assert_array_equal(a["oof"], b["oof"])
    np.testing.assert_array_equal(a["inner"], b["inner"])
    Y3 = Y.copy()
    Y3[tr] += 1.0
    c = fit_fold_fixed(d.X, Y3, DIMS2, GROUPS2, groups, d.style, tr, te, 100, ref["chosen"])
    assert not np.allclose(a["oof"], c["oof"])


def test_mono_mode(real):
    d, _ = real
    names = list(ChemNAEncoder().fit(d.X).transform(d.X).columns)
    best = {"mono": True}
    assert not any(monotone_cst(names, "alcohol", mono_flag("off", best)))
    cst = monotone_cst(names, "alcohol", mono_flag("on", {"mono": False}))
    assert [names[i] for i, v in enumerate(cst) if v] == ["ethanol..v.v."]
    assert mono_flag("chosen", best) is True and mono_flag("chosen", {"mono": False}) is False


def test_native_nan_and_indicator_counts(real):
    d, _ = real
    assert int(d.X.isna().sum().sum()) == 73
    med = ChemNAEncoder("median_ind").fit(d.X).transform(d.X)
    flo = ChemNAEncoder("floor_ind").fit(d.X).transform(d.X)
    assert sum(c.endswith("_missing") for c in med.columns) == len(NA_COLS)
    assert sum(c.endswith("_missing") for c in flo.columns) == len(NA_COLS) - 1


def _unit(tr, te, D=2):
    n_tr, n_te = len(tr), len(te)
    return {
        "train_idx": np.array(tr),
        "test_idx": np.array(te),
        "oof": np.zeros((n_te, D)),
        "inner": np.ones((n_tr, D)),
        "median": np.zeros((n_te, D), int),
        "round": np.zeros((n_te, D), int),
        "quant": np.zeros((n_te, D), int),
        "p": np.tile(np.eye(7)[0], (n_te, D, 1)),
    }


def test_merge_arm():
    units = {(0, 0): _unit([2, 3], [0, 1]), (0, 1): _unit([0, 1], [2, 3])}
    out = merge_arm(units, shape=(1, 4, 2), n_fold=2)
    filled = np.where(~np.isnan(out["inner_mu"][0, 0]).any(1))[0]
    assert filled.tolist() == [2, 3]
    bad = {(0, 0): _unit([2, 3], [0, 1]), (0, 1): _unit([1, 3], [0, 2])}
    with pytest.raises(AssertionError):
        merge_arm(bad, shape=(1, 4, 2), n_fold=2)


def test_changed_share():
    a = np.array([[0, 1], [2, 3]])
    b = np.array([[0, 2], [2, 4]])
    assert changed_share(a, b) == 0.5
