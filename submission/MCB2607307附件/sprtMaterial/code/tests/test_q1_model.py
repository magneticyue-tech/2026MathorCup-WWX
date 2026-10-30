import numpy as np
import pytest
from src.q1.data import ChemNAEncoder, load_q1
from src.q1.features import MONOTONE_PAIRS, assert_pair_features, monotone_cst
from src.q1.model import (
    CONFIGS,
    ITERS,
    fit_outer_fold,
    fit_predict_path,
    make_hgb,
    nested_cv_predict,
    tune_group,
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
def setup():
    d = load_q1()
    groups = d.targets.date_group
    splits = outer_splits(groups, d.style)
    cols = [d.dims.index(x) for x in DIMS2]
    Y = d.targets.xbar[:, cols]
    return d, groups, splits, Y


def test_pair_features_and_cst(setup):
    d = setup[0]
    names = list(d.X.columns)
    assert_pair_features(names)
    with pytest.raises(KeyError):
        assert_pair_features([n for n in names if n != "pH"])
    assert "diacetyl" not in MONOTONE_PAIRS

    cst = monotone_cst(names, "alcohol", True)
    assert [names[i] for i, v in enumerate(cst) if v != 0] == ["ethanol..v.v."]
    assert cst[names.index("ethanol..v.v.")] == 1
    cst = monotone_cst(names, "acidity", True)
    assert cst[names.index("pH")] == -1 and sum(abs(v) for v in cst) == 1
    assert not any(monotone_cst(names, "diacetyl", True))
    assert not any(monotone_cst(names, "alcohol", False))


def test_warm_start_equivalence(setup):
    d, _, splits, _ = setup
    tr, te = splits[0][0]
    enc = ChemNAEncoder("col_mean").fit(d.X.iloc[tr])
    Xtr = enc.transform(d.X.iloc[tr]).to_numpy(float)
    Xte = enc.transform(d.X.iloc[te]).to_numpy(float)
    y = d.targets.xbar[tr, d.dims.index("alcohol")]
    names = list(d.X.columns)
    cst = monotone_cst(names, "alcohol", True)
    base = {"max_depth": 3, "min_samples_leaf": 10, "learning_rate": 0.05, "l2_regularization": 1.0}
    path = fit_predict_path(Xtr, y, [Xte], base, cst)
    assert len(path) == len(ITERS)
    for it, preds in zip(ITERS, path):
        ref = make_hgb({**base, "max_iter": it}, cst).fit(Xtr, y).predict(Xte)
        np.testing.assert_allclose(preds[0], ref, atol=1e-10, rtol=0)


def _run(d, groups, Y, tr, te):
    return fit_outer_fold(
        d.X, Y, DIMS2, GROUPS2, groups, d.style, tr, te, inner_seed=100, configs=SMALL
    )


def test_no_leakage(setup):
    d, groups, splits, Y = setup
    tr, te = splits[0][0]
    a = _run(d, groups, Y, tr, te)
    Y2 = Y.copy()
    Y2[te] = np.random.default_rng(1).uniform(0, 6, size=Y2[te].shape)
    b = _run(d, groups, Y2, tr, te)
    assert a["chosen"] == b["chosen"]
    np.testing.assert_array_equal(a["scores"]["Basic"], b["scores"]["Basic"])
    np.testing.assert_array_equal(a["inner"], b["inner"])
    np.testing.assert_array_equal(a["oof"], b["oof"])

    Y3 = Y.copy()
    Y3[tr] += 1.0
    c = _run(d, groups, Y3, tr, te)
    assert not np.allclose(a["oof"], c["oof"])


def test_merge_inner_rows(setup):
    d, groups, splits, Y = setup
    res = nested_cv_predict(
        d.X, Y, DIMS2, GROUPS2, groups, d.style, splits, repeats=[0], folds=[0, 1], configs=SMALL
    )
    for k in (0, 1):
        tr, te = splits[0][k]
        filled = np.where(~np.isnan(res.inner_mu[0, k]).any(1))[0]
        assert np.array_equal(filled, np.sort(tr))
        assert np.isnan(res.inner_mu[0, k][te]).all()
        assert not np.isnan(res.oof_mu[0, te]).any()
        assert (res.fold_id[0, te] == k).all()


def test_tune_group_excludes_constant_validation_cells():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.column_stack([X[:, 0] + rng.normal(scale=0.1, size=60), X[:, 1]])
    blocks = np.split(np.arange(60), 3)
    y[blocks[2], 1] = 0.0
    inner = []
    for b in blocks:
        a = np.setdiff1d(np.arange(60), b)
        inner.append((X[a], X[b], a, b))
    with np.errstate(divide="raise", invalid="raise"):
        best, scores, pred = tune_group(
            inner, y, ["a", "b", "c", "d"], ["d0", "d1"], "Aroma", SMALL
        )
    assert np.isfinite(scores[0]).all() and np.isnan(scores[1]).all()
    ref = []
    for c in SMALL:
        cells = []
        for f, (Xa, Xb, a, b) in enumerate(inner):
            for j in (0, 1):
                if j == 1 and f == 2:
                    continue
                p = make_hgb(c, None).fit(Xa, y[a, j]).predict(Xb)
                cells.append(np.sqrt(np.mean((p - y[b, j]) ** 2)) / np.std(y[b, j]))
        ref.append(np.mean(cells))
    np.testing.assert_allclose(scores[0], ref, rtol=0, atol=1e-12)
    assert not np.isnan(pred).any()
