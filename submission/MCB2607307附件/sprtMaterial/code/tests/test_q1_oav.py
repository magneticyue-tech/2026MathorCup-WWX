import numpy as np
import pandas as pd
import pytest
from src.q1.data import load_q1
from src.q1.model import CONFIGS, fit_outer_fold
from src.q1.oav import (
    OAV_CLASSES,
    exceed_count,
    fit_fold_oav,
    inner_ratio,
    load_thresholds,
    oav_class_sums,
)
from src.q1.paths import RAW_DIR
from src.q1.sensitivity import fit_fold_fixed
from src.q1.splits import outer_splits

SMALL = [
    c
    for c in CONFIGS
    if c["max_depth"] == 2 and c["min_samples_leaf"] == 10 and c["l2_regularization"] == 0.0
]
DIMS2 = ["alcohol", "body"]
GROUPS2 = {"Basic": DIMS2}


def test_load_thresholds():
    t = load_thresholds()
    assert len(t) == 43
    assert "DES" in t and "ethyl.butyrate" in t
    assert t["DES"][1] is True and t["ethyl.butyrate"][1] is False


def test_oav_class_sums_toy():
    X = pd.DataFrame({"e1": [2.0], "e2": [3.0], "s1": [1.0]})
    thr = {"e1": (1.0, False), "e2": (3.0, False), "s1": (5.0, True)}
    out = oav_class_sums(X, thr, classes={"esters": ("e1", "e2"), "sulfur": ("s1",)})
    assert abs(out["oav_esters"].iloc[0] - 3.0) < 1e-12
    assert abs(out["oav_sulfur"].iloc[0] - 2.0) < 1e-12
    with pytest.raises(KeyError):
        oav_class_sums(
            X.drop(columns="e2"), thr, classes={"esters": ("e1", "e2"), "sulfur": ("s1",)}
        )
    Xn = X.copy()
    Xn.loc[0, "e1"] = np.nan
    with pytest.raises(ValueError):
        oav_class_sums(Xn, thr, classes={"esters": ("e1", "e2"), "sulfur": ("s1",)})


def test_classes():
    e, s = OAV_CLASSES["esters"], OAV_CLASSES["sulfur"]
    assert len(e) == 13 and len(s) == 4 and not set(e) & set(s)
    cols = set(pd.read_csv(RAW_DIR / "1.csv", nrows=0).columns)
    thr = load_thresholds()
    assert set(e) | set(s) <= cols and set(e) | set(s) <= set(thr)
    assert "ethyl.nicotinate" not in e + s and "SO2.mg.L." not in e + s


@pytest.fixture(scope="module")
def real():
    d = load_q1()
    splits = outer_splits(d.targets.date_group, d.style)
    tr, te = splits[0][0]
    Y = d.targets.xbar[:, [d.dims.index(x) for x in DIMS2]]
    groups = d.targets.date_group
    ref = fit_outer_fold(d.X, Y, DIMS2, GROUPS2, groups, d.style, tr, te, 100, SMALL)
    return d, tr, te, Y, groups, ref["chosen"], load_thresholds()


def _run(d, Y, groups, tr, te, chosen, thr, add):
    return fit_fold_oav(d.X, Y, DIMS2, GROUPS2, groups, d.style, tr, te, 100, chosen, thr, add)


def test_fit_fold_oav_identity_and_leakage(real):
    d, tr, te, Y, groups, chosen, thr = real
    a = _run(d, Y, groups, tr, te, chosen, thr, False)
    ref = fit_fold_fixed(d.X, Y, DIMS2, GROUPS2, groups, d.style, tr, te, 100, chosen)
    np.testing.assert_allclose(a["oof"], ref["oof"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(a["inner"], ref["inner"], atol=1e-10, rtol=0)
    b = _run(d, Y, groups, tr, te, chosen, thr, True)
    Y2 = Y.copy()
    Y2[te] = np.random.default_rng(1).uniform(0, 6, size=Y2[te].shape)
    b2 = _run(d, Y2, groups, tr, te, chosen, thr, True)
    np.testing.assert_array_equal(b["oof"], b2["oof"])
    np.testing.assert_array_equal(b["inner"], b2["inner"])
    Y3 = Y.copy()
    Y3[tr] += 1.0
    b3 = _run(d, Y3, groups, tr, te, chosen, thr, True)
    assert not np.allclose(b["oof"], b3["oof"])
    assert not np.array_equal(a["oof"], b["oof"])


def test_inner_ratio():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(20, 3))
    y[:, 2] = 1.0
    tr = np.arange(20)
    r = inner_ratio(y.copy(), y, tr)
    assert np.allclose(r[:2], 0.0) and np.isnan(r[2])
    r = inner_ratio(np.tile(y.mean(0), (20, 1)), y, tr)
    assert np.allclose(r[:2], 1.0)


def test_exceed_count_real():
    d = load_q1()
    n = exceed_count(d.X, load_thresholds())
    assert 0 <= n <= 43
