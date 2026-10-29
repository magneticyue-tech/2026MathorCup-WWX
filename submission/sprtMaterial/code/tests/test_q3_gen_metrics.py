import numpy as np
from src.q3.gen_metrics import acceptance, c2st, c2st_perm, dcr, dcr_ratio, floor_metrics


def test_c2st_same_and_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(100, 5)), rng.normal(size=(100, 5))
    assert abs(c2st(a, b)["auc"] - 0.5) <= 0.1
    assert c2st(a, b + 3)["auc"] > 0.9


def test_perm_null():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
    n1 = c2st_perm(a, b, None, None, n_perm=8)
    n2 = c2st_perm(a, b, None, None, n_perm=8)
    assert np.array_equal(n1, n2)
    assert abs(n1.mean() - 0.5) <= 0.08


def test_dcr_and_floor():
    rng = np.random.default_rng(2)
    tr, real = rng.normal(size=(30, 4)), rng.normal(size=(20, 4))
    assert dcr_ratio(dcr(tr[:10], tr), dcr(real, tr)) == 0
    x = np.r_[np.zeros(10), rng.normal(size=10)][:, None]
    assert floor_metrics(x, x, x.min(0))[0] == 0


def test_acceptance():
    good = {
        "main": dict(
            auc_marg=0.51,
            auc_cond=0.52,
            dcr_ratio=1.0,
            near_copy=0.05,
            tstr=0.5,
            floor_err=0.02,
            module_rho=0.7,
            wcorr_err=0.05,
        ),
        "G0": dict(auc_marg=0.6, auc_cond=0.6, tstr=0.1),
        "G1": dict(floor_err=0.015, wcorr_err=0.04),
        "trtr": 0.55,
        "real_module_rho": 0.72,
    }
    assert all(acceptance(good).values())
    bad = {**good, "main": {**good["main"], "auc_marg": 0.6, "dcr_ratio": 0.5}}
    r = acceptance(bad)
    assert not r["a"] and not r["b"] and r["c"]
