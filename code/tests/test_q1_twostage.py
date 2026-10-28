import warnings

import numpy as np
from src.q1.data import load_q1
from src.q1.panels import infer_panels
from src.q1.splits import outer_splits
from src.q1.twostage import (
    c1_fold,
    c2_fold,
    group_zeros,
    local_scale,
    mixed_reference,
    select_c1_map,
)

NAN = np.nan


def test_group_zeros():
    Z = np.array([[0.1, 0.3], [0.1, 0.3], [0.5, 0.9], [0.2, 0.4]])
    dims = ["d0", "F_malt_all"]
    panels = np.array([0, 0, 0, 1])
    out = group_zeros(Z, panels, np.array([0, 1, 2, 3]), dims)
    np.testing.assert_array_equal(out[:3, 0], [0.1, 0.1, 0.1])
    assert np.isnan(out[3, 0]) and np.isnan(out[:, 1]).all()
    assert np.isnan(group_zeros(Z, panels, np.array([0, 2]), dims)).all()
    out = group_zeros(Z, panels, np.array([0, 1]), dims)
    np.testing.assert_array_equal(out[:2, 0], [0.1, 0.1])
    assert np.isnan(out[2, 0])


def test_local_scale():
    panels = np.array([0, 0, 1, 1])
    idx = np.arange(4)
    zeros = np.array([[-0.4], [-0.4], [-0.6], [-0.6]])
    np.testing.assert_allclose(local_scale(zeros, panels, idx), [1.2], atol=1e-12)
    assert np.isnan(local_scale(-zeros, panels, idx)).all()
    assert np.isnan(local_scale(np.full((4, 1), NAN), panels, idx)).all()


def test_mixed_reference():
    z = np.array([[1.0], [0.0]])
    zb = np.array([[0.0], [NAN]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        R, use = mixed_reference(z, zb, np.array([1.2]), np.array([-1.0]), np.array([2.0]))
    assert R.tolist() == [[1], [2]]
    assert use.tolist() == [[True], [False]]


def _setup():
    d = load_q1()
    Z = d.Z.to_numpy(float)
    groups = d.targets.date_group
    splits = outer_splits(groups, d.style)
    panels = infer_panels(d.Z, k=3)
    return d, Z, splits, panels


def test_c1_fold_no_leakage():
    d, Z, splits, panels = _setup()
    tr, te = splits[0][0]
    rng = np.random.default_rng(0)
    oof_z = rng.normal(size=(len(te), 50))
    inner_z = rng.normal(size=(len(tr), 50))
    a = c1_fold(Z, d.targets.xbar, panels, d.dims, tr, te, oof_z, inner_z)
    Z2 = Z.copy()
    Z2[np.sort(te)] = rng.normal(size=(len(te), 50))
    b = c1_fold(Z2, d.targets.xbar, panels, d.dims, tr, te, oof_z, inner_z)
    for k in ("round_T", "qwk_T", "quant_T"):
        np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_array_equal(a["sigma"], b["sigma"])
    Z3 = Z.copy()
    Z3[np.sort(tr)] *= 1.5
    c = c1_fold(Z3, d.targets.xbar, panels, d.dims, tr, te, oof_z, inner_z)
    changed_sigma = not np.array_equal(a["sigma"], c["sigma"], equal_nan=True)
    assert changed_sigma or not np.array_equal(a["round_T"], c["round_T"])


def test_panels_within_test_fold():
    _, _, splits, panels = _setup()
    labels, counts = np.unique(panels, return_counts=True)
    multi = labels[counts > 1]
    assert len(multi) == 41
    for rep in splits:
        for _, te in rep:
            s = set(te)
            for g in multi:
                members = set(np.where(panels == g)[0])
                assert members <= s or not (members & s)


def test_select_c1_map():
    rng = np.random.default_rng(0)
    R = rng.integers(0, 7, size=(5, 30, 4))
    noisy = np.clip(R + rng.integers(-2, 3, size=R.shape), 0, 6)
    name, scores = select_c1_map(R, {"round_T": noisy, "qwk_T": R})
    assert name == "qwk_T" and scores["qwk_T"] > scores["round_T"]
    name, _ = select_c1_map(R, {"round_T": R, "qwk_T": R.copy()})
    assert name == "round_T"


def test_c2_fold():
    d, Z, splits, _ = _setup()
    tr, te = splits[0][0]
    rng = np.random.default_rng(0)
    oof_z = rng.normal(size=(len(te), 50))
    inner_z = rng.normal(size=(len(tr), 50))
    a = c2_fold(d.targets.xbar, Z, tr, te, oof_z, inner_z)
    for g in a["grades"].values():
        assert g.min() >= 0 and g.max() <= 6
    np.testing.assert_allclose(a["p"].sum(-1), 1.0, atol=1e-12)
    Z2 = Z.copy()
    Z2[np.sort(te)] = rng.normal(size=(len(te), 50))
    b = c2_fold(d.targets.xbar, Z2, tr, te, oof_z, inner_z)
    np.testing.assert_array_equal(a["coef"], b["coef"])
