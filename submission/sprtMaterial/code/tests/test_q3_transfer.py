import numpy as np
import pytest
from scipy.optimize import minimize
from src.q3.data import load_chem, load_dictionary
from src.q3.defects import LatentBounder, build_defect_spec, chem_bounds, violation
from src.q3.fullmodel import PRERUN_CONFIG, fit_full
from src.q3.transfer import project


def _pd(n, seed):
    A = np.random.default_rng(seed).normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def test_project_properties():
    C = _pd(6, 0)
    ut = np.array([2.0, 1.5, 0.3, -0.2, 0.8, 0.1])
    idx = np.array([0, 1])
    beta = np.array([0.0, 0.5])
    u, act, lam = project(ut, beta, C, idx)
    assert (u[idx] <= beta + 1e-9).all() and (lam >= -1e-12).all()
    A = np.where(act)[0]
    ja = idx[A]
    ref = ut - C[:, ja] @ np.linalg.solve(C[np.ix_(ja, ja)], ut[ja] - beta[A])
    assert np.abs(u - ref).max() <= 1e-9
    Ci = np.linalg.inv(C)
    sol = minimize(
        lambda x: (x - ut) @ Ci @ (x - ut),
        ut,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda x, k=k: beta[k] - x[idx[k]]} for k in range(2)],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert np.abs(sol.x - u).max() <= 1e-5


def test_project_feasible_and_noconv():
    C = _pd(6, 1)
    ut = np.zeros(6)
    u, act, _ = project(ut, np.ones(2), C, np.array([0, 1]))
    assert np.array_equal(u, ut) and act.sum() == 0
    C = np.eye(3)
    C[0, 1] = C[1, 0] = 0.9
    with pytest.raises(RuntimeError):
        project(np.array([2.0, 0.0, 0.0]), np.array([0.0, -1.0]), C, np.array([0, 1]), max_iter=1)


def test_zero_transfer_and_bounds():
    dic = load_dictionary()
    chem = load_chem(dic=dic)
    gen, _ = fit_full(chem, PRERUN_CONFIG)
    spec = build_defect_spec(chem.gen_cols, dic)
    lb = LatentBounder(gen.mg, spec)
    x0 = gen.mg.inverse(gen.U)
    b = np.array([chem_bounds(x, spec) for x in x0])
    beta = np.array([lb.beta(b[i], gen.U[i]) for i in range(len(b))])
    assert (beta >= gen.U[:, spec.idx]).all()
    assert not violation(x0, b, spec).any()
    obs = ~np.isnan(chem.XG)
    assert np.abs(x0[obs] - chem.XG[obs]).max() <= 1e-9
    s = np.array([chem_bounds(x, spec, "strict") for x in x0[:5]])
    assert np.array_equal(s, x0[:5][:, spec.idx])
    t = b[:5]
    assert (t[:, spec.has_threshold] >= s[:, spec.has_threshold]).all()
    assert np.array_equal(t[:, ~spec.has_threshold], s[:, ~spec.has_threshold])
