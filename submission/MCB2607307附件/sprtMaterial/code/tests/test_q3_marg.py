import numpy as np
import pytest
from src.q3.marg import Marg
from src.q3.modules import modules_from_corr


def _data(seed=0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(80, 4))
    M[:20, 0] = M[:, 0].min() - 1
    M[5, 1] = np.nan
    return M


@pytest.mark.parametrize("scheme", ["cond_mean", "mean_rank"])
def test_roundtrip(scheme):
    M = _data()
    mg = Marg(scheme).fit(M)
    Mf = np.where(np.isnan(M), np.nanmean(M, 0), M)
    back = mg.inverse(mg.transform(Mf))
    obs = ~np.isnan(M)
    assert np.abs(back[obs] - M[obs]).max() <= 1e-9


def test_floor_atom():
    M = _data()
    mg = Marg().fit(M)
    lo = 0.0 + 1e-6
    hi = 20 / 80 - 1e-6
    from scipy.stats import norm

    u = norm.ppf(np.linspace(lo, hi, 50))
    assert np.allclose(mg.inverse_column(0, u), M[:, 0].min())


def test_inverse_range_and_nan():
    M = _data()
    mg = Marg().fit(M)
    U = np.random.default_rng(1).normal(scale=3, size=(500, 4))
    X = mg.inverse(U)
    assert (X >= np.nanmin(M, 0) - 1e-12).all() and (X <= np.nanmax(M, 0) + 1e-12).all()
    with pytest.raises(ValueError):
        mg.transform(M)


def test_inverse_column_matches():
    M = _data()
    mg = Marg().fit(M)
    U = np.random.default_rng(2).normal(size=(100, 4))
    for j in range(4):
        assert np.array_equal(mg.inverse_column(j, U[:, j]), mg.inverse(U)[:, j])


def test_modules_two_blocks():
    C = np.full((6, 6), 0.1)
    C[:3, :3] = 0.9
    C[3:, 3:] = 0.9
    np.fill_diagonal(C, 1)
    lab = modules_from_corr(C, 0.6)
    assert len(np.unique(lab)) == 2
