import numpy as np
import pandas as pd
import pytest
from src.q1 import external
from src.q1.external import (
    EXPECTED_BASE_CODES,
    EXPECTED_CONTROL_N,
    check_md5,
    compare_rows,
    control_means,
    load_control_ratings,
    match_base_beers,
    resolve_workbook,
)
from src.q1.paths import RAW_DIR


def _workbook():
    try:
        return resolve_workbook()
    except FileNotFoundError:
        return None


def test_resolve_workbook_missing(tmp_path):
    env = {"ZENODO_XLSX": str(tmp_path / "nope.xlsx")}
    with pytest.raises(FileNotFoundError) as e:
        resolve_workbook(env)
    assert "https://doi.org/10.5281/zenodo.10653704" in str(e.value)
    with pytest.raises(FileNotFoundError):
        resolve_workbook({}, raw_dir=tmp_path)


def test_check_md5_mismatch(tmp_path):
    f = tmp_path / "x.xlsx"
    f.write_bytes(b"abc")
    with pytest.raises(RuntimeError):
        check_md5(f)


def _ctrl():
    rows = []
    vals = {"Blond": [2, 4], "NABLAB": [1, 2, 3]}
    for style, vs in vals.items():
        for v in vs:
            rows.append({"BeerStyle": style, **{a: v for a in external.ATT}})
    return pd.DataFrame(rows)


def test_control_means():
    t = control_means(_ctrl(), "blond")
    assert len(t) == 11 and set(t["dim"]) == {d for ds in external.ATT_TO_DIMS.values() for d in ds}
    r = t[t["dim"] == "A_malt_all"].iloc[0]
    assert (
        r["n"] == 2
        and r["mean"] == 3.0
        and abs(r["se"] - np.std([2, 4], ddof=1) / np.sqrt(2)) < 1e-12
    )
    t2 = control_means(_ctrl(), "nablab")
    assert t2["n"].iloc[0] == 3 and t2["mean"].iloc[0] == 2.0


@pytest.mark.parametrize("source", ["main", "chemstyle"])
def test_compare_rows(source):
    dims = ["A_malt_all", "body"]
    table = pd.DataFrame(
        {
            "dim": dims,
            "attr": ["Malt", "Fullness (body)"],
            "n": [3, 3],
            "mean": [2.4, 0.6],
            "se": [0.1, 0.2],
        }
    )
    shift = 0 if source == "main" else 1
    G_hat = np.array([[0, 0], [3 + shift, 0]])
    G_hat_Q = G_hat.copy()
    G_obs = np.array([[0, 0], [2, 1]])
    xbar = np.array([[0.0, 0.0], [2.2, 0.9]])
    p = np.zeros((2, 2, 7))
    p[1, 0, 2], p[1, 1, 1] = 0.4, 0.7
    df = compare_rows(64, [1], table, G_hat, G_hat_Q, G_obs, xbar, p, dims)
    assert df["ctrl_grade"].tolist() == [2, 1]
    assert df["err_hat"].tolist() == [1 + shift, -1]
    assert df["err_obs"].tolist() == [0, 0]
    assert df["p_hat_at_ctrl"].tolist() == [0.4, 0.7]


@pytest.mark.skipif(_workbook() is None, reason="Zenodo 工作簿不存在")
def test_real_workbook():
    path = _workbook()
    assert check_md5(path) == external.EXPECTED_MD5
    cal = pd.read_excel(RAW_DIR / "8.xlsx", sheet_name="Calibrated chemical dataset")
    m = match_base_beers(path, cal)
    assert {k: v["code"] for k, v in m.items()} == EXPECTED_BASE_CODES
    ctrl, _ = load_control_ratings(path)
    for style, n in EXPECTED_CONTROL_N.items():
        assert (ctrl["BeerStyle"].str.lower() == style.lower()).sum() == n
