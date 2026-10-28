import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.q1 import paths
from src.q1.data import (
    DIM_GROUPS,
    DIMS,
    NA_COLS,
    SUM_NA_COL,
    ZERO_HEAVY_DIMS,
    ChemNAEncoder,
    additive_effects,
    date_groups,
    grade,
    load_q1,
    load_raw_ratings,
)

G_COUNTS = [4674, 3579, 2057, 1545, 524, 112, 9]


@pytest.fixture(scope="module")
def q1():
    return load_q1()


def test_grade():
    v = np.array([0.5, 5.5, -0.2, 6.4, 0.49])
    assert grade(v).tolist() == [1, 6, 0, 6, 0]


def test_dim_groups_partition_dims():
    assert len(DIMS) == 50
    assert len(DIM_GROUPS) == 6
    members = [d for g in DIM_GROUPS.values() for d in g]
    assert len(members) == len(set(members)) == len(DIMS)
    assert set(members) == set(DIMS)


def test_date_groups_union_and_labels():
    ratings = pd.DataFrame(
        {
            "row": [1, 2, 3, 3, 4, 5],
            "date": ["2020-09", "2020-09", "2020-01", "2020-05", "2020-05", "2020-01"],
        }
    )
    assert date_groups(ratings).tolist() == [0, 0, 1, 1, 1]


def _toy_chem():
    cols = NA_COLS + ["other"]
    train = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in cols})
    test = pd.DataFrame({c: [100.0, 200.0] for c in cols})
    test.loc[0, NA_COLS] = np.nan
    return train, test


def test_col_mean_uses_train_mean():
    train, test = _toy_chem()
    out = ChemNAEncoder("col_mean").fit(train).transform(test)
    assert out.loc[0, NA_COLS].tolist() == [2.5] * len(NA_COLS)
    assert not out.isna().any().any()
    assert list(out.columns) == list(test.columns)


def test_col_mean_all_nan_train_column_raises():
    train, _ = _toy_chem()
    train["other"] = np.nan
    with pytest.raises(ValueError):
        ChemNAEncoder("col_mean").fit(train)


def test_indicator_schemes():
    train, test = _toy_chem()
    med = ChemNAEncoder("median_ind").fit(train).transform(test)
    assert sum(c.endswith("_missing") for c in med.columns) == 9
    assert med.loc[0, "X2.3.butanedione_missing"] == 1
    flo = ChemNAEncoder("floor_ind").fit(train).transform(test)
    assert sum(c.endswith("_missing") for c in flo.columns) == 8
    assert f"{SUM_NA_COL}_missing" not in flo.columns
    assert flo.loc[0, "furfural"] == 1.0
    assert np.isnan(flo.loc[0, SUM_NA_COL])


def test_native_nan_keeps_nan():
    train, test = _toy_chem()
    out = ChemNAEncoder("native_nan").fit(train).transform(test)
    assert out.isna().sum().sum() == len(NA_COLS)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        ChemNAEncoder("main")


def test_additive_effects_recovers_taster_effect():
    alpha_true = {1: 1.0, 2: 2.0, 3: 3.0}
    beta_true = {"t1": 0.5, "t2": -0.5}
    rows = [(b, t, a + beta_true[t]) for b, a in alpha_true.items() for t in beta_true]
    ratings = pd.DataFrame(rows, columns=["row", "taster", "d"])
    alpha, beta = additive_effects(ratings, [1, 2, 3], dims=["d"])
    assert np.allclose(beta.loc[["t1", "t2"], "d"], [0.5, -0.5])
    assert np.allclose(alpha.loc[[1, 2, 3], "d"], [1.0, 2.0, 3.0])
    counts = ratings.groupby("taster").size().loc[beta.index].values
    assert abs((beta["d"].values * counts).sum()) < 1e-10


def test_load_q1_and_grade_counts(q1):
    assert q1.X.shape == (250, 231)
    assert np.bincount(q1.targets.G.ravel(), minlength=7).tolist() == G_COUNTS


def test_zero_heavy_dims_match_g(q1):
    zero_heavy = (q1.targets.G == 0).mean(0) >= 0.5
    assert set(np.array(DIMS)[zero_heavy]) == set(ZERO_HEAVY_DIMS)


def test_formal_npz_alignment(q1):
    z = np.load(paths.PROCESSED_DIR / "q1_data.npz", allow_pickle=False)
    assert z["beer_row"].tolist() == list(range(1, 251))
    assert (z["beer"] == q1.beer.values).all()
    assert (z["style"] == q1.style.values.astype(str)).all()
    counts = load_raw_ratings().groupby("row").size().loc[range(1, 251)].values
    assert (z["n_ratings"] == counts).all()


def _tree(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_import_paths_creates_no_dirs(tmp_path):
    target = tmp_path / "repo" / "code" / "src" / "q1" / "paths.py"
    target.parent.mkdir(parents=True)
    shutil.copy(Path(paths.__file__), target)
    before = _tree(tmp_path)
    spec = importlib.util.spec_from_file_location("paths_copy", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.ROOT == tmp_path / "repo"
    assert _tree(tmp_path) == before


def test_no_main_in_src_q1():
    for f in Path(paths.__file__).parent.glob("*.py"):
        assert "__main__" not in f.read_text(encoding="utf-8"), f.name
