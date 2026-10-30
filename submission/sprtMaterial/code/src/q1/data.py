from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from .panels import infer_panels
from .paths import RAW_DIR, RAW_SENSORY, SEED

DIMS = [
    "A_malt_all",
    "A_malt_grain",
    "A_malt_bread",
    "A_malt_cara",
    "A_malt_burn",
    "A_hops_all",
    "A_hops_citrus",
    "A_hops_tropical",
    "A_hops_noble",
    "A_hops_woody",
    "A_esters_all",
    "A_esters_ethac",
    "A_esters_isoaa",
    "A_esters_flower",
    "A_esters_fruity",
    "F_malt_all",
    "F_malt_grain",
    "F_malt_bread",
    "F_malt_cara",
    "F_malt_burn",
    "F_hops_all",
    "F_hops_citrus",
    "F_hops_tropical",
    "F_hops_noble",
    "F_hops_woody",
    "F_esters_all",
    "F_esters_ethac",
    "F_esters_isoaa",
    "F_esters_flower",
    "F_esters_fruity",
    "acidity",
    "bitternes",
    "sweetness",
    "X4vg",
    "diacetyl",
    "dms",
    "metallic",
    "stale_hops",
    "t2n",
    "orange",
    "coriander",
    "clove",
    "lactic",
    "acetic",
    "barnyard",
    "alcohol",
    "aftertaste",
    "body",
    "co2",
    "overall",
]

DIM_GROUPS = {
    "Aroma": [d for d in DIMS if d.startswith("A_")],
    "Flavour": [d for d in DIMS if d.startswith("F_")],
    "Defect": ["dms", "metallic", "X4vg", "t2n", "diacetyl", "stale_hops"],
    "Spice": ["orange", "coriander", "clove"],
    "Sour": ["lactic", "acetic", "barnyard"],
    "Basic": [
        "body",
        "alcohol",
        "bitternes",
        "sweetness",
        "co2",
        "acidity",
        "aftertaste",
        "overall",
    ],
}
_all_group_dims = [d for g in DIM_GROUPS.values() for d in g]
assert len(_all_group_dims) == len(set(_all_group_dims)) == len(DIMS)
assert set(_all_group_dims) == set(DIMS)

ZERO_HEAVY_DIMS = [
    "A_malt_burn",
    "A_hops_citrus",
    "A_hops_tropical",
    "A_hops_woody",
    "A_esters_flower",
    "F_malt_burn",
    "F_hops_citrus",
    "F_hops_tropical",
    "F_hops_woody",
    "F_esters_flower",
    "acidity",
    "X4vg",
    "diacetyl",
    "metallic",
    "stale_hops",
    "t2n",
    "orange",
    "coriander",
    "clove",
    "lactic",
    "acetic",
    "barnyard",
]

NA_COLS = [
    "X2.3.butanedione",
    "X3.methylbutanal",
    "ethyl.isobutyrate",
    "methylpyrazine",
    "furfural",
    "amyl.acetate",
    "phenylacetaldehyde",
    "ethyl.nicotinate",
    "aroma_hops_sum",
]
SUM_NA_COL = "aroma_hops_sum"
FLOOR_VALUE = -1.81781854

N_BEERS = 250
ROWS = np.arange(1, N_BEERS + 1)


@dataclass
class Q1Targets:
    xbar: np.ndarray
    G: np.ndarray
    n_ratings: np.ndarray
    date_group: np.ndarray


@dataclass
class Q1Data:
    X: pd.DataFrame
    Z: pd.DataFrame
    targets: Q1Targets
    beer: pd.Series
    style: pd.Series
    dims: list


def grade(v) -> np.ndarray:
    return np.floor(np.clip(v, 0, 6) + 0.5).astype(int)


def load_raw_ratings(path=RAW_SENSORY) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp1252")
    df = df.rename(
        columns={
            "A_ester_fruity": "A_esters_fruity",
            "F_ester_fruity": "F_esters_fruity",
            "beer_name": "row",
        }
    )
    df[DIMS] = df[DIMS].astype(float)
    assert len(df) == 2197, len(df)
    assert set(df["row"]) == set(ROWS)
    assert df["taster"].nunique() == 16
    assert df["date"].nunique() == 37
    assert not df[DIMS].isna().any().any()
    assert df[DIMS].isin(range(7)).all().all()
    return df


def date_groups(ratings: pd.DataFrame) -> np.ndarray:
    parent = {d: d for d in ratings["date"].unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for dates in ratings.groupby("row")["date"].unique():
        for d in dates[1:]:
            parent[find(d)] = find(dates[0])

    first_date = ratings.groupby("row")["date"].first()
    roots = first_date.map(find)
    min_row = roots.groupby(roots).apply(lambda s: s.index.min()).sort_values()
    label = {root: k for k, root in enumerate(min_row.index)}
    rows = np.sort(ratings["row"].unique())
    out = np.full(rows.max(), -1, dtype=int)
    out[rows - 1] = roots.loc[rows].map(label).values
    return out


def panel_targets(ratings: pd.DataFrame) -> Q1Targets:
    by_row = ratings.groupby("row")
    xbar = by_row[DIMS].mean().loc[ROWS].values
    return Q1Targets(
        xbar=xbar,
        G=grade(xbar),
        n_ratings=by_row.size().loc[ROWS].values,
        date_group=date_groups(ratings),
    )


def additive_effects(ratings, rows, dims=DIMS, n_iter=500, tol=1e-8):
    sub = ratings[ratings["row"].isin(rows)]
    bi, ub = pd.factorize(sub["row"])
    pi, up = pd.factorize(sub["taster"])
    nb, n_p = len(ub), len(up)
    cb = np.bincount(bi, minlength=nb)
    cp = np.bincount(pi, minlength=n_p)
    X = sub[list(dims)].values
    alpha = np.zeros((nb, len(dims)))
    beta = np.zeros((n_p, len(dims)))
    for _ in range(n_iter):
        old = alpha.copy()
        for j in range(len(dims)):
            alpha[:, j] = np.bincount(bi, X[:, j] - beta[pi, j], nb) / cb
            beta[:, j] = np.bincount(pi, X[:, j] - alpha[bi, j], n_p) / cp
        m = (beta * cp[:, None]).sum(0) / cp.sum()
        beta -= m
        alpha += m
        if np.max(np.abs(alpha - old)) < tol:
            break
    return (
        pd.DataFrame(alpha, index=ub, columns=list(dims)),
        pd.DataFrame(beta, index=up, columns=list(dims)),
    )


def _median_spearman(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.median([spearmanr(A[:, j], B[:, j])[0] for j in range(A.shape[1])]))


def load_q1() -> Q1Data:
    c1 = pd.read_csv(RAW_DIR / "1.csv")
    a4 = pd.read_csv(RAW_DIR / "4.csv")
    X = c1.drop(columns=["beer", "beer_id", "tasting_category_fine"]).astype(float)
    Z = a4[DIMS].astype(float)
    targets = panel_targets(load_raw_ratings())

    assert X.shape == (N_BEERS, 231), X.shape
    assert (c1["beer"].values == a4["beer"].values).all()
    assert list(Z.columns) == DIMS
    assert _median_spearman(targets.xbar, Z.values) >= 0.95
    assert _median_spearman(targets.xbar[a4["beer"].values - 1], Z.values) <= 0.1

    return Q1Data(
        X=X,
        Z=Z,
        targets=targets,
        beer=a4["beer"].copy(),
        style=a4["tasting_category_fine"].copy(),
        dims=list(DIMS),
    )


class ChemNAEncoder:
    SCHEMES = ("col_mean", "median_ind", "native_nan", "floor_ind")

    def __init__(self, scheme: str = "col_mean"):
        if scheme not in self.SCHEMES:
            raise ValueError(f"未知方案: {scheme}")
        self.scheme = scheme

    def fit(self, X_train: pd.DataFrame):
        if self.scheme == "col_mean":
            self.fill_ = X_train.mean()
        elif self.scheme == "median_ind":
            self.fill_ = X_train.median()
        elif self.scheme == "floor_ind":
            self.fill_ = X_train[NA_COLS].min()
        if self.scheme == "col_mean":
            empty = self.fill_.index[self.fill_.isna()].tolist()
            if empty:
                raise ValueError(f"训练折中全为 NaN 的列: {empty}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scheme == "native_nan":
            return X.copy()
        if self.scheme == "col_mean":
            return X.fillna(self.fill_)
        ind_cols = (
            NA_COLS if self.scheme == "median_ind" else [c for c in NA_COLS if c != SUM_NA_COL]
        )
        ind = X[ind_cols].isna().astype(int).add_suffix("_missing")
        if self.scheme == "median_ind":
            out = X.fillna(self.fill_)
        else:
            out = X.copy()
            out[ind_cols] = out[ind_cols].fillna(self.fill_[ind_cols])
        return pd.concat([out, ind], axis=1)


def floor_indicators(X: pd.DataFrame) -> pd.DataFrame:
    hit = np.round(X, 8) == FLOOR_VALUE
    cols = hit.columns[hit.any()]
    return hit[cols].astype(int).add_suffix("_floor")


def check_id_mapping(ratings: pd.DataFrame, xbar: np.ndarray, a4: pd.DataFrame) -> dict:
    A = a4[DIMS].values
    Ac = A - A.mean(1, keepdims=True)
    Bc = xbar - xbar.mean(1, keepdims=True)
    C = (Bc / np.linalg.norm(Bc, axis=1, keepdims=True)) @ (
        Ac / np.linalg.norm(Ac, axis=1, keepdims=True)
    ).T
    _, ci = linear_sum_assignment(-C)
    return {
        "spearman_row": _median_spearman(xbar, A),
        "spearman_beer": _median_spearman(xbar[a4["beer"].values - 1], A),
        "assignment_match": int((ci == np.arange(len(ci))).sum()),
    }


def reconstruct_attachment4(ratings: pd.DataFrame, a4: pd.DataFrame):
    g = ratings.groupby("taster")[DIMS]
    zr = (ratings[DIMS] - g.transform("mean")) / g.transform("std")
    zr["row"] = ratings["row"].values
    zb = zr.groupby("row")[DIMS].mean().loc[ROWS].values
    ok = np.abs(zb - a4[DIMS].values) < 1e-6
    const = g.std() == 0
    return ok, int(const.values.sum()), int(const.any().sum())


def split_half_reliability(ratings: pd.DataFrame, n_splits=20, seed=SEED) -> dict:
    r = ratings.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rel, exact, qwk = [], [], []
    for _ in range(n_splits):
        h = np.zeros(len(r), bool)
        for _, idx in r.groupby("row").groups.items():
            idx = rng.permutation(np.asarray(idx))
            h[idx[: len(idx) // 2]] = True
        m1 = r[h].groupby("row")[DIMS].mean().loc[ROWS].values
        m2 = r[~h].groupby("row")[DIMS].mean().loc[ROWS].values
        rh = np.array([np.corrcoef(m1[:, j], m2[:, j])[0, 1] for j in range(len(DIMS))])
        rel.append(2 * rh / (1 + rh))
        g1, g2 = grade(m1), grade(m2)
        exact.append((g1 == g2).mean())
        k = [
            cohen_kappa_score(g1[:, j], g2[:, j], weights="quadratic", labels=range(7))
            for j in range(len(DIMS))
        ]
        qwk.append(np.nanmean(k))
    return {
        "reliability": np.mean(rel, 0),
        "half_grade_exact": float(np.mean(exact)),
        "half_grade_qwk": float(np.mean(qwk)),
    }


def attendance_share(ratings: pd.DataFrame, xbar: np.ndarray, dims=DIMS) -> np.ndarray:
    alpha, _ = additive_effects(ratings, ROWS, dims)
    a = alpha.loc[ROWS, list(dims)].values
    return np.var(xbar - a, axis=0, ddof=1) / np.var(xbar, axis=0, ddof=1)


def barnyard_check(ratings: pd.DataFrame, xbar: np.ndarray) -> dict:
    same = ratings["barnyard"] == ratings["stale_hops"]
    jb, js = DIMS.index("barnyard"), DIMS.index("stale_hops")
    return {
        "identical_share": float(same.mean()),
        "n_differ": int((~same).sum()),
        "spearman": float(spearmanr(xbar[:, jb], xbar[:, js])[0]),
    }


def panel_date_nesting(Z: pd.DataFrame, date_group: np.ndarray) -> dict:
    panels = infer_panels(Z, k=3)
    labels, counts = np.unique(panels, return_counts=True)
    multi = labels[counts > 1]
    return {
        "n_multi": int(len(multi)),
        "n_covered": int(counts[counts > 1].sum()),
        "n_cross_date": int(sum(len(np.unique(date_group[panels == p])) > 1 for p in multi)),
    }
