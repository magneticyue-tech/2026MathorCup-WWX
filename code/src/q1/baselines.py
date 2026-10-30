import warnings

import numpy as np
import pandas as pd
from scipy.stats import shapiro, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .data import FLOOR_VALUE, ChemNAEncoder
from .paths import SEED
from .splits import inner_splits

B1_K = 5
LIN_CONFIGS = [{}]
EN_CONFIGS = [
    {"alpha": float(a), "l1_ratio": l1} for l1 in (0.1, 0.5, 0.9) for a in np.logspace(-3, 1, 9)
]
RF_CONFIGS = [
    {"n_estimators": 500, "max_features": mf, "min_samples_leaf": leaf}
    for mf in (0.33, "sqrt")
    for leaf in (5, 10)
]
RULES = ("const", "round", "median", "quant", "qwk_cut", "median_knn")
B2_GRID = {"n_estimators": [100, 200, 400], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]}


def b0_mean(y_tr, n_te) -> np.ndarray:
    return np.full(n_te, np.mean(y_tr))


def b1_style_shrunk(y_tr, style_tr, style_te, k=B1_K) -> np.ndarray:
    y_tr = np.asarray(y_tr, float)
    style_tr, style_te = np.asarray(style_tr), np.asarray(style_te)
    g = y_tr.mean()
    out = np.empty(len(style_te))
    for i, s in enumerate(style_te):
        m = style_tr == s
        n_s = int(m.sum())
        out[i] = g if n_s == 0 else (n_s * y_tr[m].mean() + k * g) / (n_s + k)
    return out


def _simple_predict(kind, Y_fit, style_fit, style_pred):
    if kind == "b0":
        return np.tile(Y_fit.mean(0), (len(style_pred), 1))
    if kind == "b1":
        return np.column_stack(
            [b1_style_shrunk(Y_fit[:, d], style_fit, style_pred) for d in range(Y_fit.shape[1])]
        )
    raise ValueError(f"未知简单基线: {kind}")


def simple_fold(kind, Y, style, groups, train_idx, test_idx, inner_seed) -> dict:
    tr, te = np.sort(train_idx), np.sort(test_idx)
    s = pd.Series(style).to_numpy()
    oof = _simple_predict(kind, Y[tr], s[tr], s[te])
    inner = np.full((len(tr), Y.shape[1]), np.nan)
    for a, b in inner_splits(tr, groups, style, inner_seed):
        inner[np.searchsorted(tr, b)] = _simple_predict(kind, Y[a], s[a], s[b])
    return {"train_idx": tr, "test_idx": te, "oof": oof, "inner": inner}


def prep_plain(raw_fit, raw_others):
    enc = ChemNAEncoder("col_mean").fit(raw_fit)
    X_fit = enc.transform(raw_fit).to_numpy(np.float64)
    others = [enc.transform(r).to_numpy(np.float64) for r in raw_others]
    return X_fit, others, list(raw_fit.columns)


def prep_linear(raw_fit, raw_others):
    A_fit, A_others, _ = prep_plain(raw_fit, raw_others)
    qt = QuantileTransformer(
        n_quantiles=min(200, len(raw_fit)), output_distribution="uniform", random_state=SEED
    ).fit(A_fit)
    fc = [c for c in raw_fit.columns if (raw_fit[c].round(8) == FLOOR_VALUE).any()]

    def ind(raw):
        return (raw[fc].round(8) == FLOOR_VALUE).to_numpy(float)

    B_fit = np.hstack([qt.transform(A_fit), ind(raw_fit)])
    B_others = [np.hstack([qt.transform(A), ind(r)]) for A, r in zip(A_others, raw_others)]
    sc = StandardScaler().fit(B_fit)
    names = list(raw_fit.columns) + [f"{c}_floor" for c in fc]
    return sc.transform(B_fit), [sc.transform(B) for B in B_others], names


def _spearman_cols(X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*constant.*")
        rho = np.array([spearmanr(X[:, j], y)[0] for j in range(X.shape[1])], dtype=float)
    return np.nan_to_num(rho, nan=0.0)


def fit_predict_lin10(config, X_tr, y_tr, X_list, names, return_selected=False):
    chem = [i for i, n in enumerate(names) if not n.endswith("_floor")]
    rho = _spearman_cols(X_tr[:, chem], y_tr)
    top = [chem[i] for i in np.argsort(-np.abs(rho), kind="stable")[:10]]
    pos = {n: i for i, n in enumerate(names)}
    cols = top + [pos[f"{names[i]}_floor"] for i in top if f"{names[i]}_floor" in pos]
    model = LinearRegression().fit(X_tr[:, cols], y_tr)
    preds = [model.predict(X[:, cols]) for X in X_list]
    return (preds, top) if return_selected else preds


def fit_predict_enet(config, X_tr, y_tr, X_list, names):
    model = ElasticNet(alpha=config["alpha"], l1_ratio=config["l1_ratio"], max_iter=10000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_tr, y_tr)
    return [model.predict(X) for X in X_list]


def fit_predict_rf(config, X_tr, y_tr, X_list, names):
    model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_features=config["max_features"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=SEED,
        n_jobs=1,
    ).fit(X_tr, y_tr)
    return [model.predict(X) for X in X_list]


def fit_outer_fold_generic(
    X_raw,
    Y,
    dims,
    dim_groups,
    groups,
    style,
    train_idx,
    test_idx,
    inner_seed,
    configs,
    prep,
    fit_predict,
) -> dict:
    tr, te = np.sort(train_idx), np.sort(test_idx)
    dims = list(dims)
    Y_tr = Y[tr]
    folds = []
    for a, b in inner_splits(tr, groups, style, inner_seed):
        X_a, (X_b,), names = prep(X_raw.iloc[a], [X_raw.iloc[b]])
        folds.append((X_a, X_b, names, np.searchsorted(tr, a), np.searchsorted(tr, b)))

    X_tr, (X_te,), names_out = prep(X_raw.iloc[tr], [X_raw.iloc[te]])
    oof = np.full((len(te), len(dims)), np.nan)
    inner = np.full((len(tr), len(dims)), np.nan)
    chosen, scores = {}, {}
    for group, group_dims in dim_groups.items():
        cols = [dims.index(d) for d in group_dims]
        cell = np.full((len(configs), len(folds), len(cols)), np.nan)
        preds = np.full((len(configs), len(tr), len(cols)), np.nan)
        for ci, cfg in enumerate(configs):
            for f, (X_a, X_b, names, pa, pb) in enumerate(folds):
                for jj, j in enumerate(cols):
                    (p,) = fit_predict(cfg, X_a, Y_tr[pa, j], [X_b], names)
                    preds[ci, pb, jj] = p
                    sd = np.std(Y_tr[pb, j], ddof=0)
                    if sd > 0:
                        cell[ci, f, jj] = np.sqrt(np.mean((p - Y_tr[pb, j]) ** 2)) / sd
        defined = ~np.isnan(cell)
        total = np.where(defined, cell, 0.0).sum(axis=(1, 2))
        sc = total / defined.sum(axis=(1, 2))
        best = int(np.argmin(sc))
        chosen[group], scores[group] = dict(configs[best]), sc.tolist()
        inner[:, cols] = preds[best]
        for j in cols:
            (oof[:, j],) = fit_predict(configs[best], X_tr, Y_tr[:, j], [X_te], names_out)
    assert not np.isnan(inner).any(), "内层验证集未覆盖全部外层训练行"
    return {
        "train_idx": tr,
        "test_idx": te,
        "oof": oof,
        "inner": inner,
        "chosen": chosen,
        "scores": scores,
    }


def merge_generic(units, n=250, n_dims=50, n_rep=5, n_fold=5):
    oof = np.full((n_rep, n, n_dims), np.nan)
    inner = np.full((n_rep, n_fold, n, n_dims), np.nan)
    fold_id = np.full((n_rep, n), -1, dtype=int)
    writes = np.zeros((n_rep, n), dtype=int)
    for (r, k), u in sorted(units.items()):
        tr, te = u["train_idx"], u["test_idx"]
        oof[r, te] = u["oof"]
        fold_id[r, te] = k
        writes[r, te] += 1
        inner[r, k, tr] = u["inner"]
        filled = np.where(~np.isnan(inner[r, k]).any(1))[0]
        assert np.array_equal(filled, tr), f"单元 {(r, k)} 的 inner 非 NaN 行不等于训练行"
    for r in range(n_rep):
        assert writes[r].max() <= 1, f"重复 {r} 有啤酒的 oof 被写入多次"
        if sum(1 for rr, _ in units if rr == r) == n_fold:
            assert (writes[r] == 1).all(), f"重复 {r} 有啤酒的 oof 未被写入"
    return oof, inner, fold_id


def chemstyle_input(X, style) -> pd.DataFrame:
    dummies = pd.get_dummies(pd.Series(style).reset_index(drop=True), prefix="style", dtype=float)
    return pd.concat([X.reset_index(drop=True), dummies[sorted(dummies.columns)]], axis=1)


def b2_log_columns(X_tr) -> list[str]:
    out = []
    for c in X_tr.columns:
        v = X_tr[c].dropna().to_numpy(float)
        if len(v) >= 3 and (v > 0).all() and np.ptp(v) > 0 and shapiro(v).pvalue < 0.05:
            out.append(c)
    return out


def b2_prepare(X_tr, X_te):
    cols = b2_log_columns(X_tr)
    A, B = X_tr.copy(), X_te.copy()
    A[cols] = np.log10(A[cols])
    B[cols] = np.log10(B[cols].where(B[cols] > 0))
    mean = A.mean()
    A, B = A.fillna(mean), B.fillna(mean)
    sc = StandardScaler().fit(A.to_numpy(float))
    A_out, B_out = sc.transform(A.to_numpy(float)), sc.transform(B.to_numpy(float))
    assert np.isfinite(A_out).all() and np.isfinite(B_out).all(), "B2 特征含非有限值"
    return A_out, B_out


def b2_fit_dim(X_tr, y_tr, X_te, seed=SEED):
    gs = GridSearchCV(
        GradientBoostingRegressor(random_state=seed),
        B2_GRID,
        cv=KFold(5, shuffle=True, random_state=seed),
        scoring="r2",
        refit=True,
        n_jobs=1,
    ).fit(X_tr, y_tr)
    return gs.predict(X_te), gs.best_params_
