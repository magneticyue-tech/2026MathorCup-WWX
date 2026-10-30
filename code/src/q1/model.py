from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import DIM_GROUPS, ChemNAEncoder
from .features import MONOTONE_PAIRS, assert_pair_features, monotone_cst
from .paths import SEED
from .splits import inner_splits

ITERS = (100, 200, 400)
CONFIGS = [
    {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "learning_rate": 0.05,
        "l2_regularization": l2_regularization,
        "max_iter": max_iter,
    }
    for max_depth in (2, 3)
    for min_samples_leaf in (10, 20)
    for l2_regularization in (0.0, 1.0)
    for max_iter in ITERS
]
PARAM_KEYS = ("max_depth", "min_samples_leaf", "learning_rate", "l2_regularization", "max_iter")
MONO_OPTIONS = (False, True)
GROUPS_WITH_MONO = frozenset(
    g for g, ds in DIM_GROUPS.items() if any(d in MONOTONE_PAIRS for d in ds)
)


def make_hgb(params, cst, warm_start=False):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=False,
        random_state=SEED,
        monotonic_cst=cst,
        warm_start=warm_start,
        **params,
    )


def fit_predict_path(X_tr, y_tr, X_pred_list, params_base, cst, iters=ITERS):
    model = make_hgb({**params_base, "max_iter": iters[0]}, cst, warm_start=True)
    out = []
    for it in iters:
        model.set_params(max_iter=it)
        model.fit(X_tr, y_tr)
        out.append([model.predict(X) for X in X_pred_list])
    return out


def _config_paths(configs):
    paths = {}
    for ci, c in enumerate(configs):
        base = tuple((k, c[k]) for k in PARAM_KEYS if k != "max_iter")
        paths.setdefault(base, []).append((ci, c["max_iter"]))
    return [(dict(base), members) for base, members in paths.items()]


def tune_group(inner_data, Y_cols, feature_names, dim_names, group, configs=CONFIGS):
    n_train, n_dims = Y_cols.shape
    n_folds, n_cfg = len(inner_data), len(configs)
    monos = MONO_OPTIONS if group in GROUPS_WITH_MONO else (False,)
    paths = _config_paths(configs)
    fold_scores = np.full((len(MONO_OPTIONS), n_cfg, n_folds, n_dims), np.nan)
    preds = np.full((len(MONO_OPTIONS), n_cfg, n_train, n_dims), np.nan)

    for mi, mono in enumerate(MONO_OPTIONS):
        if mono not in monos:
            continue
        for j, dim in enumerate(dim_names):
            cst = monotone_cst(feature_names, dim, mono)
            for f, (X_tr, X_va, tr_pos, va_pos) in enumerate(inner_data):
                y_tr, y_va = Y_cols[tr_pos, j], Y_cols[va_pos, j]
                sd = np.std(y_va, ddof=0)
                for base, members in paths:
                    iters = tuple(sorted({it for _, it in members}))
                    path = fit_predict_path(X_tr, y_tr, [X_va], base, cst, iters)
                    for ci, it in members:
                        p = path[iters.index(it)][0]
                        if sd > 0:
                            rmse = np.sqrt(np.mean((p - y_va) ** 2))
                            fold_scores[mi, ci, f, j] = rmse / sd
                        preds[mi, ci, va_pos, j] = p

    # 验证行目标恒定时 RMSE/SD 无定义，该单元对所有配置一致剔除。
    defined = ~np.isnan(fold_scores)
    n_defined = defined.sum(axis=(2, 3))
    total = np.where(defined, fold_scores, 0.0).sum(axis=(2, 3))
    scores = np.where(n_defined > 0, total / np.maximum(n_defined, 1), np.nan)
    mi, ci = np.unravel_index(np.nanargmin(scores), scores.shape)
    best = {**configs[ci], "mono": bool(MONO_OPTIONS[mi])}
    inner_pred = preds[mi, ci]
    assert not np.isnan(inner_pred).any(), "内层验证集未覆盖全部外层训练行"
    return best, scores, inner_pred


def _encode_inner(X_raw, folds, base_idx, scheme):
    data = []
    for a, b in folds:
        enc = ChemNAEncoder(scheme).fit(X_raw.iloc[a])
        X_tr = enc.transform(X_raw.iloc[a]).to_numpy(np.float64)
        X_va = enc.transform(X_raw.iloc[b]).to_numpy(np.float64)
        data.append((X_tr, X_va, np.searchsorted(base_idx, a), np.searchsorted(base_idx, b)))
    return data


def _fit_dim(X_tr, y_tr, feature_names, dim, best):
    params = {k: best[k] for k in PARAM_KEYS}
    return make_hgb(params, monotone_cst(feature_names, dim, best["mono"])).fit(X_tr, y_tr)


def fit_outer_fold(
    X_raw,
    Y,
    dims,
    dim_groups,
    groups,
    style,
    train_idx,
    test_idx,
    inner_seed,
    configs=CONFIGS,
    scheme="col_mean",
):
    tr, te = np.sort(train_idx), np.sort(test_idx)
    dims = list(dims)
    inner_data = _encode_inner(X_raw, inner_splits(tr, groups, style, inner_seed), tr, scheme)

    enc = ChemNAEncoder(scheme).fit(X_raw.iloc[tr])
    X_tr_df = enc.transform(X_raw.iloc[tr])
    feature_names = list(X_tr_df.columns)
    assert_pair_features(feature_names)
    X_tr = X_tr_df.to_numpy(np.float64)
    X_te = enc.transform(X_raw.iloc[te]).to_numpy(np.float64)

    oof = np.full((len(te), len(dims)), np.nan)
    inner = np.full((len(tr), len(dims)), np.nan)
    chosen, scores = {}, {}
    for group, group_dims in dim_groups.items():
        cols = [dims.index(d) for d in group_dims]
        best, sc, ipred = tune_group(
            inner_data, Y[tr][:, cols], feature_names, group_dims, group, configs
        )
        chosen[group], scores[group] = best, sc
        inner[:, cols] = ipred
        for d, j in zip(group_dims, cols):
            oof[:, j] = _fit_dim(X_tr, Y[tr, j], feature_names, d, best).predict(X_te)
    return {
        "train_idx": tr,
        "test_idx": te,
        "oof": oof,
        "inner": inner,
        "chosen": chosen,
        "scores": scores,
    }


@dataclass
class NestedCVResult:
    oof_mu: np.ndarray
    inner_mu: np.ndarray
    fold_id: np.ndarray
    chosen: dict
    scores: dict


def merge_units(units, shape) -> NestedCVResult:
    R, K, n, d = shape
    oof_mu = np.full((R, n, d), np.nan)
    inner_mu = np.full((R, K, n, d), np.nan)
    fold_id = np.full((R, n), -1, dtype=int)
    writes = np.zeros((R, n), dtype=int)
    chosen, scores = {}, {}
    for (r, k), u in sorted(units.items()):
        tr, te = u["train_idx"], u["test_idx"]
        oof_mu[r, te] = u["oof"]
        fold_id[r, te] = k
        writes[r, te] += 1
        inner_mu[r, k, tr] = u["inner"]
        n_filled = int((~np.isnan(inner_mu[r, k])).any(axis=1).sum())
        assert n_filled == len(tr), f"单元 {(r, k)} 的 inner 填入 {n_filled} 行，训练行 {len(tr)}"
        for g in u["chosen"]:
            chosen[f"{r}_{k}_{g}"] = u["chosen"][g]
            scores[f"{r}_{k}_{g}"] = u["scores"][g]
    for r in range(R):
        assert writes[r].max() <= 1, f"重复 {r} 有啤酒的 oof_mu 被写入多次"
        if sum(1 for rr, _ in units if rr == r) == K:
            assert (writes[r] == 1).all(), f"重复 {r} 有啤酒的 oof_mu 未被写入"
    return NestedCVResult(oof_mu, inner_mu, fold_id, chosen, scores)


def nested_cv_predict(
    X_raw,
    Y,
    dims,
    dim_groups,
    groups,
    style,
    splits,
    repeats=None,
    folds=None,
    configs=CONFIGS,
    seed=SEED,
) -> NestedCVResult:
    R, K = len(splits), len(splits[0])
    repeats = range(R) if repeats is None else repeats
    folds = range(K) if folds is None else folds
    units = {}
    for r in repeats:
        for k in folds:
            tr, te = splits[r][k]
            units[(r, k)] = fit_outer_fold(
                X_raw, Y, dims, dim_groups, groups, style, tr, te, seed + 100 + r, configs
            )
    return merge_units(units, (R, K, len(X_raw), len(dims)))


def refit_full(X_raw, Y, dims, dim_groups, groups, style, configs=CONFIGS, seed=SEED) -> dict:
    n = len(X_raw)
    dims = list(dims)
    idx = np.arange(n)
    folds = inner_splits(idx, groups, style, seed + 200, n_splits=5)
    inner_data = _encode_inner(X_raw, folds, idx, "col_mean")

    encoder = ChemNAEncoder("col_mean").fit(X_raw)
    X_df = encoder.transform(X_raw)
    feature_names = list(X_df.columns)
    assert_pair_features(feature_names)
    X_all = X_df.to_numpy(np.float64)

    chosen, models = {}, {}
    for group, group_dims in dim_groups.items():
        cols = [dims.index(d) for d in group_dims]
        best, _, _ = tune_group(inner_data, Y[:, cols], feature_names, group_dims, group, configs)
        chosen[group] = best
        for d, j in zip(group_dims, cols):
            models[d] = _fit_dim(X_all, Y[:, j], feature_names, d, best)
    return {"encoder": encoder, "feature_names": feature_names, "chosen": chosen, "models": models}
