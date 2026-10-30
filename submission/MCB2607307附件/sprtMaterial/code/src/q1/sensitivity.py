import numpy as np
import pandas as pd

from .data import DIMS, ChemNAEncoder, additive_effects
from .discretize import apply_rules, dist_interval, dist_median
from .features import monotone_cst
from .model import PARAM_KEYS, make_hgb
from .paths import ROOT
from .splits import inner_splits

SUM_COLS = ("esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum")
ARMS = (
    "na_median_ind",
    "na_native_nan",
    "na_floor_ind",
    "drop_sum",
    "att8",
    "mono_on",
    "mono_off",
    "alpha",
)
DISCORDANCE_CSV = (
    ROOT / "experimentalCode" / "results" / "data_audit" / "calibrated_discordances.csv"
)
ATT8_EXPECTED = (80, 24)


def alpha_target(ratings, train_idx, dims=DIMS) -> np.ndarray:
    dims = list(dims)
    out = np.full((int(ratings["row"].max()), len(dims)), np.nan)
    rows = np.sort(np.asarray(train_idx)) + 1
    alpha, _ = additive_effects(ratings, rows=rows, dims=dims)
    out[rows - 1] = alpha.loc[rows, dims].to_numpy()
    return out


def apply_att8(X_raw, beer):
    disc = pd.read_csv(DISCORDANCE_CSV, encoding="utf-8-sig")
    out = X_raw.copy()
    codes = np.asarray(beer)
    for b, field, lit, cal in disc[["beer", "field", "csv1_literal", "calibrated_8"]].itertuples(
        index=False
    ):
        rows = np.where(codes == b)[0]
        assert len(rows) == 1, f"beer {b} 在附件中出现 {len(rows)} 次"
        cur = out.iloc[rows[0]][field]
        assert (
            abs(cur - lit) <= 1e-6
        ), f"beer {b} 的 {field} 当前值 {cur} 与 csv1_literal {lit} 不符"
        out.iloc[rows[0], out.columns.get_loc(field)] = cal
    counts = {"n_cells": int(len(disc)), "n_beers": int(disc["beer"].nunique())}
    assert (counts["n_cells"], counts["n_beers"]) == tuple(ATT8_EXPECTED), counts
    return out, counts


def mono_flag(mono_mode, best) -> bool:
    if mono_mode == "chosen":
        return bool(best["mono"])
    if mono_mode == "on":
        return True
    if mono_mode == "off":
        return False
    raise ValueError(f"未知 mono_mode: {mono_mode}")


def fit_fold_fixed(
    X_raw,
    Y_fit,
    dims,
    dim_groups,
    groups,
    style,
    train_idx,
    test_idx,
    inner_seed,
    chosen,
    scheme="col_mean",
    mono_mode="chosen",
    drop_cols=(),
) -> dict:
    tr, te = np.sort(train_idx), np.sort(test_idx)
    dims = list(dims)
    X_use = X_raw.drop(columns=list(drop_cols))
    group_of = {d: g for g, ds in dim_groups.items() for d in ds}

    def fit_dims(a, b_rows, out, pos):
        enc = ChemNAEncoder(scheme).fit(X_use.iloc[a])
        A = enc.transform(X_use.iloc[a])
        names = list(A.columns)
        A = A.to_numpy(np.float64)
        B = enc.transform(X_use.iloc[b_rows]).to_numpy(np.float64)
        for j, d in enumerate(dims):
            best = chosen[group_of[d]]
            params = {k: best[k] for k in PARAM_KEYS}
            cst = monotone_cst(names, d, mono_flag(mono_mode, best))
            out[pos, j] = make_hgb(params, cst).fit(A, Y_fit[a, j]).predict(B)

    inner = np.full((len(tr), len(dims)), np.nan)
    for a, b in inner_splits(tr, groups, style, inner_seed):
        fit_dims(a, b, inner, np.searchsorted(tr, b))
    oof = np.full((len(te), len(dims)), np.nan)
    fit_dims(tr, te, oof, np.arange(len(te)))
    return {"train_idx": tr, "test_idx": te, "oof": oof, "inner": inner}


def arm_kwargs(arm, X_raw, beer):
    if arm == "identity" or arm in ("mono_on", "mono_off", "alpha"):
        kw = {"mono_on": {"mono_mode": "on"}, "mono_off": {"mono_mode": "off"}}.get(arm, {})
        return X_raw, kw
    if arm == "na_median_ind":
        return X_raw, {"scheme": "median_ind"}
    if arm == "na_native_nan":
        return X_raw, {"scheme": "native_nan"}
    if arm == "na_floor_ind":
        return X_raw, {"scheme": "floor_ind"}
    if arm == "drop_sum":
        assert set(SUM_COLS) <= set(X_raw.columns), "化学表缺少 _sum 列"
        return X_raw, {"drop_cols": SUM_COLS}
    if arm == "att8":
        return apply_att8(X_raw, beer)[0], {}
    raise ValueError(f"未知臂: {arm}")


def unit_grades(mu_te, mu_inner, y_tr) -> dict:
    n, D = mu_te.shape
    out = {k: np.empty((n, D), dtype=int) for k in ("median", "round", "quant")}
    out["p"] = np.empty((n, D, 7))
    for d in range(D):
        r = apply_rules(mu_te[:, d], mu_inner[:, d], y_tr[:, d])
        for k in ("median", "round", "quant", "p"):
            out[k][:, d] = r[k]
    return out


def changed_share(g_a, g_b) -> float:
    return float(np.mean(np.asarray(g_a) != np.asarray(g_b)))


def merge_arm(units, shape=(5, 250, 50), n_fold=5) -> dict:
    R, n, D = shape
    out = {
        "oof_mu": np.full((R, n, D), np.nan),
        "inner_mu": np.full((R, n_fold, n, D), np.nan),
        "p": np.full((R, n, D, 7), np.nan),
    }
    for k in ("median", "round", "quant"):
        out[f"grade_{k}"] = np.full((R, n, D), -1, dtype=int)
    writes = np.zeros((R, n), dtype=int)
    for (r, k), u in sorted(units.items()):
        tr, te = u["train_idx"], u["test_idx"]
        out["oof_mu"][r, te] = u["oof"]
        out["inner_mu"][r, k, tr] = u["inner"]
        out["p"][r, te] = u["p"]
        for g in ("median", "round", "quant"):
            out[f"grade_{g}"][r, te] = u[g]
        writes[r, te] += 1
        filled = np.where(~np.isnan(out["inner_mu"][r, k]).any(1))[0]
        assert np.array_equal(filled, np.sort(tr)), f"单元 {(r, k)} 的 inner 非 NaN 行不等于训练行"
    assert writes.max() <= 1, "有啤酒在同一次重复中被写入多次"
    for r in range(R):
        if sum(1 for rr, _ in units if rr == r) == n_fold:
            assert (writes[r] == 1).all(), f"重复 {r} 有啤酒未被测试"
    return out


def delivered(p_runs) -> dict:
    p_final = p_runs.mean(0)
    lo, hi = dist_interval(p_final)
    return {"p_final": p_final, "G_hat": dist_median(p_final), "lo": lo, "hi": hi}
