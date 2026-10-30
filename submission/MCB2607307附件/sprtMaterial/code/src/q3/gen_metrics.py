import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .paths import SEED

GEN_NAMES = ("main", "pure_fa", "G0", "G1", "G2", "G3")
NEAR_DUP = ("G1", "G3")
SCALARS = (
    "auc_marg",
    "auc_cond",
    "dcr_ratio",
    "near_copy",
    "trts",
    "tstr",
    "floor_err",
    "floor_row",
    "corr_err",
    "wcorr_err",
    "module_rho",
)
REAL_SCALARS = ("trtr", "real_floor_row", "real_module_rho", "rr_auc_marg", "rr_auc_cond")


def make_rf():
    return RandomForestClassifier(500, random_state=SEED, n_jobs=1)


def onehot(s, styles):
    s = np.asarray(s)
    return (s[:, None] == np.asarray(styles)[None, :]).astype(float)


def _stack(real, gen, style_real, styles):
    X = np.vstack([real, gen])
    if style_real is not None:
        oh = onehot(style_real, styles)
        X = np.hstack([X, np.vstack([oh, oh])])
    y = np.r_[np.zeros(len(real)), np.ones(len(gen))]
    return X, y


def _cv_auc(X, y, seed, n_splits, want_oof=True):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, oof = [], np.full(len(y), np.nan)
    for tr, te in cv.split(X, y):
        p = make_rf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) == 2 else np.nan)
    return float(np.nanmean(aucs)), oof


def c2st(real, gen, style_real=None, styles=None, seed=SEED, n_splits=5) -> dict:
    X, y = _stack(real, gen, style_real, styles)
    auc, oof = _cv_auc(X, y, seed, n_splits)
    return {"auc": auc, "oof": oof, "y": y}


def c2st_perm(real, gen, style_real, styles, n_perm=200, seed=SEED, n_splits=5) -> np.ndarray:
    X, y = _stack(real, gen, style_real, styles)
    rng = np.random.default_rng(seed)
    return np.array([_cv_auc(X, rng.permutation(y), seed, n_splits)[0] for _ in range(n_perm)])


def pvalue(obs, null) -> float:
    return float((1 + np.sum(null >= obs)) / (len(null) + 1))


def real_vs_real(real, style_real, styles, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(real))
    h = len(real) // 2
    a, b = perm[:h], perm[h : 2 * h]
    out = {"idx_a": a, "idx_b": b}
    out["marg"] = c2st(real[a], real[b], seed=seed)
    sa = np.asarray(style_real)
    X = np.vstack(
        [np.hstack([real[a], onehot(sa[a], styles)]), np.hstack([real[b], onehot(sa[b], styles)])]
    )
    y = np.r_[np.zeros(h), np.ones(h)]
    auc, oof = _cv_auc(X, y, seed, 5)
    out["cond"] = {"auc": auc, "oof": oof, "y": y}
    return out


def dcr(U_query, U_train) -> np.ndarray:
    d2 = (U_query**2).sum(1)[:, None] + (U_train**2).sum(1)[None, :] - 2 * U_query @ U_train.T
    return np.sqrt(np.clip(d2, 0, None)).min(1)


def dcr_ratio(d_gen, d_real) -> float:
    return float(np.median(d_gen) / np.median(d_real))


def near_copy_rate(d_gen, d_real) -> float:
    return float(np.mean(d_gen < np.quantile(d_real, 0.05)))


def rf_style_accuracy(X_fit, s_fit, X_eval, s_eval) -> np.ndarray:
    return make_rf().fit(X_fit, s_fit).predict(X_eval) == np.asarray(s_eval)


def floor_metrics(real, gen, mins):
    fr = np.isclose(real, mins)
    fg = np.isclose(gen, mins)
    has = fr.mean(0) > 0.02
    err = float(np.abs(fg.mean(0) - fr.mean(0))[has].mean())
    return err, float(np.median(fg[:, has].sum(1))), float(np.median(fr[:, has].sum(1)))


def spearman(M) -> np.ndarray:
    R = rankdata(M, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(R, rowvar=False)
    return np.nan_to_num(C, nan=0.0)


def _upper(C):
    return C[np.triu_indices(len(C), 1)]


def corr_error(real, gen) -> float:
    return float(np.abs(_upper(spearman(real)) - _upper(spearman(gen))).mean())


def center_by_style(U, s):
    R = U.copy()
    for st in np.unique(s):
        m = s == st
        R[m] -= U[m].mean(0)
    return R


def within_style_corr_error(U_real, s_real, U_gen, s_gen) -> float:
    return corr_error(center_by_style(U_real, s_real), center_by_style(U_gen, s_gen))


def module_pairs(labels) -> np.ndarray:
    labels = np.asarray(labels)
    return _upper((labels[:, None] == labels[None, :]).astype(float)).astype(bool)


def module_rho(corr, inmod) -> float:
    return float(_upper(corr)[inmod].mean())


def dispersion_values(U_te, s_te, mu, C) -> np.ndarray:
    """C 为共享协方差矩阵，或 {风格: 协方差} 的 dict（风格专属对角缩放臂）。"""
    M = np.array([mu[st] for st in s_te])
    if isinstance(C, dict):
        V = np.array([np.diag(C[st]) for st in s_te])
    else:
        V = np.diag(C)
    return ((U_te - M) ** 2 / V).mean(1)


def dispersion_ratio(values, s, styles) -> dict:
    s = np.asarray(s)
    means = {st: float(values[s == st].mean()) for st in styles if (s == st).any()}
    med = np.median(list(means.values()))
    return {st: v / med for st, v in means.items()}


def auc_from_oof(y, p) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def paired_auc(oof, y, idx, n) -> float:
    """按啤酒行号重抽：真实第 i 行与成对的生成第 i 行同进同出。"""
    sel = np.r_[idx, idx + n]
    return auc_from_oof(y[sel], oof[sel])


def acceptance(sm: dict) -> dict:
    """sm 为 5 次重复均值：main、G0、G1 各指标与真实参照（方法论 §6.2）。"""
    m, g0, g1 = sm["main"], sm["G0"], sm["G1"]
    a = (
        abs(m["auc_marg"] - 0.5) <= 0.05
        and abs(m["auc_cond"] - 0.5) <= 0.05
        and m["auc_marg"] < g0["auc_marg"]
        and m["auc_cond"] < g0["auc_cond"]
    )
    b = m["dcr_ratio"] >= 0.9 and m["near_copy"] <= 0.10
    c = sm["trtr"] - m["tstr"] <= 0.10 and m["tstr"] > g0["tstr"]
    d = m["floor_err"] <= 2 * g1["floor_err"]
    e = sm["real_module_rho"] - m["module_rho"] <= 0.10 and m["wcorr_err"] <= 2 * g1["wcorr_err"]
    return {"a": bool(a), "b": bool(b), "c": bool(c), "d": bool(d), "e": bool(e)}
