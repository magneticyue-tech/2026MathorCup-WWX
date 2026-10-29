# code/src/q2/core.py
"""Q2 统计核心：秩、逐对 Spearman、模块、图约束前向选择、置换门槛与 bootstrap。

所有选择都在相关矩阵上完成。两列标准化秩的均值内积就是 Spearman 系数；
给定已选集合的偏相关用递推公式计算，与对 [1, x_S] 回归取残差的定义等价。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

import config as C

NEG = -np.inf


# ---------------------------------------------------------------- 秩与相关
def zrank(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐列平均秩并标准化；返回 (Z, ok)，常数列置 0 且 ok=False。"""
    R = rankdata(M, axis=0, method="average")
    R = R - R.mean(axis=0)
    s = R.std(axis=0)
    ok = s > 1e-12
    Z = np.zeros_like(R)
    Z[:, ok] = R[:, ok] / s[ok]
    return Z, ok


def fill_array(M: np.ndarray, scheme: str) -> np.ndarray:
    out = np.array(M, dtype=float, copy=True)
    na = np.isnan(out)
    if not na.any():
        return out
    if scheme == "mean":
        mu = np.nanmean(out, axis=0)
        out[na] = np.take(mu, np.where(na)[1])
    elif scheme == "lowest":
        out[na] = -np.inf
    else:
        raise ValueError(scheme)
    return out


def _corr_z(Za, oka, Zb, okb):
    Cm = Za.T @ Zb / Za.shape[0]
    Cm[~oka, :] = np.nan
    Cm[:, ~okb] = np.nan
    return Cm


def pairwise_spearman(M: np.ndarray) -> np.ndarray:
    """逐对有效样本的 Spearman（与 pandas corr(method='spearman') 同口径）。"""
    n, p = M.shape
    na = np.isnan(M)
    na_cols = np.where(na.any(axis=0))[0]
    full = np.where(~na.any(axis=0))[0]
    out = np.full((p, p), np.nan)
    Zf, okf = zrank(M[:, full])
    out[np.ix_(full, full)] = _corr_z(Zf, okf, Zf, okf)
    for a in na_cols:
        rows = ~na[:, a]
        sub = M[rows]
        clean = np.where(~np.isnan(sub).any(axis=0))[0]
        Zs, oks = zrank(sub[:, clean])
        ia = int(np.where(clean == a)[0][0])
        r = _corr_z(Zs[:, [ia]], oks[[ia]], Zs, oks)[0]
        out[a, clean] = r
        out[clean, a] = r
        for b in np.setdiff1d(np.arange(p), clean):
            rows2 = rows & ~na[:, b]
            if rows2.sum() < 3:
                continue
            Zp, okp = zrank(M[rows2][:, [a, b]])
            r2 = _corr_z(Zp, okp, Zp, okp)[0, 1]
            out[a, b] = out[b, a] = r2
    diag_ok = ~np.isnan(np.diag(out))
    out[np.arange(p)[diag_ok], np.arange(p)[diag_ok]] = 1.0
    return out


def modules(abs_corr: np.ndarray, tau: float) -> np.ndarray:
    """1-|ρ| 上平均连接层次聚类，高度 1-τ 处切开；缺失相关按 0 处理。"""
    A = np.nan_to_num(np.asarray(abs_corr, dtype=float), nan=0.0)
    D = np.clip(1.0 - A, 0.0, None)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), "average"), 1.0 - tau, "distance")


# ---------------------------------------------------------------- 选择
def _partial(r_jy, r_js, r_sy):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (r_jy - r_js * r_sy) / np.sqrt((1.0 - r_js**2) * (1.0 - r_sy**2))


def partial_given(Cxx, Cxy, s1, s2=None):
    """全部候选 j 对每个维度 d 的偏相关 r_{jd|S}；s1、s2 为每维的已选下标（长度 D）。"""
    d = np.arange(Cxy.shape[1])
    r1 = _partial(Cxy, Cxx[:, s1], Cxy[s1, d][None, :])
    if s2 is None:
        return r1
    r_j2_1 = _partial(Cxx[:, s2], Cxx[:, s1], Cxx[s2, s1][None, :])
    r_2y_1 = r1[s2, d]
    return _partial(r1, r_j2_1, r_2y_1[None, :])


def _pick(score, mask):
    a = np.abs(score)
    a = np.where(np.isfinite(a) & mask, a, NEG)
    idx = np.argmax(a, axis=0)
    d = np.arange(score.shape[1])
    if np.any(~np.isfinite(a[idx, d])):
        raise RuntimeError("某维度没有可选候选")
    return idx, score[idx, d]


def forward_select(Cxx, Cxy, mod, valid):
    """图约束前向选择。返回 (sel D×3, score D×3)。"""
    p, D = Cxy.shape
    base = valid[:, None] & np.ones((1, D), bool)
    s1, v1 = _pick(Cxy, base)
    r2 = partial_given(Cxx, Cxy, s1)
    m2 = base & (mod[:, None] != mod[s1][None, :])
    s2, v2 = _pick(r2, m2)
    r3 = partial_given(Cxx, Cxy, s1, s2)
    m3 = m2 & (mod[:, None] != mod[s2][None, :])
    s3, v3 = _pick(r3, m3)
    return np.column_stack([s1, s2, s3]), np.column_stack([v1, v2, v3])


def ranked_select(score, mod, valid, tie=None):
    """按 |score| 降序（并列时按 |tie|）取前 3，每模块至多一个；mod=None 时不加约束。"""
    p, D = score.shape
    sel = np.zeros((D, 3), int)
    for d in range(D):
        keys = (-np.abs(tie[:, d]),) if tie is not None else ()
        order = np.lexsort(keys + (-np.abs(score[:, d]),))
        used, out = set(), []
        for j in order:
            if not valid[j]:
                continue
            if mod is not None:
                if mod[j] in used:
                    continue
                used.add(mod[j])
            out.append(j)
            if len(out) == 3:
                break
        sel[d] = out
    return sel


def fixed_stats(Cxx, Cxy, sel):
    """固定全样本 Top-3 的得分：第 1 位 r，第 2、3 位给定前序的偏 r。"""
    d = np.arange(Cxy.shape[1])
    t1, t2, t3 = sel[:, 0], sel[:, 1], sel[:, 2]
    s1 = Cxy[t1, d]
    s2 = partial_given(Cxx, Cxy, t1)[t2, d]
    s3 = partial_given(Cxx, Cxy, t1, t2)[t3, d]
    return np.column_stack([s1, s2, s3])


def corr_mats(Zx, okx, Zy, oky):
    n = Zx.shape[0]
    Cxx = Zx.T @ Zx / n
    Cxy = Zx.T @ Zy / n
    Cxx[~okx, :] = np.nan
    Cxx[:, ~okx] = np.nan
    Cxy[~okx, :] = np.nan
    Cxy[:, ~oky] = np.nan
    return Cxx, Cxy


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


# ---------------------------------------------------------------- 置换门槛
def permutation_thresholds(Zx, Zy, sel, mod, valid, n_perm, seed):
    """第 1 位：打乱 y 取 max|r| 的 95% 分位；第 2、3 位：Freedman–Lane（打乱 e_y）。"""
    rng = np.random.default_rng(seed)
    n, D = Zy.shape
    q = np.zeros((D, 3))
    Xv = Zx[:, valid]
    vidx = np.where(valid)[0]
    for d in range(D):
        y = Zy[:, d]
        P = np.argsort(rng.random((n, n_perm)), axis=0)
        Yp = y[P]
        q[d, 0] = np.quantile(np.abs(Xv.T @ Yp / n).max(axis=0), 0.95)
        for k in (1, 2):
            S = sel[d, :k]
            Q = np.column_stack([np.ones(n), Zx[:, S]])
            H = Q @ np.linalg.pinv(Q)
            e_y = y - H @ y
            keep = ~np.isin(mod[vidx], mod[S])
            E = Xv[:, keep] - H @ Xv[:, keep]
            nE = np.linalg.norm(E, axis=0)
            E = E[:, nE > 1e-10]
            nE = nE[nE > 1e-10]
            P = np.argsort(rng.random((n, n_perm)), axis=0)
            Ep = e_y[P]
            r = (E.T @ Ep) / (nE[:, None] * np.linalg.norm(e_y))
            q[d, k] = np.quantile(np.abs(r).max(axis=0), 0.95)
    return q


# ---------------------------------------------------------------- bootstrap
def bootstrap_indices(groups, n_boot, seed, unit="date_group"):
    """簇 bootstrap：每次有放回抽取与簇数相同个簇并拼接其啤酒；unit="beer" 时按单款啤酒重抽。"""
    rng = np.random.default_rng(seed)
    n = len(groups)
    if unit == "beer":
        return [rng.integers(0, n, n) for _ in range(n_boot)]
    if unit != "date_group":
        raise ValueError(unit)
    labels = np.unique(groups)
    groups = [np.where(groups == g)[0] for g in labels]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[g] for g in pick]))
    return out


# ---------------------------------------------------------------- 方案
@dataclass
class ArmSpec:
    name: str
    tau: float = C.TAU_MAIN
    candidates: list[str] | None = None
    na_scheme: str = "mean"
    chem: pd.DataFrame | None = None
    target: pd.DataFrame | None = None
    graph_source: str = "attachment2"
    boot_unit: str = "date_group"
    rebuild_graph: bool = True
    n_perm: int = C.N_PERMUTATION
    n_boot: int = C.N_BOOTSTRAP
    seed: int = C.SEED
    notes: dict = field(default_factory=dict)


@dataclass
class ArmData:
    cand: list[str]
    dims: list[str]
    Xraw: np.ndarray
    Yraw: np.ndarray
    corr: np.ndarray
    mod: np.ndarray
    Zx: np.ndarray
    okx: np.ndarray
    Zy: np.ndarray
    oky: np.ndarray
    groups: np.ndarray


def prepare_arm(ds, spec: ArmSpec) -> ArmData:
    cand = list(spec.candidates or ds.candidate_cols)
    chem = spec.chem if spec.chem is not None else ds.chem_raw
    Ydf = spec.target if spec.target is not None else ds.xbar[ds.delivery_sensory_cols]
    Xraw = chem[cand].to_numpy(float)
    Yraw = Ydf[ds.delivery_sensory_cols].to_numpy(float)
    if spec.graph_source == "attachment2":
        corr = ds.corr.loc[cand, cand].to_numpy(float)
    elif spec.graph_source == "recompute":
        corr = pairwise_spearman(Xraw)
    else:
        raise ValueError(spec.graph_source)
    mod = modules(np.abs(corr), spec.tau)
    Zx, okx = zrank(fill_array(Xraw, spec.na_scheme))
    Zy, oky = zrank(Yraw)
    if not okx.all() or not oky.all():
        raise ValueError("全样本存在常数列")
    return ArmData(cand, list(ds.delivery_sensory_cols), Xraw, Yraw, corr, mod, Zx, okx, Zy, oky, ds.date_group)


def select_full(a: ArmData):
    Cxx, Cxy = corr_mats(a.Zx, a.okx, a.Zy, a.oky)
    sel, score = forward_select(Cxx, Cxy, a.mod, a.okx)
    return sel, score, Cxx, Cxy


def run_bootstrap(a: ArmData, spec: ArmSpec, sel_full, progress=False):
    """返回每次重复的 Top-3（B×D×3）、固定 Top-3 的得分（B×D×3）与每次抽到的不同日期组数（B）。"""
    idx_list = bootstrap_indices(a.groups, spec.n_boot, spec.seed, spec.boot_unit)
    n_distinct = np.array([len(np.unique(a.groups[idx])) for idx in idx_list])
    B, D = len(idx_list), len(a.dims)
    boot_sel = np.zeros((B, D, 3), int)
    boot_fix = np.full((B, D, 3), np.nan)
    for b, idx in enumerate(idx_list):
        Zx, okx = zrank(fill_array(a.Xraw[idx], spec.na_scheme))
        Zy, oky = zrank(a.Yraw[idx])
        Cxx, Cxy = corr_mats(Zx, okx, Zy, oky)
        if spec.rebuild_graph:
            mod_b = modules(np.abs(pairwise_spearman(a.Xraw[idx])), spec.tau)
        else:
            mod_b = a.mod
        if not oky.all():
            raise RuntimeError(f"bootstrap {b}：感官列为常数")
        boot_sel[b], _ = forward_select(Cxx, Cxy, mod_b, okx)
        boot_fix[b] = fixed_stats(Cxx, Cxy, sel_full)
        if progress and (b + 1) % max(1, B // 10) == 0:
            print(f"    bootstrap {b + 1}/{B}", flush=True)
    return boot_sel, boot_fix, n_distinct


def summarize_bootstrap(a: ArmData, sel_full, boot_sel, boot_fix):
    B, D, _ = boot_sel.shape
    p = len(a.cand)
    pi_c = np.zeros((D, p))
    pi_pos = np.zeros((D, p, 3))
    mod_freq = {}
    jac_c = np.zeros((B, D))
    jac_m = np.zeros((B, D))
    for d in range(D):
        S = boot_sel[:, d, :]
        for k in range(3):
            pi_pos[d, :, k] = np.bincount(S[:, k], minlength=p) / B
        pi_c[d] = pi_pos[d].sum(axis=1)
        M = a.mod[S]
        mf = {}
        for b in range(B):
            for m in set(M[b]):
                mf[m] = mf.get(m, 0) + 1
            jac_c[b, d] = jaccard(S[b], sel_full[d])
            jac_m[b, d] = jaccard(M[b], a.mod[sel_full[d]])
        mod_freq[d] = {m: v / B for m, v in mf.items()}
    with np.errstate(invalid="ignore"):
        ci_lo = np.nanpercentile(boot_fix, 2.5, axis=0)
        ci_hi = np.nanpercentile(boot_fix, 97.5, axis=0)
    n_valid = np.isfinite(boot_fix).sum(axis=0)
    return dict(pi_c=pi_c, pi_pos=pi_pos, mod_freq=mod_freq, jac_c=jac_c.mean(0), jac_m=jac_m.mean(0),
                ci_lo=ci_lo, ci_hi=ci_hi, n_valid=n_valid)


def stability_label(pi):
    if pi >= C.STABLE_THRESHOLD:
        return "稳定"
    if pi >= C.MEDIUM_THRESHOLD:
        return "中等"
    return "不稳定"


def grade_from(sig1: bool, pi1: float) -> str:
    if not sig1:
        return "C"
    return "A" if pi1 >= C.GRADE_A_PI else "B"


def build_table(a: ArmData, sel, score, q, summ) -> pd.DataFrame:
    rows = []
    for d, dim in enumerate(a.dims):
        sel_mods = set(a.mod[sel[d]])
        for k in range(3):
            j = sel[d, k]
            lo, hi = summ["ci_lo"][d, k], summ["ci_hi"][d, k]
            ci_ok = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
            passed = bool(abs(score[d, k]) > q[d, k])
            mf = summ["mod_freq"][d]
            own = mf.get(a.mod[j], 0.0)
            comp = sorted(
                [(m, f) for m, f in mf.items() if m not in sel_mods and f > own], key=lambda t: -t[1]
            )
            rows.append({
                "dimension": dim,
                "rank": k + 1,
                "compound": a.cand[j],
                "module_id": int(a.mod[j]),
                "score_type": "spearman" if k == 0 else "partial_spearman",
                "score": float(score[d, k]),
                "q_d": float(q[d, k]),
                "pass_threshold": passed,
                "r_ci_lower": float(lo),
                "r_ci_upper": float(hi),
                "n_boot_valid": int(summ["n_valid"][d, k]),
                "ci_excludes_0": ci_ok,
                "is_significant": passed and ci_ok,
                "pi_c": float(summ["pi_c"][d, j]),
                "pi_c_pos": float(summ["pi_pos"][d, j, k]),
                "pi_m": float(own),
                "jaccard_boot": float(summ["jac_c"][d]),
                "jaccard_boot_module": float(summ["jac_m"][d]),
                "competitor_modules": ";".join(f"{m}:{f:.3f}" for m, f in comp),
            })
    df = pd.DataFrame(rows)
    df["stability"] = df["pi_c"].map(stability_label)
    r1 = df[df["rank"] == 1].set_index("dimension")
    gmap = {dim: grade_from(bool(r1.at[dim, "is_significant"]), float(r1.at[dim, "pi_c"])) for dim in r1.index}
    df["grade"] = df["dimension"].map(gmap)
    return df


def run_arm(ds, spec: ArmSpec, q_override=None, progress=False):
    """完整流程：选择 → 置换门槛 → bootstrap → 等级。返回 (表, 附加数组)。"""
    a = prepare_arm(ds, spec)
    sel, score, _, _ = select_full(a)
    if q_override is None:
        q = permutation_thresholds(a.Zx, a.Zy, sel, a.mod, a.okx, spec.n_perm, spec.seed)
    else:
        q = q_override
    boot_sel, boot_fix, n_distinct = run_bootstrap(a, spec, sel, progress=progress)
    summ = summarize_bootstrap(a, sel, boot_sel, boot_fix)
    table = build_table(a, sel, score, q, summ)
    return table, dict(arm=a, sel=sel, score=score, q=q, boot_sel=boot_sel, boot_fix=boot_fix, summ=summ,
                       n_distinct=n_distinct)


def module_label_map(cand, mod):
    return {c: int(m) for c, m in zip(cand, mod)}


def compare_tables(base: pd.DataFrame, other: pd.DataFrame, base_mod: dict) -> pd.DataFrame:
    """逐维比较：化合物层与模块层 Jaccard、Top-1 是否改变、等级是否改变。"""
    def mods(cs):
        return [base_mod.get(c, f"x:{c}") for c in cs]

    rows = []
    for dim in base["dimension"].unique():
        b = base[base["dimension"] == dim].sort_values("rank")
        o = other[other["dimension"] == dim].sort_values("rank")
        bc, oc = b["compound"].tolist(), o["compound"].tolist()
        row = {
            "dimension": dim,
            "base_top3": "|".join(bc),
            "arm_top3": "|".join(oc),
            "jaccard_compound": jaccard(bc, oc),
            "jaccard_module": jaccard(mods(bc), mods(oc)),
            "top1_changed": bc[0] != oc[0],
        }
        if "grade" in o.columns:
            row.update({
                "base_grade": b["grade"].iloc[0],
                "arm_grade": o["grade"].iloc[0],
                "grade_changed": b["grade"].iloc[0] != o["grade"].iloc[0],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def sel_from_table(df: pd.DataFrame, cand: list[str], dims: list[str]) -> np.ndarray:
    pos = {c: i for i, c in enumerate(cand)}
    sel = np.zeros((len(dims), 3), int)
    for d, dim in enumerate(dims):
        g = df[df["dimension"] == dim].sort_values("rank")
        if g["rank"].tolist() != [1, 2, 3]:
            raise ValueError(f"{dim} 不是恰好 3 行")
        sel[d] = [pos[c] for c in g["compound"]]
    return sel


def check_147(df: pd.DataFrame, label: str) -> None:
    if len(df) != 147 or df["dimension"].nunique() != 49 or (df.groupby("dimension").size() != 3).any():
        raise AssertionError(f"{label}：应为 49 维 × 3 行")
    if "barnyard" in set(df["dimension"]):
        raise AssertionError(f"{label}：含 barnyard")
