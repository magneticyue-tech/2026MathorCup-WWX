import numpy as np

ARMS = {
    "G-S1": {"kind": "generator", "change": "协方差：纯因子", "compare": "§6.2 指标"},
    "G-S2": {"kind": "generator", "change": "协方差：Ledoit–Wolf", "compare": "§6.2 指标"},
    "G-S3": {"kind": "generator", "change": "k、κ 取相邻网格点", "compare": "§6.2 指标"},
    "G-S4": {"kind": "generator", "change": "风格专属对角方差缩放（ν=10）", "compare": "§6.2 指标"},
    "G-S5": {"kind": "generator", "change": "并列块潜变量：平均秩正态得分", "compare": "§6.2 指标"},
    "G-S6": {"kind": "generator", "change": "化学 NA：单化合物最低位", "compare": "§6.2 指标"},
    "G-S7": {
        "kind": "generator",
        "change": "附件 1 与附件 8：替换 80 个单元格",
        "compare": "§6.2 指标",
    },
    "G-S8": {
        "kind": "portrait",
        "change": "Q1 删除四个 _sum 特征的对照模型",
        "compare": "画像与 α*",
    },
    "G-S9": {"kind": "portrait", "change": "内标：从经验分布独立抽取", "compare": "画像"},
    "T-S1": {"kind": "transfer", "change": "综合指标只用岭回归预测", "compare": "迁移指标"},
    "T-S2": {"kind": "transfer", "change": "候选选择：α_fit", "compare": "迁移指标"},
    "T-S3": {"kind": "transfer", "change": "阈值口径：中位", "compare": "迁移指标"},
    "T-S4": {"kind": "transfer", "change": "化学层规则：严格不增加", "compare": "迁移指标"},
    "T-S5a": {"kind": "transfer", "change": "感官层：连续预测不升高", "compare": "迁移指标"},
    "T-S5b": {"kind": "transfer", "change": "不设感官层", "compare": "迁移指标"},
    "T-S6": {"kind": "transfer", "change": "Q1 映射：分位数切点", "compare": "迁移指标"},
    "T-S7a": {
        "kind": "transfer",
        "change": "G-S6 生成器（NA 最低位）重跑迁移，11 款",
        "compare": "迁移指标",
    },
    "T-S7b": {
        "kind": "transfer",
        "change": "G-S7 生成器（附件 8）重跑迁移，11 款",
        "compare": "迁移指标",
    },
    "T-S7c": {
        "kind": "transfer",
        "change": "G-S5 生成器（平均秩）重跑迁移，11 款",
        "compare": "迁移指标",
    },
}


def _ci(d):
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def compare_generator(base, arm) -> dict:
    """base、arm 为 dict：keys、estimate、boot（键 × n_boot），以及逐风格 boot。"""
    out = {}
    bk = list(base["keys"])
    for i, k in enumerate(arm["keys"]):
        if k not in bk:
            continue
        j = bk.index(k)
        d = arm["boot"][i] - base["boot"][j]
        out[k] = {
            "base": float(base["estimate"][j]),
            "arm": float(arm["estimate"][i]),
            "diff": float(arm["estimate"][i] - base["estimate"][j]),
            "diff_ci95": _ci(d),
        }
    for name in ("main", "dispersion"):
        if name not in base.get("per_style", {}) or name not in arm.get("per_style", {}):
            continue
        b, a, w = base["per_style"][name], arm["per_style"][name], base["weights"]
        bm, am = np.nanmean(b["boot"], 1), np.nanmean(a["boot"], 1)
        bw, aw = np.nansum(b["boot"] * w, 1), np.nansum(a["boot"] * w, 1)
        out[f"per_style.{name}"] = {
            "macro_base": float(np.nanmean(b["value"])),
            "macro_arm": float(np.nanmean(a["value"])),
            "macro_diff_ci95": _ci(am - bm),
            "weighted_base": float(np.nansum(b["value"] * w)),
            "weighted_arm": float(np.nansum(a["value"] * w)),
            "weighted_diff_ci95": _ci(aw - bw),
            "per_style_arm": a["value"].tolist(),
            "per_style_base": b["value"].tolist(),
        }
    return out


def compare_transfer(base, arm, rows=None, ti=None) -> dict:
    """base、arm 为 dict：alpha（n×T）、p（n×T）、active（n×T 0/1），NaN 表示未计算。"""

    def pick(a):
        if rows is None:
            return a
        return a[rows, ti]

    ab, aa = pick(base["alpha"]), pick(arm["alpha"])
    m = ~np.isnan(ab) & ~np.isnan(aa)
    pb, pa = pick(base["p"]), pick(arm["p"])
    cb, ca = pick(base["active"]), pick(arm["active"])
    return {
        "n": int(m.sum()),
        "alpha_median_base": float(np.median(ab[m])),
        "alpha_median_arm": float(np.median(aa[m])),
        "alpha_median_diff": float(np.median(aa[m]) - np.median(ab[m])),
        "active_frac_base": float(np.mean(cb[m])),
        "active_frac_arm": float(np.mean(ca[m])),
        "active_frac_diff": float(np.mean(ca[m]) - np.mean(cb[m])),
        "p_median_base": float(np.median(pb[m])),
        "p_median_arm": float(np.median(pa[m])),
        "p_median_diff": float(np.median(pa[m]) - np.median(pb[m])),
        "n_alpha_changed": int((np.abs(aa[m] - ab[m]) > 1e-12).sum()),
    }


def compare_portrait(base_q50, arm_q50) -> dict:
    from scipy.stats import spearmanr

    rho = [spearmanr(base_q50[s], arm_q50[s])[0] for s in range(len(base_q50))]
    mad = np.abs(arm_q50 - base_q50).mean(1)
    return {
        "median_spearman_q50": float(np.median(rho)),
        "min_spearman_q50": float(np.min(rho)),
        "mean_abs_diff_q50": float(mad.mean()),
        "max_abs_diff_q50": float(np.abs(arm_q50 - base_q50).max()),
    }
