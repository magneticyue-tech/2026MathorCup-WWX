import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402

ARM_LABELS = {
    "G-S1": "纯因子协方差",
    "G-S2": "Ledoit–Wolf 协方差",
    "G-S3_k-": "因子数 k 下移一格",
    "G-S3_k+": "因子数 k 上移一格",
    "G-S3_kappa-": "收缩强度 κ 下移一格",
    "G-S3_kappa+": "收缩强度 κ 上移一格",
    "G-S4": "风格专属对角方差缩放",
    "G-S5": "并列块潜变量改用平均秩正态得分",
    "G-S6": "化学缺失值编码到最低位",
    "G-S7": "附件 1 替换为附件 8 的 80 个单元格",
    "G-S8": "删除 4 个综合求和列的对照模型",
    "G-S9": "内标从经验分布独立抽取",
    "T-S1": "综合指标只用岭回归预测",
    "T-S2": "候选选择改用生成器后验最大 α",
    "T-S3": "阈值口径改为中位",
    "T-S4": "化学层规则改为严格不增加",
    "T-S5a": "感官层规则：连续预测不升高",
    "T-S5b": "不设感官层约束",
    "T-S6": "问题 1 映射改为分位数切点",
    "T-S7a": "缺失值最低位编码的生成器重跑迁移（11 款啤酒）",
    "T-S7b": "附件 8 取值的生成器重跑迁移（11 款啤酒）",
    "T-S7c": "平均秩正态得分的生成器重跑迁移（11 款啤酒）",
}

GEN_PANELS = [
    ("main.auc_marg", "合并两样本分类检验（C2ST）\n曲线下面积（AUC）"),
    ("main.tstr", "生成样本训练、真实样本测试（TSTR）\n风格判别准确率"),
    ("main.module_rho", "问题二模块内化合物对\n平均 Spearman"),
    ("main.wcorr_err", "风格内相关误差\n（Spearman 矩阵\n逐元素绝对差均值）"),
]
TR_PANELS = [
    ("pils_tripel.alpha_median", "11 款啤酒的最大可行迁移强度 α*\n中位数"),
    ("pils_tripel.p_median", "11 款啤酒的目标风格为 Tripel\n的概率中位数"),
    ("all_pairs.alpha_median", "462 个风格对的最大可行迁移强度 α*\n中位数"),
    ("all_pairs.active_frac", "462 个风格对中\n约束起作用比例"),
]


def main():
    apply()
    df = pd.read_csv(PROCESSED_DIR / "q3_sensitivity.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 4, figsize=(18, 11))
    panels = iter("abcdefgh")
    g = df[df["arm"].str.startswith("G-S")]
    for ax, (k, metric) in zip(axes[0], GEN_PANELS):
        sub = g[g[f"{k}.diff"].notna()] if f"{k}.diff" in g else g.iloc[:0]
        y = np.arange(len(sub))[::-1]
        if len(sub):
            for yi, lo, hi in zip(y, sub[f"{k}.diff_lo"], sub[f"{k}.diff_hi"]):
                ax.plot([lo, hi], [yi, yi], color=PALETTE["main"], lw=1.4)
            ax.plot(sub[f"{k}.diff"], y, "o", color=PALETTE["main"], ms=4)
        ax.axvline(0, ls="--", color="#444444", lw=0.8)
        ax.set_yticks(y, [ARM_LABELS.get(a, a) for a in sub["arm"]], fontsize=9)
        ax.set_xlabel(f"{metric}\n配对差（敏感性设定 − 基线）\n95% 配对区间", fontsize=9)
        ax.set_title(f"({next(panels)})", loc="left", fontsize=11)
    t = df[df["arm"].str.startswith("T-S") | (df["arm"] == "G-S8")]
    for ax, (k, metric) in zip(axes[1], TR_PANELS):
        col = f"{k}.diff"
        sub = t[t[col].notna()] if col in t else t.iloc[:0]
        y = np.arange(len(sub))[::-1]
        ax.barh(y, sub[col] if len(sub) else [], color=PALETTE["main"])
        ax.axvline(0, ls="--", color="#444444", lw=0.8)
        ax.set_yticks(y, [ARM_LABELS.get(a, a) for a in sub["arm"]], fontsize=9)
        ax.set_xlabel(f"{metric}\n配对差（敏感性设定 − 基线）", fontsize=9)
        ax.set_title(f"({next(panels)})", loc="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.98, 1))
    save(fig, FIG_DIR, "fig_q3_sensitivity.pdf")


if __name__ == "__main__":
    main()
