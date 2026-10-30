<<<<<<< HEAD
"""图：合并与风格内关联散点（fig:q2-scatter）。

147 个点 = 主方法（图约束前向选择）Top-3 的 49 维 × 3 名次。
横轴：合并边际/偏相关系数（Top-1 为合并边际 Spearman，第 2、3 位为模块内偏 Spearman）；
纵轴：风格内 Spearman r_style。
点形状表示名次；颜色表示风格内 95% bootstrap 置信区间是否含 0。
参考线 y = x 与 y = 0；标注 |r_style| < 0.1（关联几乎完全来自风格之间）的
代表性配对：pH→酸感、乙醇→麦芽风味、乳酸→果香。

输入：code/data/processed/q2/final_top3_with_rationality.csv
输出：figures/q2/fig_q2_scatter.pdf
运行：python code/scripts/fig_q2_scatter.py
"""
=======
>>>>>>> 23399956bce0b4f25e3acd5228eb096d617187ea
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
<<<<<<< HEAD
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q2 import config as C  # noqa: E402

# 正文点名的代表性配对（compound → dimension, 展示名）
HIGHLIGHTS = (
    ("pH", "acidity", "pH→酸感"),
    ("ethanol..v.v.", "F_malt_all", "乙醇→麦芽风味"),
    ("lactic_acid.mg.L.", "A_esters_fruity", "乳酸→果香"),
)


def main() -> None:
    args = plotstyle.script_args(C.FIGURES)
    plotstyle.apply()
    p = plotstyle.PALETTE

    df = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    d = df[df["source_label"] == "主方法：图约束前向选择"].copy()
    assert len(d) == 147, f"主方法应为 147 行，实际 {len(d)}"
    assert d["score"].notna().all() and d["r_style"].notna().all()

    x = d["score"].to_numpy()
    y = d["r_style"].to_numpy()
    rank = d["rank"].to_numpy()
    lo = d["r_style_ci_lower"].to_numpy()
    hi = d["r_style_ci_upper"].to_numpy()
    ci_excl0 = (lo > 0) | (hi < 0)  # 风格内 CI 不含 0
    same_sign = np.sign(x) == np.sign(y)
    ratio = np.median(np.abs(y[np.abs(x) > 1e-12] / x[np.abs(x) > 1e-12]))
    print(f"风格内与合并同号 {same_sign.mean() * 100:.0f}%；|r_style/r| 中位 {ratio:.2f}")
    print(f"风格内 CI 不含 0：{int(ci_excl0.sum())}/147")

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    markers = {1: "o", 2: "s", 3: "^"}
    for r in (1, 2, 3):
        for excl, color, label_ci in ((True, p["main"], None), (False, "#BBBBBB", None)):
            m = (rank == r) & (ci_excl0 == excl)
            if not m.any():
                continue
            lab = f"第 {r} 位"
            if r == 1:
                lab += "（风格内 CI 不含 0）" if excl else "（风格内 CI 含 0）"
            ax.scatter(x[m], y[m], s=34, marker=markers[r], facecolor=color,
                       edgecolor="white", linewidths=0.5, zorder=3,
                       label=lab, alpha=0.95)

    lim = max(0.75, np.abs(np.concatenate([x, y])).max() + 0.05)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#444444", lw=0.9, label="y = x", zorder=1)
    ax.axhline(0, ls="--", color="#D55E00", lw=0.9, label="y = 0", zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("合并边际 Spearman $r_{jd}$（第 2、3 位为模块内偏 Spearman）")
    ax.set_ylabel("风格内 Spearman $r^{(s)}_{jd}$")
    ax.set_title("合并边际相关与风格内相关（147 项 Top-3）", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)

    # 标注代表性配对（|r_style|<0.1，关联几乎完全来自风格之间）
    for compound, dim, text in HIGHLIGHTS:
        row = d[(d["compound"] == compound) & (d["dimension"] == dim)]
        assert len(row) == 1, (compound, dim, len(row))
        xx, yy = float(row["score"].iloc[0]), float(row["r_style"].iloc[0])
        ax.annotate(
            text, xy=(xx, yy), xytext=(12, -14), textcoords="offset points",
            fontsize=8, color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.7},
        )

    fig.tight_layout()
    print(plotstyle.save(fig, args.out, "fig_q2_scatter.pdf"))
=======
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q2 import config as C  # noqa: E402

from src import paperfig as pf  # noqa: E402

MARKS = {1: "o", 2: "s", 3: "^"}
HIGHLIGHT = (("pH", "lactic"), ("pH", "acidity"), ("ethanol..v.v.", "F_malt_all"))


def main():
    pf.apply()
    d = pd.read_csv(C.Q2_DIR / "q2_marginal.csv")
    ci0 = d["flag_style_ci0"].astype(bool)
    fig, ax = plt.subplots(figsize=(pf.TEXTWIDTH * 0.72, 3.9))
    lim = 0.85
    ax.plot([-lim, lim], [-lim, lim], color=pf.MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.axhline(0, color=pf.MUTED, lw=0.7)
    ax.axvline(0, color=pf.GRID, lw=0.6)
    for k, mk in MARKS.items():
        for flag, col in ((False, pf.BLUE), (True, pf.LIGHT)):
            s = d[(d["rank"] == k) & (ci0 == flag)]
            ax.scatter(
                s["r_pooled"],
                s["r_style"],
                marker=mk,
                s=18,
                color=col,
                ec="white",
                lw=0.4,
                zorder=3,
            )
    for comp, dim in HIGHLIGHT:
        r = d[(d["compound"] == comp) & (d["dimension"] == dim)]
        if len(r):
            x, y = float(r["r_pooled"].iloc[0]), float(r["r_style"].iloc[0])
            ax.annotate(
                f"{comp} → {dim}",
                (x, y),
                xytext=(6, 8 if y >= 0 else -10),
                textcoords="offset points",
                fontsize=6.5,
                color=pf.ORANGE,
                arrowprops={"arrowstyle": "-", "color": pf.ORANGE, "lw": 0.6},
            )
    same = (d["r_pooled"] * d["r_style"] > 0).mean()
    ratio = (d["r_style"] / d["r_pooled"]).median()
    n_small = int((d["r_style"].abs() < 0.1).sum())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, 0.7)
    ax.set_xlabel("合并边际 Spearman $r_{jd}$")
    ax.set_ylabel("风格内 $r^{(s)}_{jd}$")
    ax.set_title(
        f"{len(d)} 项中同号 {same:.0%}，比值中位数 {ratio:.2f}，$|r^{{(s)}}|<0.1$ 共 {n_small} 项",
        fontsize=7.5,
        color=pf.MUTED,
        loc="left",
    )
    handles = [
        Line2D([], [], marker=MARKS[k], ls="", color=pf.MUTED, ms=5, label=f"第 {k} 名")
        for k in MARKS
    ]
    handles += [
        Line2D([], [], marker="o", ls="", color=pf.BLUE, ms=5, label="风格内区间不含 0"),
        Line2D([], [], marker="o", ls="", color=pf.LIGHT, ms=5, label="风格内区间含 0"),
        Line2D([], [], color=pf.MUTED, lw=0.7, ls=(0, (3, 2)), label="$y=x$"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, ncol=2, handletextpad=0.3)
    fig.tight_layout()
    print(pf.save(fig, "q2", "fig_q2_scatter.pdf"))
>>>>>>> 23399956bce0b4f25e3acd5228eb096d617187ea


if __name__ == "__main__":
    main()
