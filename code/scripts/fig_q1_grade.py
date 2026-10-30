<<<<<<< HEAD
"""图：误差集中的等级（fig:q1-grade）。

左栏：真值 G、主输出（预测分布中位数 G_hat）、第二输出（分位数切点 G_hat_Q）
     的 0--6 级频率（占全部 12 500 单元的百分比）。
右栏：主输出按真实等级分层的召回率——精确召回与 ±1 级内比例，
     柱下标注每个真实等级的分母（单元数）。

输入：code/data/processed/q1/q1_data.npz（G）
     code/data/processed/q1/q1_scores.npz（G_hat、G_hat_Q）
输出：figures/q1/fig_q1_grade.pdf（并复制一份到 paper/figures/ 供 LaTeX 引用）
运行：python code/scripts/fig_q1_grade.py
"""
import shutil
=======
>>>>>>> 23399956bce0b4f25e3acd5228eb096d617187ea
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
<<<<<<< HEAD
from src.q1.paths import FIG_DIR, PROCESSED_DIR, ROOT  # noqa: E402

PAPER_FIG_DIR = ROOT / "paper" / "figures"


def main() -> None:
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    p = plotstyle.PALETTE

    d = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    s = np.load(PROCESSED_DIR / "q1_scores.npz", allow_pickle=True)
    g_true = d["G"].astype(int).ravel()
    g_main = s["G_hat"].astype(int).ravel()   # 主输出：预测分布中位数
    g_sec = s["G_hat_Q"].astype(int).ravel()  # 第二输出：分位数切点
    grades = np.arange(7)

    freq = {
        "真值": np.bincount(g_true, minlength=7) / g_true.size * 100,
        "主输出": np.bincount(g_main, minlength=7) / g_main.size * 100,
        "第二输出": np.bincount(g_sec, minlength=7) / g_sec.size * 100,
    }
    print("0--6 级频率（%）：")
    for k, v in freq.items():
        print(f"  {k}: {np.round(v, 2)}")

    # 按真实等级分层的主输出召回率
    recall_exact, recall_pm1, n_true = [], [], []
    for k in grades:
        m = g_true == k
        n_true.append(int(m.sum()))
        recall_exact.append(float((g_main[m] == k).mean()) if m.any() else np.nan)
        recall_pm1.append(float((np.abs(g_main[m] - g_true[m]) <= 1).mean()) if m.any() else np.nan)
    recall_exact = np.array(recall_exact)
    recall_pm1 = np.array(recall_pm1)
    pm1_overall = float((np.abs(g_main - g_true) <= 1).mean())
    print(f"主输出整体 ±1 内比例 {pm1_overall * 100:.1f}%")
    print("分母 n：", n_true)

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # ---- 左栏：三种等级频率 ----
    w = 0.26
    xs = np.arange(7)
    series = [
        ("真值", freq["真值"], "#444444"),
        ("主输出（预测分布中位数）", freq["主输出"], p["main"]),
        ("第二输出（分位数切点）", freq["第二输出"], p["warn"]),
    ]
    for i, (name, v, c) in enumerate(series):
        axl.bar(xs + (i - 1) * w, v, width=w * 0.92, color=c, label=name)
    axl.set_xticks(xs)
    axl.set_xlabel("等级")
    axl.set_ylabel("占全部单元的比例（%）")
    axl.set_title("真值、主输出与第二输出的 0--6 级频率", fontsize=10)
    axl.legend(fontsize=7.5, frameon=False)

    # ---- 右栏：按真实等级分层的主输出召回率 ----
    axr.bar(xs - 0.19, recall_exact, width=0.36, color=p["main"], label="精确召回")
    axr.bar(xs + 0.19, recall_pm1, width=0.36, color=p["accent"], label="±1 级内")
    for x, r1, r2, n in zip(xs, recall_exact, recall_pm1, n_true):
        if not np.isnan(r1):
            axr.text(x - 0.19, r1 + 0.015, f"{r1:.2f}", ha="center", va="bottom", fontsize=6.5)
        axr.text(x, -0.155, f"n={n}", ha="center", va="top", fontsize=6.5,
                 transform=axr.transData, clip_on=False)
    axr.set_xticks(xs)
    axr.set_xlabel("真实等级（柱下为该等级单元数，即分母）")
    axr.set_ylabel("比例")
    axr.set_ylim(0, 1.08)
    axr.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axr.set_title("主输出按真实等级分层的召回率", fontsize=10)
    axr.legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.tight_layout()
    out = plotstyle.save(fig, args.out, "fig_q1_grade.pdf")
    print("已保存", out)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, PAPER_FIG_DIR / "fig_q1_grade.pdf")
    print("已复制到", PAPER_FIG_DIR / "fig_q1_grade.pdf")
=======
from src.q1.discretize import N_GRADES  # noqa: E402
from src.q1.paths import PROCESSED_DIR  # noqa: E402

from src import paperfig as pf  # noqa: E402


def freq(g):
    g = np.asarray(g)
    reps = g.reshape(-1, *g.shape[-2:]) if g.ndim == 3 else g[None]
    return np.mean([np.bincount(r.ravel(), minlength=N_GRADES) / r.size for r in reps], axis=0)


def main():
    pf.apply()
    G = plotstyle.load_npz(PROCESSED_DIR / "q1_data.npz")["G"]
    s = plotstyle.load_npz(PROCESSED_DIR / "q1_scores.npz")
    gm, gq = s["grade_median"], s["grade_quant"]
    R = gm.shape[0]
    k = np.arange(N_GRADES)
    n_true = np.bincount(G.ravel(), minlength=N_GRADES)

    fig, axes = plt.subplots(1, 2, figsize=(pf.TEXTWIDTH, 3.2))
    ax = axes[0]
    series = (
        ("真值等级", freq(G), pf.INK),
        ("主输出：分布中位数", freq(gm), pf.BLUE),
        ("第二输出：分位数切点", freq(gq), pf.ORANGE),
    )
    w = 0.27
    for i, (lab, f, c) in enumerate(series):
        ax.bar(k + (i - 1) * w, f, width=w - 0.03, color=c, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xticks(k)
    ax.set_xlabel("等级")
    ax.set_ylabel("等级频率，对数刻度")
    ax.grid(axis="y", visible=True, which="major")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=1, handlelength=1.0)
    pf.panel_label(ax, "a  三种等级频率")

    ax = axes[1]
    hit = np.broadcast_to(G, gm.shape) == gm
    recall = np.array(
        [hit[np.broadcast_to(G, gm.shape) == j].mean() if n_true[j] else np.nan for j in k]
    )
    near = np.abs(np.broadcast_to(G, gm.shape) - gm) <= 1
    recall1 = np.array(
        [near[np.broadcast_to(G, gm.shape) == j].mean() if n_true[j] else np.nan for j in k]
    )
    ax.bar(k, recall1, width=0.62, color=pf.SKY, label="误差不超过一级")
    ax.bar(k, recall, width=0.62, color=pf.BLUE, label="完全一致，即召回率")
    for j in k:
        ax.text(j, recall1[j] + 0.03, f"{recall[j]:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(k, [f"{j}\n{n_true[j]}" for j in k])
    ax.set_xlabel("真实等级与单元数")
    ax.set_ylabel("主输出的比例")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", visible=True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=1, handlelength=1.0)
    pf.panel_label(ax, "b  主输出按真实等级分层的召回率")
    fig.text(
        0.5,
        -0.05,
        f"分母：a 栏为全部 {G.size:,} 个啤酒乘维度单元，预测等级取 {R} 次重复的平均；"
        "b 栏为各真实等级的单元数，柱顶数字为召回率",
        ha="center",
        fontsize=7,
        color=pf.MUTED,
    )
    fig.tight_layout(w_pad=2.0)
    print(pf.save(fig, "q1", "fig_q1_grade.pdf"))
>>>>>>> 23399956bce0b4f25e3acd5228eb096d617187ea


if __name__ == "__main__":
    main()
