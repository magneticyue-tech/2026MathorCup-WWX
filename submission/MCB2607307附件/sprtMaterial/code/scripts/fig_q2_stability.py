import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q2 import config as C  # noqa: E402

from src import paperfig as pf  # noqa: E402

TIER_COLORS = ("#e3e3e3", "#9ecae1", "#2171b5")


def main():
    pf.apply()
    final = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    m = final[final["method"] == "main"]
    present = set(m["dimension"])
    order, bounds = [], []
    for g, members in DIM_GROUPS.items():
        start = len(order)
        order += [d for d in members if d in present]
        bounds.append((g, start, len(order)))
    pi = m.pivot(index="dimension", columns="rank", values="pi_c").loc[order].to_numpy()
    sig = m.pivot(index="dimension", columns="rank", values="is_significant").loc[order]
    sig = sig.astype(bool).to_numpy()
    grade = m[m["rank"] == 1].set_index("dimension")["grade"].loc[order]
    lo, hi = C.MEDIUM_THRESHOLD, C.STABLE_THRESHOLD

    fig, axes = plt.subplots(
        1, 2, figsize=(pf.TEXTWIDTH, 7.6), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )
    ax = axes[0]
    cmap = ListedColormap(TIER_COLORS)
    norm = BoundaryNorm([0, lo, hi, 1.0001], 3)
    ax.imshow(pi, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    for i in range(len(order)):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{pi[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if pi[i, j] >= hi else pf.INK,
            )
            if not sig[i, j]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="#7f7f7f", lw=0
                    )
                )
    for _, _, end in bounds[:-1]:
        ax.axhline(end - 0.5, color="white", lw=2.2)
    ax.set_xticks(range(3), ["第 1 名", "第 2 名", "第 3 名"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)), [f"{d}  {grade[d]}" for d in order], fontsize=6.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for g, start, end in bounds:
        tall = end - start >= 6
        label = (
            pf.GROUP_LABELS[g] if tall else pf.GROUP_LABELS[g][:2] + "\n" + pf.GROUP_LABELS[g][2:]
        )
        ax.text(
            -0.62,
            (start + end - 1) / 2,
            label,
            rotation=90 if tall else 0,
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=7,
            linespacing=0.95,
        )
    pf.panel_label(ax, "a  化合物层入选频率 $\\pi^c$", y=1.05)

    ax = axes[1]
    counts = np.array(
        [
            [
                (pi[:, j] >= hi).sum(),
                ((pi[:, j] >= lo) & (pi[:, j] < hi)).sum(),
                (pi[:, j] < lo).sum(),
            ]
            for j in range(3)
        ]
    )
    x = np.arange(3)
    bottom = np.zeros(3)
    for t, (lab, col) in enumerate(zip(("稳定", "中等", "不稳定"), TIER_COLORS[::-1])):
        ax.bar(
            x,
            counts[:, t],
            bottom=bottom,
            width=0.55,
            color=col,
            label=lab,
            edgecolor="white",
            linewidth=1.0,
        )
        for xi, (b0, v) in enumerate(zip(bottom, counts[:, t])):
            if v:
                ax.text(
                    xi,
                    b0 + v / 2,
                    str(int(v)),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if t == 0 else pf.INK,
                )
        bottom += counts[:, t]
    ax.set_xticks(x, ["第 1 名", "第 2 名", "第 3 名"])
    ax.set_ylabel("维度数")
    ax.set_ylim(0, len(order) * 1.02)
    ax2 = ax.twinx()
    pic = m.groupby("rank")["pi_c"].mean().reindex([1, 2, 3]).to_numpy()
    pim = m.groupby("rank")["pi_m"].mean().reindex([1, 2, 3]).to_numpy()
    ax2.plot(x, pic, "o-", color=pf.ORANGE, lw=1.4, ms=5, label="$\\pi^c$ 均值")
    ax2.plot(x, pim, "D--", color=pf.INK, lw=1.0, ms=4, label="$\\pi^m$ 均值")
    for xi, (a, b) in enumerate(zip(pic, pim)):
        ax2.text(xi + 0.32, a - 0.035, f"{a:.2f}", color=pf.ORANGE, fontsize=7, va="top")
        ax2.text(xi + 0.32, b + 0.035, f"{b:.2f}", color=pf.INK, fontsize=7, va="bottom")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("平均入选频率")
    ax2.spines["right"].set_visible(True)
    ax.set_box_aspect(1.15)
    ax2.set_box_aspect(1.15)
    pf.panel_label(ax, "b  三个名次的稳定程度")
    handles = [
        Patch(color=TIER_COLORS[2], label=f"稳定，$\\pi^c\\geq{hi:g}$"),
        Patch(color=TIER_COLORS[1], label=f"中等，$1/3\\leq\\pi^c<{hi:g}$"),
        Patch(color=TIER_COLORS[0], label="不稳定，$\\pi^c<1/3$"),
        Patch(facecolor="white", edgecolor="#7f7f7f", hatch="////", label="不显著，a 栏"),
        Line2D([], [], color=pf.ORANGE, marker="o", lw=1.4, ms=5, label="化合物层 $\\pi^c$ 均值"),
        Line2D(
            [], [], color=pf.INK, marker="D", ls="--", lw=1.0, ms=4, label="模块层 $\\pi^m$ 均值"
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, -0.12),
        ncol=2,
        handlelength=1.3,
        columnspacing=1.0,
    )
    fig.text(
        0.02,
        0.005,
        "a 栏行名后为维度等级：A 首位显著且 $\\pi^c\\geq0.6$，B 首位显著，C 首位不显著",
        fontsize=7,
        color=pf.MUTED,
    )
    fig.subplots_adjust(left=0.2, right=0.92, top=0.95, bottom=0.04, wspace=0.55)
    print(pf.save(fig, "q2", "fig_q2_stability.pdf"))


if __name__ == "__main__":
    main()
