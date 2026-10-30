import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q3.config import NAMED_STYLES  # noqa: E402
from src.q3.paths import PROCESSED_DIR, Q1_DIR  # noqa: E402

from src import paperfig as pf  # noqa: E402


def main():
    pf.apply()
    z = np.load(PROCESSED_DIR / "q3_portraits.npz", allow_pickle=False)
    d = np.load(Q1_DIR / "q1_data.npz", allow_pickle=False)
    styles = [str(s) for s in z["styles"]]
    dims = [str(x) for x in z["dims"]]
    style = np.array([str(s) for s in d["style"]])
    xbar = d["xbar"]
    weak = z["weak_dim"].astype(bool)
    order, bounds = [], []
    for g, members in DIM_GROUPS.items():
        start = len(order)
        order += [dims.index(c) for c in members]
        bounds.append((g, start, len(order)))
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, len(NAMED_STYLES), figsize=(pf.TEXTWIDTH, 8.4), sharey=True)
    for ax, st in zip(axes, NAMED_STYLES):
        si = styles.index(st)
        q10, q50, q90 = (z[k][si, order] for k in ("mu_q10", "mu_q50", "mu_q90"))
        ax.hlines(y, q10, q90, color=pf.SKY, lw=1.6)
        ax.plot(q50, y, "o", color=pf.BLUE, ms=3.2)
        ax.plot(z["real_oof_q50"][si, order], y, "D", mfc="none", mec=pf.ORANGE, ms=3.4, mew=0.8)
        ax.plot(np.median(xbar[style == st], 0)[order], y, "x", color=pf.INK, ms=3.4, mew=0.8)
        for _, _, end in bounds[:-1]:
            ax.axhline(end - 0.5, color=pf.GRID, lw=0.8)
        ax.set_xlim(0, 5)
        ax.grid(axis="x", visible=True)
        ax.set_axisbelow(True)
        ax.set_title(f"{st}，真实 {int(z['n_style'][si])} 款", fontsize=8.5)
        ax.set_xlabel("预测评分 0–6")
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    axes[0].set_yticks(y, [dims[i] for i in order], fontsize=6.3)
    for t, i in zip(axes[0].get_yticklabels(), order):
        if weak[i]:
            t.set_color(pf.LIGHT)
    for g, start, end in bounds:
        axes[0].text(
            -0.5,
            (start + end - 1) / 2,
            pf.GROUP_LABELS[g],
            rotation=90,
            transform=axes[0].get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=7,
        )
    n = z["mu"].shape[1]
    handles = [
        Line2D([], [], color=pf.SKY, lw=1.6, label=f"生成画像 10%–90%，{n} 个化合物谱"),
        Line2D([], [], marker="o", ls="", color=pf.BLUE, ms=4, label="生成画像中位数"),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            mfc="none",
            mec=pf.ORANGE,
            ms=4,
            label="真实啤酒折外预测中位数",
        ),
        Line2D([], [], marker="x", ls="", color=pf.INK, ms=4, label="真实啤酒观测均分中位数"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.005))
    fig.text(
        0.02,
        -0.03,
        f"灰色维度名：问题一化学模型未胜过风格基线，共 {int(weak.sum())} 维",
        fontsize=7,
        color=pf.MUTED,
    )
    fig.subplots_adjust(left=0.2, right=0.98, top=0.96, bottom=0.1, wspace=0.08)
    print(pf.save(fig, "q3", "fig_q3_portrait.pdf"))


if __name__ == "__main__":
    main()
