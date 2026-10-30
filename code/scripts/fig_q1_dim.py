import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q1.paths import PROCESSED_DIR  # noqa: E402

from src import paperfig as pf  # noqa: E402


def main():
    pf.apply()
    m = plotstyle.load_npz(PROCESSED_DIR / "q1_metrics.npz")
    d = plotstyle.load_npz(PROCESSED_DIR / "q1_data.npz")
    r2 = np.asarray(plotstyle.load_json(PROCESSED_DIR / "q1_metrics.json")["r2_main"], float)
    dims = [str(x) for x in d["dims"]]
    diff = m["main__median__mae_dim"] - m["b1__median__mae_dim"]
    rel = d["reliability"]

    order, bounds = [], []
    for g, members in DIM_GROUPS.items():
        start = len(order)
        order += [dims.index(c) for c in members]
        bounds.append((g, start, len(order)))
    y = np.arange(len(order))

    fig, axes = plt.subplots(
        1, 2, figsize=(pf.TEXTWIDTH, 8.4), sharey=True, gridspec_kw={"width_ratios": [1, 1]}
    )
    ax = axes[0]
    dv = diff[order]
    ax.barh(y, dv, height=0.7, color=[pf.BLUE if v < 0 else pf.ORANGE for v in dv])
    ax.axvline(0, color=pf.INK, lw=0.7)
    ax.set_xlabel("逐维等级 MAE 差：主模型减风格收缩基线")
    pf.panel_label(ax, f"a  主模型胜出 {int((diff < 0).sum())} / {len(dims)} 维")
    ax.legend(
        handles=[
            Patch(color=pf.BLUE, label="主模型更低"),
            Patch(color=pf.ORANGE, label="风格收缩基线更低"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.11),
        ncol=2,
        handlelength=1.0,
        columnspacing=1.0,
    )

    ax = axes[1]
    rv, lv = r2[order], rel[order]
    ax.barh(y, rv, height=0.7, color=[pf.SKY if v >= 0 else pf.LIGHT for v in rv])
    ax.scatter(lv, y, marker="|", s=55, color=pf.INK, linewidths=1.2, zorder=3)
    ax.axvline(0, color=pf.INK, lw=0.7)
    ax.set_xlabel("逐维连续 $R^2$ 与半组信度")
    pf.panel_label(
        ax,
        f"b  $R^2$ 均值 {r2.mean():.3f}，信度中位数 {np.median(rel):.2f}",
    )
    ax.legend(
        handles=[
            Patch(color=pf.SKY, label="连续 $R^2$"),
            Patch(color=pf.LIGHT, label=f"$R^2<0$，共 {int((r2 < 0).sum())} 维"),
            Line2D([], [], marker="|", ls="", color=pf.INK, ms=8, mew=1.2, label="半组信度"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.11),
        ncol=3,
        handlelength=1.0,
        columnspacing=1.0,
    )

    for ax in axes:
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.grid(axis="x", visible=True)
        ax.set_axisbelow(True)
        for _, _, end in bounds[:-1]:
            ax.axhline(end - 0.5, color=pf.GRID, lw=0.8)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([dims[i] for i in order], fontsize=6.5)
    for g, start, end in bounds:
        wins = int((diff[order[start:end]] < 0).sum())
        axes[0].text(
            -0.62,
            (start + end - 1) / 2,
            f"{pf.GROUP_LABELS[g]}\n{wins}/{end - start}",
            transform=axes[0].get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=7.5,
            color=pf.INK,
        )
    fig.subplots_adjust(left=0.2, right=0.98, top=0.96, bottom=0.08, wspace=0.08)
    print(pf.save(fig, "q1", "fig_q1_dim.pdf"))


if __name__ == "__main__":
    main()
