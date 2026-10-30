import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402


def panel_layout(panels):
    sizes = np.bincount(panels)
    multi = [g for g in np.argsort(-sizes, kind="stable") if sizes[g] > 1]
    ncol = int(np.ceil(np.sqrt(len(multi) * 1.6)))
    pos = np.full((len(panels), 2), np.nan)
    for i, g in enumerate(multi):
        cx, cy = i % ncol, -(i // ncol)
        m = np.where(panels == g)[0]
        ang = 2 * np.pi * np.arange(len(m)) / len(m)
        pos[m] = np.column_stack([cx + 0.36 * np.cos(ang), cy + 0.36 * np.sin(ang)])
    single = np.where(sizes[panels] == 1)[0]
    base = -(len(multi) // ncol) - 1.2
    pos[single] = np.column_stack([np.arange(len(single)) * 0.5, np.full(len(single), base)])
    return pos, multi, single


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    z = plotstyle.load_npz(args.src / "q1_panels.npz")
    p = plotstyle.PALETTE
    dims = [str(d) for d in z["dims"]]
    panels, edges, at_zero, Z = z["panels"], z["edges"], z["at_zero"], z["z"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 2.4]})
    ax = axes[0]
    pos, multi, single = panel_layout(panels)
    for i, j, _ in edges:
        ax.plot(*pos[[i, j]].T, color=p["neutral"], lw=0.4, alpha=0.6, zorder=1)
    in_multi = np.isin(panels, multi)
    ax.scatter(*pos[in_multi].T, s=16, color=p["main"], zorder=2, label="品评团成员")
    ax.scatter(*pos[single].T, s=16, color=p["warn"], zorder=2, label="未归入品评团")
    ax.set_title("(a)", fontsize=11, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_axis_off()
    ax.set_aspect("equal")

    ax = axes[1]
    d = int(np.argmax(at_zero.mean(axis=0)))
    bins = np.histogram_bin_edges(Z[:, d], bins=30)
    ax.hist(
        [Z[at_zero[:, d], d], Z[~at_zero[:, d], d]],
        bins=bins,
        stacked=True,
        color=[p["better"], p["main"]],
        label=[
            f"位于品评团零点上的啤酒（{at_zero[:, d].mean():.0%}）",
            "不在零点上的啤酒",
        ],
    )
    ax.set_xlabel(f"附件 4 标准化品评均分 z（{dims[d]}）", fontsize=9)
    ax.set_ylabel("啤酒数", fontsize=9)
    ax.set_title("(b)", fontsize=11, loc="left")
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=9)

    ax = axes[2]
    cov, share = z["coverage"], at_zero.mean(axis=0)
    o = np.argsort(-cov, kind="stable")
    x = np.arange(len(dims))
    ax.bar(x, cov[o], color=p["main"], label="有可用零点的啤酒比例")
    ax.bar(x, share[o], color=p["better"], width=0.5, label="恰在零点上的啤酒比例")
    ax.set_xticks(x)
    ax.set_xticklabels([dims[i] for i in o], rotation=90, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("比例", fontsize=9)
    ax.set_title("(c)", fontsize=11, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    print(plotstyle.save(fig, args.out, "fig_q1_panel.pdf"))


if __name__ == "__main__":
    main()
