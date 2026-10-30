import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.discretize import N_GRADES  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402


def confusion(G, Gh, mask):
    cm = np.zeros((N_GRADES, N_GRADES))
    np.add.at(cm, (np.broadcast_to(G, Gh.shape)[mask], Gh[mask]), 1)
    return cm


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    G = plotstyle.load_npz(args.src / "q1_data.npz")["G"]
    Gh = plotstyle.load_npz(args.src / "q1_scores.npz")["grade_median"]
    anchored = plotstyle.load_npz(args.src / "q1_twostage.npz")["I_te"].astype(bool)
    R = Gh.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    panels = (
        (axes[0], anchored, "(a) 有零点参照的单元"),
        (axes[1], ~anchored, "(b) 无零点参照的单元"),
    )
    for ax, mask, title in panels:
        cm = confusion(G[None], Gh, mask)
        rows = cm.sum(1, keepdims=True)
        share = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
        im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(N_GRADES):
            for j in range(N_GRADES):
                if cm[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        f"{share[i, j]:.2f}\n{cm[i, j] / R:.0f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if share[i, j] > 0.6 else "#222222",
                    )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("预测等级（浅层梯度提升树，分布中位数）", fontsize=9)
        ax.set_ylabel("实测等级", fontsize=9)
        ax.set_xticks(range(N_GRADES))
        ax.set_yticks(range(N_GRADES))
        ax.set_xticks(np.arange(-0.5, N_GRADES, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_GRADES, 1), minor=True)
        ax.tick_params(labelsize=9)
        ax.tick_params(which="minor", length=0)
        ax.grid(False)
        ax.grid(which="minor", color="#dddddd", linewidth=0.8)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(
        "颜色：行内比例（单元格上行）；每次重复的平均单元数（下行）",
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=9)
    print(plotstyle.save(fig, args.out, "fig_q1_confusion_anchor.pdf"))


if __name__ == "__main__":
    main()
