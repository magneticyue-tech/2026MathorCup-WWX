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

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), constrained_layout=True)
    for ax, mask, name in ((axes[0], anchored, "有锚单元"), (axes[1], ~anchored, "无锚单元")):
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
                        fontsize=5.5,
                        color="white" if share[i, j] > 0.6 else "#222222",
                    )
        err = np.abs(np.broadcast_to(G[None], Gh.shape) - Gh)[mask]
        ax.set_title(
            f"{name}：每次重复 {mask.sum() / R:.0f} 格；等级 MAE {err.mean():.3f}，"
            f"完全一致 {(err == 0).mean():.1%}",
            fontsize=8,
        )
        ax.set_xlabel("预测等级（主模型，分布中位数）")
        ax.set_ylabel("真值等级 G")
        ax.set_xticks(range(N_GRADES))
        ax.set_yticks(range(N_GRADES))
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="行内比例（格内第二行为每次重复的格数）")
    print(plotstyle.save(fig, args.out, "fig_q1_confusion_anchor.pdf"))


if __name__ == "__main__":
    main()
