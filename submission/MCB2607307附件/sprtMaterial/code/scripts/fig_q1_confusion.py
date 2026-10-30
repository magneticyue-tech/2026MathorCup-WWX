import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402

PANELS = (
    ("main_median", "主模型（中位数）"),
    ("k2_median", "K2（中位数）"),
    ("c1_sel", "C1（选定映射）"),
)


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    z = plotstyle.load_npz(args.src / "q1_metrics.npz")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), constrained_layout=True)
    im = None
    for ax, (key, title) in zip(axes, PANELS):
        cm = z[f"conf_{key}"]
        rows = cm.sum(1, keepdims=True)
        frac = np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)
        im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(
                    j,
                    i,
                    f"{int(round(cm[i, j]))}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if frac[i, j] > 0.5 else "black",
                )
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xlabel("预测等级")
        ax.grid(False)
    axes[0].set_ylabel("真实等级")
    fig.colorbar(im, ax=list(axes), shrink=0.85, label="行归一化比例")
    print(plotstyle.save(fig, args.out, "fig_q1_confusion.pdf"))


if __name__ == "__main__":
    main()
