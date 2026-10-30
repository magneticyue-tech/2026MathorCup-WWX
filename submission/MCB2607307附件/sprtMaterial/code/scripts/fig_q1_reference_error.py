import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402

CLASSES = (("all", "全部单元"), ("A", "局部参照"), ("L", "线性参照"))


def bars(ax, ref, specs, x, labels):
    w = 0.38
    for off, (key, name, color) in zip((-w / 2, w / 2), specs):
        ax.bar(x + off, [ref[k][key] for k, _ in CLASSES], w, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=7, frameon=False)


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    ref = plotstyle.load_json(args.src / "q1_metrics.json")["reference_error"]
    x = np.arange(len(CLASSES))
    labels = [f"{name}\n{ref[k]['cells_per_rep']:.0f} 格" for k, name in CLASSES]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    p = plotstyle.PALETTE
    bars(
        axes[0],
        ref,
        (("exact", "完全一致率", p["main"]), ("within1", "±1 以内", p["accent"])),
        x,
        labels,
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("比例")
    bars(
        axes[1],
        ref,
        (("bias", "平均偏差 $R-G$", p["warn"]), ("mae", "等级 MAE", p["ctrl"])),
        x,
        labels,
    )
    axes[1].axhline(0, color="#444444", lw=0.8)
    axes[1].set_ylabel("等级")
    fig.suptitle("C1 混合参照 $R$ 相对真值 $G$ 的误差（格数为每次重复的单元数）", fontsize=9)
    print(plotstyle.save(fig, args.out, "fig_q1_reference_error.pdf"))


if __name__ == "__main__":
    main()
