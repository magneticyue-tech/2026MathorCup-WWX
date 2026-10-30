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
    ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(
        fontsize=9,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.10, 1.0),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.2,
        columnspacing=1.0,
    )


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    ref = plotstyle.load_json(args.src / "q1_metrics.json")["reference_error"]
    x = np.arange(len(CLASSES))
    labels = [f"{name}\n{ref[k]['cells_per_rep']:,.0f} 单元" for k, name in CLASSES]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), constrained_layout=True)
    p = plotstyle.PALETTE
    bars(
        axes[0],
        ref,
        (("exact", "完全一致率", p["main"]), ("within1", "±1 级以内", p["accent"])),
        x,
        labels,
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("比例", fontsize=9)
    axes[0].set_title("(a)", fontsize=11, loc="left")
    bars(
        axes[1],
        ref,
        (
            ("bias", "平均偏差（混合参照 $R$ 减实测等级 $G$）", p["warn"]),
            ("mae", "等级 MAE", p["ctrl"]),
        ),
        x,
        labels,
    )
    axes[1].axhline(0, color="#444444", lw=0.8)
    axes[1].set_ylabel("等级误差（级）", fontsize=9)
    axes[1].set_title("(b)", fontsize=11, loc="left")
    fig.supxlabel("参照类别（只用附件 4 的两阶段方案）", fontsize=9)
    print(plotstyle.save(fig, args.out, "fig_q1_reference_error.pdf"))


if __name__ == "__main__":
    main()
