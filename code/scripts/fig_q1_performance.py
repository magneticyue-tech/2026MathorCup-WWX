import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    z = plotstyle.load_npz(args.src / "q1_spearman.npz")
    p = plotstyle.PALETTE
    models = [str(m) for m in z["models"]]
    dims = [str(d) for d in z["dims"]]
    pt, lo, hi = z["point"], z["lo"], z["hi"]
    dp, dlo, dhi = z["diff_point"], z["diff_lo"], z["diff_hi"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    o = np.argsort(pt, kind="stable")
    y = np.arange(len(models))
    for yi, i in zip(y, o):
        if models[i] == "main":
            c = p["main"]
        elif dlo[i] > 0:
            c = p["better"]
        elif dhi[i] < 0:
            c = p["neutral"]
        else:
            c = p["warn"]
        ax.plot([lo[i], hi[i]], [yi, yi], color=c, lw=1.6)
        ax.plot(pt[i], yi, "o", color=c, ms=5)
        tag = "" if models[i] == "main" else f"Δ {dp[i]:+.3f} [{dlo[i]:+.3f}, {dhi[i]:+.3f}]"
        if tag:
            ax.text(
                hi[i] + 0.012,
                yi,
                tag,
                va="center",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
            )
    ax.axvline(pt[models.index("main")], ls="--", color=p["main"], lw=0.8)
    ax.axvline(0, color="#444444", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([plotstyle.MODEL_LABELS.get(models[i], models[i]) for i in o])
    ax.set_xlabel(
        "折外连续预测与品评均分的平均 Spearman（95% 日期组 bootstrap 区间）\n"
        "Δ 为相对浅层梯度提升树的配对差，正值表示更高",
        fontsize=9,
    )
    ax.set_title("(a)", fontsize=11, loc="left")
    handles = [
        Line2D([], [], color=p["main"], marker="o", lw=0, ms=6, label="浅层梯度提升树（参照）"),
        Line2D([], [], color=p["better"], marker="o", lw=0, ms=6, label="显著高于浅层梯度提升树"),
        Line2D([], [], color=p["neutral"], marker="o", lw=0, ms=6, label="显著低于浅层梯度提升树"),
        Line2D([], [], color=p["warn"], marker="o", lw=0, ms=6, label="与浅层梯度提升树无显著差异"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(min(lo.min(), 0) - 0.03, hi.max() + 0.40)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    a, b = z["per_dim"][models.index("main")], z["per_dim"][models.index("b1")]
    group_cn = {
        "Aroma": "香气区块",
        "Flavour": "风味区块",
        "Defect": "缺陷",
        "Spice": "香料啤酒",
        "Sour": "酸啤酒",
        "Basic": "基本味、口感、总体",
    }
    for g, members in DIM_GROUPS.items():
        idx = [dims.index(d) for d in members]
        ax.scatter(
            b[idx],
            a[idx],
            s=22,
            marker=plotstyle.GROUP_MARKERS[g],
            label=f"{g}（{group_cn[g]}）",
            alpha=0.85,
        )
    lim = [min(a.min(), b.min()) - 0.05, max(a.max(), b.max()) + 0.05]
    ax.plot(lim, lim, ls="--", color="#444444", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("风格收缩均值：逐维 Spearman", fontsize=9)
    ax.set_ylabel("浅层梯度提升树：逐维 Spearman", fontsize=9)
    ax.set_title("(b)", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    print(plotstyle.save(fig, args.out, "fig_q1_performance.pdf"))


if __name__ == "__main__":
    main()
