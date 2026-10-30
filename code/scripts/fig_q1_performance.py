import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
    dlo, dhi = z["diff_lo"], z["diff_hi"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
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
        tag = (
            ""
            if models[i] == "main"
            else f"  Δ {pt[i] - pt[0]:+.3f} [{dlo[i]:+.3f}, {dhi[i]:+.3f}]"
        )
        ax.text(hi[i] + 0.01, yi, tag, va="center", fontsize=6.5)
    ax.axvline(pt[models.index("main")], ls="--", color=p["main"], lw=0.8)
    ax.axvline(0, color="#444444", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([plotstyle.MODEL_LABELS.get(models[i], models[i]) for i in o])
    ax.set_xlabel(r"折外连续预测与 $\bar{x}$ 的平均 Spearman（95% 日期组 bootstrap 区间）")
    ax.set_title(
        "(a) 模型对比；Δ 为减主模型的配对差：橙色显著高于、灰色显著低于、黄色无差异",
        fontsize=8,
    )
    ax.set_xlim(min(lo.min(), 0) - 0.03, hi.max() + 0.32)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    a, b = z["per_dim"][models.index("main")], z["per_dim"][models.index("b1")]
    for g, members in DIM_GROUPS.items():
        idx = [dims.index(d) for d in members]
        ax.scatter(b[idx], a[idx], s=18, marker=plotstyle.GROUP_MARKERS[g], label=g, alpha=0.85)
    lim = [min(a.min(), b.min()) - 0.05, max(a.max(), b.max()) + 0.05]
    ax.plot(lim, lim, ls="--", color="#444444", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("B1 风格收缩均值：逐维 Spearman")
    ax.set_ylabel("主模型：逐维 Spearman")
    ax.set_title(f"(b) 逐维对比：主模型在 {int((a > b).sum())}/{len(dims)} 维更高", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    print(plotstyle.save(fig, args.out, "fig_q1_performance.pdf"))


if __name__ == "__main__":
    main()
