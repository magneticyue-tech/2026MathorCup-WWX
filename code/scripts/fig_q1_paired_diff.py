import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402

CTRLS = ("b0", "b1", "lin10", "enet", "rf", "chemstyle", "k2", "c1", "c2")


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    paired = plotstyle.load_json(args.src / "q1_metrics.json")["paired"]
    rows = [p for p in paired if p["b"] == "main.median" and p["a"].split(".")[0] in CTRLS]
    rows.sort(key=lambda p: p["point"]["mae"], reverse=True)
    labels = [plotstyle.MODEL_LABELS[p["a"].split(".")[0]] for p in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True, constrained_layout=True)
    panels = (
        (axes[0], "mae", "等级 MAE 配对差\n（各对照减浅层梯度提升树）", "(a)"),
        (axes[1], "qwk", "二次加权 Kappa 配对差\n（各对照减浅层梯度提升树）", "(b)"),
    )
    for ax, key, xlab, panel in panels:
        pt = [p["point"][key] for p in rows]
        lo = [p["lo"][key] for p in rows]
        hi = [p["hi"][key] for p in rows]
        if key == "mae":
            colors = [plotstyle.significance_color(a, b) for a, b in zip(lo, hi)]
        else:
            colors = [plotstyle.significance_color(-b, -a) for a, b in zip(lo, hi)]
        plotstyle.forest(ax, labels, pt, lo, hi, colors)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_title(panel, fontsize=11, loc="left")
        ax.tick_params(labelsize=9)
    p = plotstyle.PALETTE
    handles = [
        Line2D(
            [], [], color=p["better"], marker="o", lw=1.4, ms=4, label="95% 区间不含 0：对照更好"
        ),
        Line2D(
            [], [], color=p["worse"], marker="o", lw=1.4, ms=4, label="95% 区间不含 0：对照更差"
        ),
        Line2D([], [], color=p["neutral"], marker="o", lw=1.4, ms=4, label="95% 区间含 0"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.15),
    )
    print(plotstyle.save(fig, args.out, "fig_q1_paired_diff.pdf"))


if __name__ == "__main__":
    main()
