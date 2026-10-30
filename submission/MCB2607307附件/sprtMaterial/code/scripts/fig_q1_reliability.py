import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.data import DIM_GROUPS, DIMS, ZERO_HEAVY_DIMS  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    j = plotstyle.load_json(args.src / "q1_metrics.json")
    rel, r2 = np.array(j["reliability"], float), np.array(j["r2_main"], float)
    color = plotstyle.PALETTE["main"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for g, dims in DIM_GROUPS.items():
        marker = plotstyle.GROUP_MARKERS[g]
        for d in dims:
            i = DIMS.index(d)
            ax.scatter(
                rel[i],
                r2[i],
                marker=marker,
                s=18,
                facecolors="none" if d in ZERO_HEAVY_DIMS else color,
                edgecolors=color,
                linewidths=0.8,
            )
        ax.scatter([], [], marker=marker, color=color, label=g, s=14)
    lim = [min(rel.min(), r2.min(), 0.0) - 0.02, max(rel.max(), r2.max()) + 0.02]
    ax.plot(lim, lim, ls="--", color="#444444", lw=0.8)
    marked = set(np.argsort(rel)[:4].tolist()) | {DIMS.index(d) for d in j["leak_dims"]}
    for n, i in enumerate(sorted(marked, key=lambda i: r2[i])):
        ax.annotate(
            DIMS[i],
            (rel[i], r2[i]),
            xytext=(-38, 10 + 12 * n),
            textcoords="offset points",
            fontsize=6,
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#666666"},
        )
    ax.set_xlabel(r"信度 $\rho_d$")
    ax.set_ylabel(r"折外 $R^2$")
    ax.legend(
        fontsize=6,
        frameon=False,
        title="空心为零分为主维度",
        title_fontsize=6,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    print(plotstyle.save(fig, args.out, "fig_q1_reliability.pdf"))


if __name__ == "__main__":
    main()
