import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    arms = plotstyle.load_json(args.src / "q1_sensitivity.json")["arms"]
    names = list(arms)
    labels = [plotstyle.ARM_LABELS.get(a, a) for a in names]
    colors = [
        plotstyle.PALETTE["accent"] if a == "alpha" else plotstyle.PALETTE["main"] for a in names
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, key, xlab in (
        (axes[0], "mae", "等级 MAE 配对差（臂减主设定）"),
        (axes[1], "qwk", "QWK 配对差（臂减主设定）"),
    ):
        plotstyle.forest(
            ax,
            labels,
            [arms[a]["point"][key] for a in names],
            [arms[a]["lo"][key] for a in names],
            [arms[a]["hi"][key] for a in names],
            colors,
        )
        ax.set_xlabel(xlab)
    xmax = axes[1].get_xlim()[1]
    for yi, a in zip(range(len(names))[::-1], names):
        axes[1].text(
            xmax, yi, f" 交付变化 {arms[a]['changed_delivered']:.1%}", va="center", fontsize=6
        )
    print(plotstyle.save(fig, args.out, "fig_q1_sensitivity.pdf"))


if __name__ == "__main__":
    main()
