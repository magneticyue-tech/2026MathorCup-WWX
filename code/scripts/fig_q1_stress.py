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
    rows = plotstyle.load_json(args.src / "q1_stress.json")["per_style"]
    rows = sorted(rows, key=lambda e: e["mae_median"])
    p = plotstyle.PALETTE
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    for yi, e in enumerate(rows):
        ax.scatter(
            yi,
            e["mae_median"],
            s=26,
            facecolors="none" if e["small"] else p["main"],
            edgecolors=p["main"],
            zorder=3,
        )
        ax.scatter(yi, e["b0_mae"], marker="_", s=60, color=p["ctrl"])
        ax.scatter(yi, e["b1_mae"], marker="x", s=16, color=p["warn"])
    ax.scatter([], [], color=p["main"], label=plotstyle.MODEL_LABELS["main"])
    ax.scatter([], [], marker="_", color=p["ctrl"], label=plotstyle.MODEL_LABELS["b0"])
    ax.scatter([], [], marker="x", color=p["warn"], label=plotstyle.MODEL_LABELS["b1"])
    ax.scatter(
        [],
        [],
        facecolors="none",
        edgecolors=p["main"],
        label="啤酒数不超过 5 的风格",
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [f"{e['style']}（n={e['n']}）" for e in rows], rotation=65, ha="right", fontsize=9
    )
    ax.set_ylabel("留一风格等级 MAE")
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    print(plotstyle.save(fig, args.out, "fig_q1_stress.pdf"))


if __name__ == "__main__":
    main()
