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
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
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
    ax.scatter([], [], color=p["main"], label="主模型")
    ax.scatter([], [], marker="_", color=p["ctrl"], label="B0")
    ax.scatter([], [], marker="x", color=p["warn"], label="B1")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [f"{e['style']}（{e['n']}）" for e in rows], rotation=60, ha="right", fontsize=6
    )
    ax.set_ylabel("留一风格等级 MAE")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.set_title("留一风格压力测试；空心点为款数不超过 5 的风格，只作定性解读", fontsize=8)
    print(plotstyle.save(fig, args.out, "fig_q1_stress.pdf"))


if __name__ == "__main__":
    main()
