import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, key, xlab in ((axes[0], "mae", "等级 MAE 配对差"), (axes[1], "qwk", "QWK 配对差")):
        pt = [p["point"][key] for p in rows]
        lo = [p["lo"][key] for p in rows]
        hi = [p["hi"][key] for p in rows]
        if key == "mae":
            colors = [plotstyle.significance_color(a, b) for a, b in zip(lo, hi)]
        else:
            colors = [plotstyle.significance_color(-b, -a) for a, b in zip(lo, hi)]
        plotstyle.forest(ax, labels, pt, lo, hi, colors)
        ax.set_xlabel(xlab)
    axes[0].set_title("对照减主模型：MAE 为正表示对照更差", fontsize=8)
    axes[1].set_title("对照减主模型：QWK 为负表示对照更差", fontsize=8)
    print(plotstyle.save(fig, args.out, "fig_q1_paired_diff.pdf"))


if __name__ == "__main__":
    main()
