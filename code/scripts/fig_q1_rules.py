import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402

RULES = ("const", "round", "median", "quant", "qwk_cut", "median_knn")
LABEL_OFFSETS = {
    "const": (6, 8, "left", "bottom"),
    "round": (6, -10, "left", "top"),
    "median": (6, 10, "left", "bottom"),
    "quant": (-6, 6, "right", "bottom"),
    "qwk_cut": (6, -10, "left", "top"),
    "median_knn": (6, -10, "left", "top"),
}


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    models = plotstyle.load_json(args.src / "q1_metrics.json")["models"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for rule in RULES:
        m = models[f"main.{rule}"]
        is_main = rule == "median"
        color = plotstyle.PALETTE["main"] if is_main else plotstyle.PALETTE["ctrl"]
        ax.scatter(m["mae"], m["qwk"], s=40 if is_main else 24, color=color, zorder=3)
        label = plotstyle.RULE_LABELS[rule] + ("（交付设定）" if is_main else "")
        dx, dy, ha, va = LABEL_OFFSETS[rule]
        ax.annotate(
            label,
            (m["mae"], m["qwk"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            ha=ha,
            va=va,
            fontweight="bold" if is_main else "normal",
            color=color,
        )
    xs = [models[f"main.{rule}"]["mae"] for rule in RULES]
    ys = [models[f"main.{rule}"]["qwk"] for rule in RULES]
    ax.set_xlim(min(xs) - 0.012, max(xs) + 0.07)
    ax.set_ylim(min(ys) - 0.05, max(ys) + 0.065)
    ax.set_xlabel("等级 MAE（越小越好）", fontsize=9)
    ax.set_ylabel("二次加权 Kappa（越大越好）", fontsize=9)
    ax.tick_params(labelsize=9)
    print(plotstyle.save(fig, args.out, "fig_q1_rules.pdf"))


if __name__ == "__main__":
    main()
