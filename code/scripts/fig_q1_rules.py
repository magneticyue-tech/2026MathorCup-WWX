import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402

RULES = ("const", "round", "median", "quant", "qwk_cut", "median_knn")
OFFSETS = {"median": (4, 6), "round": (4, -10), "median_knn": (4, -10)}


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    models = plotstyle.load_json(args.src / "q1_metrics.json")["models"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for rule in RULES:
        m = models[f"main.{rule}"]
        is_main = rule == "median"
        color = plotstyle.PALETTE["main"] if is_main else plotstyle.PALETTE["ctrl"]
        ax.scatter(m["mae"], m["qwk"], s=36 if is_main else 22, color=color, zorder=3)
        label = plotstyle.RULE_LABELS[rule] + ("（主输出）" if is_main else "")
        ax.annotate(
            label,
            (m["mae"], m["qwk"]),
            xytext=OFFSETS.get(rule, (4, 3)),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold" if is_main else "normal",
            color=color,
        )
    ax.set_xlabel("等级 MAE（越小越好）")
    ax.set_ylabel("QWK（越大越好）")
    ax.set_title("离散规则比较（左上更好）", fontsize=9)
    print(plotstyle.save(fig, args.out, "fig_q1_rules.pdf"))


if __name__ == "__main__":
    main()
