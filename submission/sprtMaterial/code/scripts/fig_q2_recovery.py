import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q2 import config as C  # noqa: E402

METHODS = (("main", "图约束前向选择（主方法）", "-"), ("M1", "图约束边际选择（M1）", "--"))
POS_COLORS = ("#0072B2", "#E69F00", "#009E73")


def main():
    plotstyle.apply()
    sim = pd.read_csv(C.Q2_DIR / "simulation_recovery_results.csv")
    final = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    main_rows = final[final["method"] == "main"]
    obs_r = float(main_rows.loc[main_rows["rank"] == 1, "score"].abs().median())

    weights = list(dict.fromkeys(sim["weights"]))
    fig, axes = plt.subplots(1, len(weights), figsize=(10, 3.8), sharey=True)
    for ax, w in zip(axes, weights):
        g = sim[sim["weights"] == w].sort_values("target_r")
        x = g["observed_r1_median"].to_numpy()
        for key, label, ls in METHODS:
            for pos, col in zip((1, 2, 3), POS_COLORS):
                ax.plot(
                    x,
                    g[f"{key}_comp_recovery_{pos}"],
                    ls=ls,
                    marker="o" if key == "main" else "s",
                    color=col,
                    label=f"{label}·第 {pos} 位" if ax is axes[0] else None,
                )
        ax.axvline(obs_r, color="#444444", ls=":", lw=1)
        ax.text(obs_r, 1.02, f"观测 Top-1 |r| 中位 {obs_r:.2f}", ha="center", fontsize=7)
        ax.set_title(f"植入权重 ({w})；每设置 {int(g['n_sim'].iloc[0])} 次", fontsize=8)
        ax.set_xlabel("植入首位化合物与伪维度的实现 |r|（中位）")
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("按植入顺序的化合物层找回率")
    fig.legend(loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("两种选择规则的找回率对比（实线：前向选择；虚线：边际选择）", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    print(plotstyle.save(fig, C.FIGURES, "fig_q2_recovery.pdf"))


if __name__ == "__main__":
    main()
