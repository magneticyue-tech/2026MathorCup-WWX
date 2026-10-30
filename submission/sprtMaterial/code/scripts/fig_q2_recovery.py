import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q2 import config as C  # noqa: E402

from src import paperfig as pf  # noqa: E402

METHODS = (("main", "前向选择", pf.BLUE, "o", "-"), ("M1", "图约束边际选择", pf.ORANGE, "s", "--"))


def main():
    pf.apply()
    sim = pd.read_csv(C.Q2_DIR / "simulation_recovery_results.csv")
    final = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    top1 = final[(final["method"] == "main") & (final["rank"] == 1)]["score"].abs()
    q1, med, q3 = np.percentile(top1, [25, 50, 75])
    weights = list(dict.fromkeys(sim["weights"]))
    n_sim = int(sim["n_sim"].iloc[0])

    fig, axes = plt.subplots(2, 3, figsize=(pf.TEXTWIDTH, 4.3), sharex=True, sharey=True)
    for i, w in enumerate(weights):
        g = sim[sim["weights"] == w].sort_values("target_r")
        x = g["target_r"].to_numpy()
        for j in range(3):
            ax = axes[i, j]
            ax.axvspan(q1, q3, color=pf.GRID, alpha=0.6, lw=0)
            ax.axvline(med, color=pf.MUTED, lw=0.7)
            for key, lab, col, mk, ls in METHODS:
                ax.plot(
                    x,
                    g[f"{key}_comp_recovery_{j + 1}"],
                    color=col,
                    marker=mk,
                    ms=4,
                    ls=ls,
                    lw=1.4,
                    label=lab,
                )
            ax.set_ylim(0, 1.02)
            ax.set_xticks(x)
            ax.grid(axis="y", visible=True)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f"植入名次 {j + 1}")
            if j == 0:
                ax.set_ylabel(f"权重 ({w.replace(',', ', ')})\n化合物层找回率")
            if i == 1:
                ax.set_xlabel("第 1 个植入化合物的 $r$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=pf.GRID, alpha=0.6))
    labels.append(f"观测 Top-1 $|r|$ 四分位 {q1:.2f}–{q3:.2f}，竖线为中位数 {med:.2f}")
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.text(
        0.5,
        -0.09,
        f"每格每点 {n_sim:,} 次模拟，二项 95% 区间约 ±0.03，未画误差线",
        ha="center",
        fontsize=7,
        color=pf.MUTED,
    )
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    print(pf.save(fig, "q2", "fig_q2_recovery.pdf"))


if __name__ == "__main__":
    main()
