import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q2 import config as C  # noqa: E402

TIERS = (("稳定", "#0072B2"), ("中等", "#E69F00"), ("不稳定", "#bbbbbb"))
TIER_LABELS = {
    "稳定": f"稳定（π^c ≥ {C.STABLE_THRESHOLD:g}）",
    "中等": f"中等（1/3 ≤ π^c < {C.STABLE_THRESHOLD:g}）",
    "不稳定": "不稳定（π^c < 1/3）",
}
GRADE_LABELS = (
    f"A 级：首位显著且 π^c ≥ {C.GRADE_A_PI:g}",
    "B 级：首位显著",
    "C 级：首位不显著",
)


def main():
    plotstyle.apply()
    final = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    m = final[final["method"] == "main"]
    dims = [d for members in DIM_GROUPS.values() for d in members if d in set(m["dimension"])]
    pi = m.pivot(index="dimension", columns="rank", values="pi_c").loc[dims]
    comp = m.pivot(index="dimension", columns="rank", values="compound").loc[dims]
    grade = m[m["rank"] == 1].set_index("dimension")["grade"].loc[dims]

    fig, axes = plt.subplots(1, 2, figsize=(15, 16), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    im = ax.imshow(pi.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for i in range(len(dims)):
        for j in range(3):
            v = pi.iat[i, j]
            ax.text(
                j,
                i,
                f"{comp.iat[i, j]}\n{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.6 else "#222222",
            )
    ax.set_xticks(range(3))
    ax.set_xticklabels(["第 1 位", "第 2 位", "第 3 位"], fontsize=9)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f"{d}（{grade[d]}）" for d in dims], fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(False)
    ax.set_title("(a)", loc="left", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.3, pad=0.02)
    cbar.set_label("化合物层入选频率 π^c", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    ax = axes[1]
    counts = m.groupby(["rank", "stability"]).size().unstack(fill_value=0)
    left = np.zeros(3)
    for tier, col in TIERS:
        v = counts.get(tier, pd.Series(0, index=counts.index)).reindex([1, 2, 3], fill_value=0)
        ax.barh([1, 2, 3], v, left=left, color=col, label=TIER_LABELS[tier])
        for y, (a, b) in enumerate(zip(left, v), start=1):
            if b > 0:
                ax.text(
                    a + b / 2,
                    y,
                    str(int(b)),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if tier != "不稳定" else "#222222",
                )
        left += v.to_numpy()
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["第 1 位", "第 2 位", "第 3 位"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("感官维度数", fontsize=9)
    ax.set_title("(b)", loc="left", fontsize=11)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
    )
    ax.grid(axis="y", visible=False)
    ax.set_box_aspect(0.6)
    ax.set_anchor("N")
    fig.legend(
        handles=[Line2D([], [], linestyle="none", label=s) for s in GRADE_LABELS],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=0,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    print(plotstyle.save(fig, C.FIGURES, "fig_q2_stability.pdf"))


if __name__ == "__main__":
    main()
