import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q2 import config as C  # noqa: E402

TIERS = (("稳定", "#0072B2"), ("中等", "#E69F00"), ("不稳定", "#bbbbbb"))


def main():
    plotstyle.apply()
    final = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    m = final[final["method"] == "main"]
    dims = [d for members in DIM_GROUPS.values() for d in members if d in set(m["dimension"])]
    pi = m.pivot(index="dimension", columns="rank", values="pi_c").loc[dims]
    comp = m.pivot(index="dimension", columns="rank", values="compound").loc[dims]
    grade = m[m["rank"] == 1].set_index("dimension")["grade"].loc[dims]

    fig, axes = plt.subplots(1, 2, figsize=(11, 13), gridspec_kw={"width_ratios": [1.6, 1]})
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
                fontsize=4.6,
                color="white" if v > 0.6 else "#222222",
            )
    ax.set_xticks(range(3))
    ax.set_xticklabels(["第 1 位", "第 2 位", "第 3 位"])
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f"{d}（{grade[d]}）" for d in dims], fontsize=6)
    ax.grid(False)
    ax.set_title(
        f"(a) {len(dims)} 维 × 3 位置的入选频率 π^c"
        f"（{int(m['n_bootstrap'].iloc[0])} 次日期组 bootstrap；"
        "括号内为维度等级）",
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, shrink=0.3, pad=0.02, label="π^c")

    ax = axes[1]
    counts = m.groupby(["rank", "stability"]).size().unstack(fill_value=0)
    left = np.zeros(3)
    for tier, col in TIERS:
        v = counts.get(tier, pd.Series(0, index=counts.index)).reindex([1, 2, 3], fill_value=0)
        ax.barh([1, 2, 3], v, left=left, color=col, label=tier)
        for y, (a, b) in enumerate(zip(left, v), start=1):
            if b > 0:
                ax.text(
                    a + b / 2,
                    y,
                    str(int(b)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if tier != "不稳定" else "#222222",
                )
        left += v.to_numpy()
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["第 1 位", "第 2 位", "第 3 位"])
    ax.invert_yaxis()
    ax.set_xlabel("维度数")
    ax.set_title(
        f"(b) 入选项稳定性分档：π^c ≥ {C.STABLE_THRESHOLD:g} 稳定，"
        f"≥ {C.MEDIUM_THRESHOLD:.2f} 中等，其余不稳定",
        fontsize=8,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    ax.grid(axis="y", visible=False)
    ax.set_box_aspect(0.6)
    fig.tight_layout()
    print(plotstyle.save(fig, C.FIGURES, "fig_q2_stability.pdf"))


if __name__ == "__main__":
    main()
