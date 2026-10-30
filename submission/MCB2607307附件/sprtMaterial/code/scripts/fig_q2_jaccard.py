import sys
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q2 import config as C  # noqa: E402

from src import paperfig as pf  # noqa: E402

METHODS = (
    ("M0", "无图约束\n边际选择"),
    ("M1", "图约束\n边际选择"),
    ("M2", "ElasticNet"),
    ("M3", "梯度提升树"),
    ("Within", "风格内分析"),
)
TOP_K = 3


def random_jaccard(n, k=TOP_K):
    total = comb(n, k)
    return sum(comb(k, j) * comb(n - k, k - j) / total * j / (2 * k - j) for j in range(k + 1))


def main():
    pf.apply()
    full = pd.read_csv(C.Q2_DIR / "full_controls_results.csv")
    n_cand = int(pd.read_csv(C.Q2_DIR / "graph_nodes.csv")["is_candidate"].sum())
    base = random_jaccard(n_cand)
    comp = [full[f"jaccard_{k}"].astype(float).mean() for k, _ in METHODS]
    mod = [full[f"jaccard_mod_{k}"].astype(float).mean() for k, _ in METHODS]
    top1 = [full[f"{k}_top1_match"].astype(float).mean() for k, _ in METHODS]

    fig, ax = plt.subplots(figsize=(pf.TEXTWIDTH, 3.0))
    x = np.arange(len(METHODS))
    w = 0.36
    ax.bar(x - w / 2, comp, w - 0.04, color=pf.BLUE, label="化合物层")
    ax.bar(x + w / 2, mod, w - 0.04, color=pf.SKY, label="模块层")
    for xi, a, b, t in zip(x, comp, mod, top1):
        ax.text(xi - w / 2, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(xi + w / 2, b + 0.01, f"{b:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(
            xi,
            max(a, b) + 0.09,
            f"Top-1 一致 {t:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=pf.ORANGE,
        )
    ax.axhline(base, color=pf.INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(
        len(METHODS) - 0.45,
        base + 0.012,
        f"随机 3 元集合基线 {base:.3f}",
        ha="right",
        va="bottom",
        fontsize=7,
        color=pf.MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )
    ax.set_xticks(x, [lab for _, lab in METHODS])
    ax.set_ylabel(f"与主方法的 Jaccard，{len(full)} 维平均")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", visible=True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, handlelength=1.0)
    fig.tight_layout()
    print(pf.save(fig, "q2", "fig_q2_jaccard.pdf"))


if __name__ == "__main__":
    main()
