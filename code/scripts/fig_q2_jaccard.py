import sys
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q2 import config as C  # noqa: E402

METHODS = (
    ("M0", "不加图约束的边际 Spearman 选择"),
    ("M1", "图约束边际选择"),
    ("M2", "弹性网络"),
    ("M3", "梯度提升树置换重要度"),
    ("Within", "风格内选择"),
)
TOP_K = 3


def random_jaccard(n, k=TOP_K):
    total = comb(n, k)
    return sum(comb(k, j) * comb(n - k, k - j) / total * j / (2 * k - j) for j in range(k + 1))


def main():
    plotstyle.apply()
    full = pd.read_csv(C.Q2_DIR / "full_controls_results.csv")
    nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    base = random_jaccard(int(nodes["is_candidate"].sum()))
    p = plotstyle.PALETTE

    y = np.arange(len(METHODS))
    specs = (
        ("jaccard_{}", "化合物层 Jaccard", p["main"], -0.27),
        ("jaccard_mod_{}", "模块层 Jaccard", "#56B4E9", 0.0),
        ("{}_top1_match", "Top-1 相同比例", p["warn"], 0.27),
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for col, label, color, off in specs:
        vals = [full[col.format(k)].astype(float).mean() for k, _ in METHODS]
        ax.barh(y + off, vals, height=0.26, color=color, label=label)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi + off, f"{v:.3f}", va="center", fontsize=9)
    rng = np.random.default_rng(C.SEED)
    for yi, (k, _) in zip(y, METHODS):
        v = full[f"jaccard_{k}"].astype(float).to_numpy()
        ax.scatter(
            v,
            yi - 0.27 + rng.uniform(-0.09, 0.09, len(v)),
            s=5,
            color="#222222",
            alpha=0.35,
            zorder=3,
        )
    ax.axvline(base, ls="--", color="#444444", lw=0.9)
    ax.text(base + 0.01, len(METHODS) - 0.45, f"随机三元集合期望重合 {base:.3f}", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in METHODS], fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(
        f"与图约束前向选择 Top-3 的一致性"
        f"（{len(full)} 个感官维度平均；黑点为逐维化合物层 Jaccard）",
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    print(plotstyle.save(fig, C.FIGURES, "fig_q2_jaccard.pdf"))


if __name__ == "__main__":
    main()
