import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.discretize import N_GRADES  # noqa: E402
from src.q1.paths import PROCESSED_DIR  # noqa: E402

from src import paperfig as pf  # noqa: E402


def freq(g):
    g = np.asarray(g)
    reps = g.reshape(-1, *g.shape[-2:]) if g.ndim == 3 else g[None]
    return np.mean([np.bincount(r.ravel(), minlength=N_GRADES) / r.size for r in reps], axis=0)


def main():
    pf.apply()
    G = plotstyle.load_npz(PROCESSED_DIR / "q1_data.npz")["G"]
    s = plotstyle.load_npz(PROCESSED_DIR / "q1_scores.npz")
    gm, gq = s["grade_median"], s["grade_quant"]
    R = gm.shape[0]
    k = np.arange(N_GRADES)
    n_true = np.bincount(G.ravel(), minlength=N_GRADES)

    fig, axes = plt.subplots(1, 2, figsize=(pf.TEXTWIDTH, 3.2))
    ax = axes[0]
    series = (
        ("真值等级", freq(G), pf.INK),
        ("主输出：分布中位数", freq(gm), pf.BLUE),
        ("第二输出：分位数切点", freq(gq), pf.ORANGE),
    )
    w = 0.27
    for i, (lab, f, c) in enumerate(series):
        ax.bar(k + (i - 1) * w, f, width=w - 0.03, color=c, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xticks(k)
    ax.set_xlabel("等级")
    ax.set_ylabel("等级频率，对数刻度")
    ax.grid(axis="y", visible=True, which="major")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=1, handlelength=1.0)
    pf.panel_label(ax, "a  三种等级频率")

    ax = axes[1]
    hit = np.broadcast_to(G, gm.shape) == gm
    recall = np.array(
        [hit[np.broadcast_to(G, gm.shape) == j].mean() if n_true[j] else np.nan for j in k]
    )
    near = np.abs(np.broadcast_to(G, gm.shape) - gm) <= 1
    recall1 = np.array(
        [near[np.broadcast_to(G, gm.shape) == j].mean() if n_true[j] else np.nan for j in k]
    )
    ax.bar(k, recall1, width=0.62, color=pf.SKY, label="误差不超过一级")
    ax.bar(k, recall, width=0.62, color=pf.BLUE, label="完全一致，即召回率")
    for j in k:
        ax.text(j, recall1[j] + 0.03, f"{recall[j]:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(k, [f"{j}\n{n_true[j]}" for j in k])
    ax.set_xlabel("真实等级与单元数")
    ax.set_ylabel("主输出的比例")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", visible=True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=1, handlelength=1.0)
    pf.panel_label(ax, "b  主输出按真实等级分层的召回率")
    fig.text(
        0.5,
        -0.05,
        f"分母：a 栏为全部 {G.size:,} 个啤酒乘维度单元，预测等级取 {R} 次重复的平均；"
        "b 栏为各真实等级的单元数，柱顶数字为召回率",
        ha="center",
        fontsize=7,
        color=pf.MUTED,
    )
    fig.tight_layout(w_pad=2.0)
    print(pf.save(fig, "q1", "fig_q1_grade.pdf"))


if __name__ == "__main__":
    main()
