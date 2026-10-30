import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, load_json, save  # noqa: E402

METHODS = [
    ("p_src", "源啤酒"),
    ("p_t0", "平移不加约束"),
    ("p_t1", "平移后截断缺陷列"),
    ("p_t4", "原谱不变"),
    ("p_t5", "截断加感官可行搜索"),
    ("p_main", "约束投影与感官可行搜索"),
    ("p_t2", "真实目标风格啤酒检索"),
]


def main():
    apply()
    tab = pd.read_csv(PROCESSED_DIR / "q3_transfer_pils_tripel.csv", encoding="utf-8-sig")
    js = load_json("q3_transfer.json")["pils_tripel"]["summary_median"]
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for k, (col, lab) in enumerate(METHODS):
        v = tab[col].to_numpy(float)
        ok = np.isfinite(v)
        jit = (np.arange(len(v)) - len(v) / 2) * 0.02
        ax.plot(
            np.full(ok.sum(), k) + jit[ok],
            v[ok],
            "o",
            ms=4,
            alpha=0.8,
            color=PALETTE["main"] if col == "p_main" else PALETTE["ctrl"],
        )
        if ok.any():
            ax.plot([k - 0.3, k + 0.3], [np.median(v[ok])] * 2, color="k", lw=1.5)
        else:
            ax.text(k, 0.02, "无满足两层\n约束的候选", ha="center", fontsize=9)
    ref = tab["ref_q25_tripel"].iloc[0]
    ax.axhspan(
        ref,
        js["p_real_tripel_median"],
        color=PALETTE["accent"],
        alpha=0.15,
        label="真实 Tripel 折外判别器概率：25% 分位至中位数",
    )
    ax.set_xticks(range(len(METHODS)), [m[1] for m in METHODS], rotation=25, ha="right")
    ax.set_ylabel("交叉拟合判别器给出的 Tripel 概率")
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], marker="o", color=PALETTE["main"], linestyle="none"),
        Line2D([], [], marker="o", color=PALETTE["ctrl"], linestyle="none"),
        Line2D([], [], color="k", lw=1.5),
    ]
    labels += ["约束投影与感官可行搜索", "对照方法", "中位数"]
    ax.legend(
        handles,
        labels,
        fontsize=9,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    save(fig, FIG_DIR, "fig_q3_transfer_pt.pdf")


if __name__ == "__main__":
    main()
