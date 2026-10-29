import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, load_json, save  # noqa: E402

METHODS = [
    ("p_src", "源啤酒"),
    ("p_t0", "T0 平移"),
    ("p_t1", "T1 截断"),
    ("p_t4", "T4 原谱"),
    ("p_t5", "T5 截断+感官"),
    ("p_main", "主方法 α*"),
    ("p_t2", "T2 检索"),
]


def main():
    apply()
    tab = pd.read_csv(PROCESSED_DIR / "q3_transfer_pils_tripel.csv", encoding="utf-8-sig")
    js = load_json("q3_transfer.json")["pils_tripel"]["summary_median"]
    fig, ax = plt.subplots(figsize=(8, 4))
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
            ax.text(k, 0.02, "无满足两层\n约束的候选", ha="center", fontsize=7)
    ref = tab["ref_q25_tripel"].iloc[0]
    ax.axhspan(
        ref,
        js["p_real_tripel_median"],
        color=PALETTE["accent"],
        alpha=0.15,
        label="真实 Tripel 折外 P：25% 分位至中位数",
    )
    ax.set_xticks(range(len(METHODS)), [m[1] for m in METHODS], rotation=20)
    ax.set_ylabel("交叉拟合判别器 P(Tripel)")
    ax.set_title("Pils/Lager → Tripel 11 款的目标风格概率（横线为中位数）")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, FIG_DIR, "fig_q3_transfer_pt.pdf")


if __name__ == "__main__":
    main()
