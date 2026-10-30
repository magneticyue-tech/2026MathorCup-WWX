import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q2 import config as C  # noqa: E402

RANK_MARK = {1: "①", 2: "②", 3: "③"}


def main():
    plotstyle.apply()
    final = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    m = final[(final["method"] == "main") & final["threshold_mean"].notna()]
    n_all = final[final["method"] == "main"]["compound"].nunique()
    g = (
        m.groupby("compound")
        .agg(
            share=("oav_ge_1_ratio", "first"),
            n_beers=("oav_n_beers", "first"),
            name=("chemical_name", "first"),
            dims=("dimension", lambda s: sorted(set(s))),
            ranks=("rank", lambda s: "".join(RANK_MARK[r] for r in sorted(set(s)))),
        )
        .sort_values("share")
    )
    p = plotstyle.PALETTE
    colors = [
        p["better"] if v >= 0.5 else (p["warn"] if v > 0 else p["neutral"]) for v in g["share"]
    ]

    fig, ax = plt.subplots(figsize=(8.4, 0.3 * len(g) + 1.2))
    y = np.arange(len(g))
    ax.barh(y, g["share"], color=colors)
    for yi, (comp, r) in zip(y, g.iterrows()):
        ax.text(
            r["share"] + 0.01,
            yi,
            f"{r['share']:.2f}（{int(r['n_beers'])} 款；{len(r['dims'])} 维 {r['ranks']}）",
            va="center",
            fontsize=6.5,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{c}" if pd.isna(r["name"]) else f"{r['name']}" for c, r in g.iterrows()], fontsize=7
    )
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("OAV ≥ 1 的啤酒比例（浓度 / 附件 8 感官阈值均值）")
    ax.set_title(
        f"Top-3 化合物中有阈值者的 OAV≥1 啤酒比例（{len(g)}/{n_all} 个入选化合物有阈值；"
        "括号内为有浓度的款数、进入的维度数与位置）",
        fontsize=8,
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    print(plotstyle.save(fig, C.FIGURES, "fig_q2_oav.pdf"))


if __name__ == "__main__":
    main()
