import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q2 import config as C  # noqa: E402

ORIGIN_LABELS = {
    "Yeast": "酵母",
    "Hops": "酒花",
    "Malt": "麦芽",
    "Spices": "香料",
    "Others": "其他",
    "Wild flora": "野生菌群",
    "Internal standard solution": "内标",
    "NA": "未标注",
}
ORIGIN_COLORS = dict(
    zip(
        ORIGIN_LABELS,
        ("#0072B2", "#009E73", "#E69F00", "#CC79A7", "#7d6aa8", "#56B4E9", "#222222", "#bbbbbb"),
    )
)


def main():
    plotstyle.apply()
    lay = plotstyle.load_npz(C.Q2_DIR / "q2_layout.npz")
    nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    edges = pd.read_csv(C.Q2_DIR / "graph_edges.csv")
    mods = pd.read_csv(C.Q2_DIR / "graph_modules.csv")
    pos = dict(zip(lay["node"], lay["xy"]))
    tau = float(lay["tau"])

    fig, ax = plt.subplots(figsize=(11, 9.5))
    for e in edges.itertuples():
        (x1, y1), (x2, y2) = pos[e.source], pos[e.target]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#888888",
            alpha=0.45,
            zorder=1,
            lw=0.3 + 2.0 * (e.weight_abs - tau),
            ls="-" if e.rho >= 0 else "--",
        )
    cand = nodes[nodes["is_candidate"]]
    for _, g in cand.groupby("module_id"):
        pts = np.array([pos[n] for n in g["node"]])
        if len(g) >= 3:
            poly = pts[ConvexHull(pts).vertices]
            ax.fill(poly[:, 0], poly[:, 1], color="#bbbbbb", alpha=0.22, zorder=0)
            ax.plot(*np.vstack([poly, poly[:1]]).T, color="#999999", lw=0.6, zorder=0)
        elif len(g) == 2:
            ax.plot(pts[:, 0], pts[:, 1], color="#bbbbbb", lw=5, alpha=0.5, zorder=0)
    origin = nodes["origin_paper"].fillna("NA")
    size = 12 + 18 * nodes["k_in"].fillna(0).to_numpy()
    for o, lab in ORIGIN_LABELS.items():
        m = (origin == o).to_numpy()
        ann = nodes["is_annotation"].to_numpy()
        pts = np.array([pos[n] for n in nodes["node"][m & ~ann]]).reshape(-1, 2)
        if len(pts):
            ax.scatter(
                *pts.T,
                s=size[m & ~ann],
                color=ORIGIN_COLORS[o],
                label=lab,
                zorder=2,
                edgecolor="white",
                linewidth=0.4,
            )
        pts = np.array([pos[n] for n in nodes["node"][m & ann]]).reshape(-1, 2)
        if len(pts):
            ax.scatter(
                *pts.T, s=40, facecolor="white", edgecolor=ORIGIN_COLORS[o], marker="s", zorder=3
            )
    for h in mods.loc[mods["size"] >= 3, "hub"]:
        ax.text(*pos[h], h, fontsize=6.5, ha="left", va="bottom", zorder=4)
    ax.scatter(
        [],
        [],
        s=40,
        facecolor="white",
        edgecolor="#555555",
        marker="s",
        label="注释节点（综合指标）",
    )
    for k in (0, 2, 5):
        ax.scatter([], [], s=12 + 18 * k, color="#aaaaaa", label=f"加权模块内连接度 {k}")
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, ncol=2)
    n_multi = int((mods["size"] > 1).sum())
    ax.set_title(
        f"化合物共表达图（τ={tau:g}：{len(edges)} 条 |ρ|≥{tau:g} 的边，虚线为负相关；"
        f"{len(mods)} 个模块，其中多成员 {n_multi} 个以灰色圈示；MDS 距离 1−|ρ|）",
        fontsize=8.5,
    )
    ax.set_axis_off()
    fig.tight_layout()
    print(plotstyle.save(fig, C.FIGURES, "fig_q2_network.pdf"))


if __name__ == "__main__":
    main()
