# code/src/q2/extra_figures.py
"""Q2 图：共表达网络（§3.4）、方法对比、π^c 分布、模块大小。只读 code/data/processed/q2 的结果。"""
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402
from sklearn.manifold import MDS  # noqa: E402

import config as C  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ORIGIN_COLORS = {
    "Yeast": "#2a78b5", "Hops": "#3a9a4a", "Malt": "#c98a1b", "Spices": "#b5465a",
    "Others": "#7d6aa8", "Wild flora": "#5aa5a0", "Internal standard solution": "#222222",
}


def plot_network_graph():
    nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    edges = pd.read_csv(C.Q2_DIR / "graph_edges.csv")
    corr = pd.read_csv(C.CORR_PATH, index_col=0)
    chem_cols = [c for c in pd.read_csv(C.CHEM_PATH, nrows=1).columns if c not in ("beer", "beer_id", "tasting_category_fine")]
    corr = pd.DataFrame(corr.values, index=chem_cols, columns=chem_cols).loc[nodes["node"], nodes["node"]]
    D = 1.0 - corr.abs().to_numpy()
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    xy = MDS(n_components=2, dissimilarity="precomputed", random_state=C.SEED, n_init=4,
             normalized_stress="auto").fit_transform(D)
    pos = dict(zip(nodes["node"], xy))

    fig, ax = plt.subplots(figsize=(12, 10))
    for e in edges.itertuples():
        (x1, y1), (x2, y2) = pos[e.source], pos[e.target]
        ax.plot([x1, x2], [y1, y2], color="#999999", lw=0.4 + 1.2 * (e.weight_abs - 0.6), alpha=0.5,
                ls="-" if e.rho >= 0 else "--", zorder=1)
    for m, g in nodes[nodes["is_candidate"]].groupby("module_id"):
        if len(g) >= 3:
            pts = np.array([pos[n] for n in g["node"]])
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1], color="#cccccc", alpha=0.25, zorder=0)
        elif len(g) == 2:
            pts = np.array([pos[n] for n in g["node"]])
            ax.plot(pts[:, 0], pts[:, 1], color="#bbbbbb", lw=4, alpha=0.4, zorder=0)
    for origin, g in nodes.groupby(nodes["origin_paper"].fillna("NA")):
        pts = np.array([pos[n] for n in g["node"]])
        ann = g["is_annotation"].to_numpy()
        col = ORIGIN_COLORS.get(origin, "#aaaaaa")
        ax.scatter(pts[~ann, 0], pts[~ann, 1], s=18, color=col, label=origin, zorder=2)
        if ann.any():
            ax.scatter(pts[ann, 0], pts[ann, 1], s=40, facecolor="white", edgecolor=col, marker="s", zorder=3)
    mods = pd.read_csv(C.Q2_DIR / "graph_modules.csv")
    for h in mods.loc[mods["size"] >= 3, "hub"]:
        x, y = pos[h]
        ax.text(x, y, h, fontsize=7, ha="left", va="bottom", zorder=4)
    ax.scatter([], [], s=40, facecolor="white", edgecolor="#555555", marker="s", label="注释节点")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("化合物共表达图（|ρ|≥0.6 的边；灰色区域为多成员模块；MDS 基于 1−|ρ|）")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(C.FIGURES / "network_graph.png", dpi=300)
    plt.close(fig)


def plot_method_jaccard():
    full = pd.read_csv(C.Q2_DIR / "full_controls_results.csv")
    labels = ["M0 无图约束边际", "M1 图约束边际", "M2 ElasticNet", "M3 GBM", "风格内"]
    keys = ["M0", "M1", "M2", "M3", "Within"]
    comp = [full[f"jaccard_{k}"].mean() for k in keys]
    mod = [full[f"jaccard_mod_{k}"].mean() for k in keys]
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y - 0.18, comp, height=0.36, color="#2a78b5", label="化合物层")
    ax.barh(y + 0.18, mod, height=0.36, color="#9cc3e4", label="模块层")
    for i, (a, b) in enumerate(zip(comp, mod)):
        ax.text(a + 0.01, i - 0.18, f"{a:.3f}", va="center", fontsize=8)
        ax.text(b + 0.01, i + 0.18, f"{b:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("与主方法 Top-3 的 Jaccard（49 维平均）")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "method_comparison.png", dpi=300)
    plt.close(fig)


def plot_stability_hist():
    df = pd.read_csv(C.Q2_DIR / "final_top3_with_stability.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    for k, col in zip((1, 2, 3), ("#2a78b5", "#3a9a4a", "#c98a1b")):
        ax.hist(df.loc[df["rank"] == k, "pi_c"], bins=bins, alpha=0.6, color=col, label=f"第 {k} 位")
    ax.axvline(C.STABLE_THRESHOLD, color="k", ls="--", lw=1)
    ax.axvline(C.MEDIUM_THRESHOLD, color="k", ls=":", lw=1)
    ax.set_xlabel("化合物层入选频率 π^c（日期组簇 bootstrap，B=1000）")
    ax.set_ylabel("项数")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "stability_pi_c_hist.png", dpi=300)
    plt.close(fig)


def plot_module_sizes():
    sizes = pd.read_csv(C.Q2_DIR / "graph_modules.csv")["size"]
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = sizes.value_counts().sort_index()
    ax.bar(vc.index, vc.values, color="#2a78b5")
    ax.set_yscale("log")
    ax.set_xlabel("模块成员数（τ=0.6）")
    ax.set_ylabel("模块数")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "module_sizes.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_network_graph()
    plot_method_jaccard()
    plot_stability_hist()
    plot_module_sizes()
    print("图已更新：network_graph、method_comparison、stability_pi_c_hist、module_sizes")
