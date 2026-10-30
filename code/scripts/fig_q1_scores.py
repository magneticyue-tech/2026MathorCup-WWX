import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402


def leaf_order(X):
    if len(X) < 3:
        return np.arange(len(X))
    return leaves_list(linkage(X, method="average", metric="euclidean"))


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    d = plotstyle.load_npz(args.src / "q1_data.npz")
    s = plotstyle.load_npz(args.src / "q1_delivery_scores.npz")
    dims = [str(x) for x in d["dims"]]
    style = np.array([str(x) for x in d["style"]])
    G = s["G_hat"]

    col = [dims.index(c) for members in DIM_GROUPS.values() for c in members]
    styles = np.unique(style)
    means = np.array([G[style == st].mean(0) for st in styles])
    rows, bounds = [], []
    for st in styles[leaf_order(means)]:
        idx = np.where(style == st)[0]
        rows += list(idx[leaf_order(G[idx][:, col].astype(float))])
        bounds.append((st, len(rows)))

    cmap = ListedColormap(plt.get_cmap("YlOrRd")(np.linspace(0.05, 0.95, 7)))
    norm = BoundaryNorm(np.arange(-0.5, 7.5), 7)
    fig, ax = plt.subplots(figsize=(13, 13))
    im = ax.imshow(G[np.ix_(rows, col)], aspect="auto", cmap=cmap, norm=norm, interpolation="none")
    prev = 0
    for st, end in bounds:
        ax.axhline(end - 0.5, color="white", lw=0.8)
        ax.text(
            -1,
            (prev + end - 1) / 2,
            f"{st}（n={end - prev}）",
            ha="right",
            va="center",
            fontsize=9,
        )
        prev = end
    pos = 0
    for g, members in DIM_GROUPS.items():
        ax.axvline(pos + len(members) - 0.5, color="black", lw=0.8)
        ax.text(pos + (len(members) - 1) / 2, -2, g, ha="center", va="bottom", fontsize=9)
        pos += len(members)
    ax.set_xticks(range(len(col)))
    ax.set_xticklabels([dims[c] for c in col], rotation=90, fontsize=9)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("感官维度")
    ax.set_ylabel("啤酒（按风格分块）", fontsize=9)
    ax.yaxis.set_label_coords(-0.28, 0.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.03, ticks=range(7))
    cb.set_label("0–6 等级", fontsize=9)
    cb.ax.tick_params(labelsize=9)
    print(plotstyle.save(fig, args.out, "fig_q1_scores.pdf"))


if __name__ == "__main__":
    main()
