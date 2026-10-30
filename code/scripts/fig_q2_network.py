import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q2 import config as C  # noqa: E402

from src import paperfig as pf  # noqa: E402

ORIGINS = {
    "Yeast": ("酵母", pf.BLUE),
    "Hops": ("酒花", pf.GREEN),
    "Malt": ("麦芽", pf.YELLOW),
    "Spices": ("香料", pf.PINK),
    "Wild flora": ("野生菌群", pf.SKY),
    "Others": ("其他", "#8c6bb1"),
    "Internal standard solution": ("内标", pf.INK),
    "NA": ("无来源标签", "#8a8a8a"),
}
NAMED_MODULES = {"ethanol..v.v.": "发酵强度模块", "ethyl_octanoate": "中链脂肪酸乙酯模块"}


def clean(name):
    try:
        return name.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def size_of(k):
    return 12 + 14 * k


def main():
    pf.apply()
    lay = np.load(C.Q2_DIR / "q2_netlayout.npz", allow_pickle=False)
    nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv").set_index("node")
    edges = pd.read_csv(C.Q2_DIR / "graph_edges.csv")
    mods = pd.read_csv(C.Q2_DIR / "graph_modules.csv")
    final = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    n_dims = final[final["method"] == "main"].groupby("compound")["dimension"].nunique()
    names = [str(n) for n in lay["node"]]
    pos = dict(zip(names, lay["xy"]))
    block = dict(zip(names, lay["module"]))
    tau = float(lay["tau"])

    fig, ax = plt.subplots(figsize=(pf.TEXTWIDTH, 8.0))
    for b in sorted(set(lay["module"])):
        if b == -1:
            continue
        pts = np.array([pos[n] for n in names if block[n] == b])
        lo, hi = pts.min(0) - 0.17, pts.max(0) + 0.17
        ax.add_patch(
            FancyBboxPatch(
                lo,
                *(hi - lo),
                boxstyle="round,pad=0,rounding_size=0.08",
                fc="#f3f3f3",
                ec="none",
                zorder=0,
            )
        )
    drawn = {n for n in names if block[n] != -1}
    sub = edges[edges["source"].isin(drawn) & edges["target"].isin(drawn)]
    for e in sub.itertuples():
        (x1, y1), (x2, y2) = pos[e.source], pos[e.target]
        inside = block[e.source] == block[e.target]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#7f7f7f" if inside else "#d0d0d0",
            lw=(0.4 + 2.2 * (e.weight_abs - tau)) if inside else 0.35,
            ls="-" if e.rho >= 0 else (0, (2, 1.5)),
            alpha=0.9 if inside else 0.5,
            zorder=1 if inside else 0.5,
        )

    for n in names:
        row = nodes.loc[n]
        origin = row["origin_paper"] if isinstance(row["origin_paper"], str) else "NA"
        col = ORIGINS[origin][1]
        k = int(n_dims.get(n, 0))
        marker = "s" if row["is_annotation"] else "o"
        if k > 0:
            ax.scatter(
                *pos[n], s=size_of(k), color=col, marker=marker, ec="white", lw=0.5, zorder=3
            )
        else:
            ax.scatter(
                *pos[n],
                s=size_of(0) + 6,
                facecolor="white",
                ec=col,
                marker=marker,
                lw=0.9,
                zorder=3,
            )

    hubs = set(mods.loc[mods["size"] >= 3, "hub"])
    labelled = [n for n in names if n in hubs or n in n_dims.index or nodes.loc[n, "is_annotation"]]
    close = {
        n: [m for m in labelled if m != n and np.all(np.abs(pos[m] - pos[n]) < (0.6, 0.4))]
        for n in labelled
    }
    placed = {}
    for n in labelled:
        row = nodes.loc[n]
        name = clean(row["chemical_name"]) if isinstance(row["chemical_name"], str) else n
        if row["is_annotation"]:
            near = row["nearest_candidate"]
            name = (
                f"{name}\n最近成分 {clean(nodes.loc[near, 'chemical_name'])}"
                f"，$|\\rho|$={abs(row['nearest_candidate_rho']):.2f}"
            )
        ha, dx, va, dy = "center", 0, "bottom", 4.0
        if not row["is_annotation"]:
            mates = [m for m in names if block[m] == block[n]]
            cx = np.mean([pos[m][0] for m in mates])
            cy = np.mean([pos[m][1] for m in mates])
            if pos[n][0] < cx - 0.05:
                ha, dx = "right", 2
            elif pos[n][0] > cx + 0.05:
                ha, dx = "left", -2
            if pos[n][1] < cy - 0.05 or (len(mates) == 2 and pos[n][0] > cx):
                va, dy = "top", -4.0
            if close[n]:
                m0 = min(close[n], key=lambda m: np.linalg.norm(pos[m] - pos[n]))
                ha, dx = ("left", -2) if pos[n][0] > pos[m0][0] else ("right", 2)
            for m in close[n]:
                if placed.get(m) == va:
                    va, dy = ("bottom", 4.0) if va == "top" else ("top", -4.0)
            placed[n] = va
        ax.annotate(
            name,
            pos[n],
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=4.6,
            color=pf.INK,
            zorder=4,
            linespacing=1.1,
            fontweight="bold" if n in hubs else "normal",
        )
    for n, title in NAMED_MODULES.items():
        b = block[n]
        pts = np.array([pos[m] for m in names if block[m] == b])
        ax.text(
            pts[:, 0].mean(),
            pts[:, 1].min() - 0.36,
            title,
            ha="center",
            va="top",
            fontsize=6.2,
            color=pf.ORANGE,
            fontweight="bold",
            zorder=4,
        )
    apts = np.array([pos[n] for n in names if block[n] == -1])
    ax.text(
        apts[:, 0].min() - 1.0,
        apts[:, 1].mean(),
        "注释节点\n不作候选",
        ha="right",
        va="center",
        fontsize=6.2,
        color=pf.MUTED,
    )
    ax.plot(
        [apts[:, 0].min() - 1.6, apts[:, 0].max() + 1.6],
        [apts[:, 1].max() + 0.75] * 2,
        color=pf.GRID,
        lw=0.6,
    )

    handles = [
        Line2D([], [], marker="o", ls="", color=ORIGINS[o][1], ms=5, label=ORIGINS[o][0])
        for o in ("Yeast", "Hops", "Malt", "Spices", "Wild flora", "Others", "NA")
    ]
    handles += [
        Line2D([], [], marker="s", ls="", color=pf.INK, ms=4.5, label="内标"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=pf.MUTED, ms=5, label="未进入 Top-3"),
        Line2D(
            [], [], marker="o", ls="", color=pf.MUTED, ms=np.sqrt(size_of(1)), label="进入 1 个维度"
        ),
        Line2D(
            [], [], marker="o", ls="", color=pf.MUTED, ms=np.sqrt(size_of(4)), label="进入 4 个维度"
        ),
        Line2D([], [], color="#7f7f7f", lw=1.3, label="模块内边，宽度随 $|\\rho|$ 增大"),
        Line2D([], [], color="#7f7f7f", lw=1.0, ls=(0, (2, 1.5)), label="负相关边"),
        Line2D([], [], color="#d0d0d0", lw=1.0, label="模块间边"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        columnspacing=0.9,
        handletextpad=0.3,
        fontsize=6.5,
    )
    n_multi = len(set(lay["module"])) - 1
    ax.set_title(
        f"$|\\rho|\\geq{tau:g}$ 连边，{n_multi} 个多成员模块共 "
        f"{int((lay['module'] != -1).sum())} 个化合物，灰底为模块；"
        f"另有 {int(lay['n_single'])} 个单成员节点未画",
        fontsize=7.5,
        color=pf.MUTED,
        loc="left",
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    print(pf.save(fig, "q2", "fig_q2_network.pdf"))


if __name__ == "__main__":
    main()
