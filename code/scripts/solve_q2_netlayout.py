"""论文网络图布局：多成员模块按大小分行排布、模块内力导向，注释节点单列一块 -> q2_netlayout.npz。"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.meta import run_meta  # noqa: E402
from src.q2 import config as C  # noqa: E402

ROW_WIDTH = 9.0
GAP = 0.45
ROW_GAP = 0.85
SPRING_ITER = 300


def block_layout(nodes, edges, seed):
    if len(nodes) == 2:
        return {nodes[0]: np.array([-0.4, 0.0]), nodes[1]: np.array([0.4, 0.0])}
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for e in edges.itertuples():
        if e.source in g and e.target in g:
            g.add_edge(e.source, e.target, weight=e.weight_abs)
    pos = nx.spring_layout(g, seed=seed, iterations=SPRING_ITER, weight="weight")
    xy = np.array([pos[n] for n in nodes])
    xy = xy - xy.mean(0)
    xy = xy / (np.abs(xy).max() or 1.0) * (0.32 * np.sqrt(len(nodes)))
    return dict(zip(nodes, xy))


def grid_layout(nodes, ncol, dx, dy):
    return {
        n: np.array([(i % ncol - (ncol - 1) / 2) * dx, -(i // ncol) * dy])
        for i, n in enumerate(nodes)
    }


def main():
    nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    edges = pd.read_csv(C.Q2_DIR / "graph_edges.csv")
    multi = nodes[nodes["is_candidate"] & (nodes["module_size"] > 1)]
    sizes = multi.groupby("module_id").size().sort_values(ascending=False, kind="stable")
    blocks = [
        (int(m), block_layout(list(multi.loc[multi["module_id"] == m, "node"]), edges, C.SEED + i))
        for i, m in enumerate(sizes.index)
    ]
    ann = list(nodes.loc[nodes["is_annotation"], "node"])
    blocks.append((-1, grid_layout(ann, 3, 2.9, 1.05)))

    names, xy, block_of = [], [], []
    x0, y0, row_h = 0.0, 0.0, 0.0
    for bid, local in blocks:
        pts = np.array(list(local.values()))
        lo, hi = pts.min(0), pts.max(0)
        w, h = hi[0] - lo[0] + GAP, hi[1] - lo[1] + ROW_GAP
        if bid == -1 or (x0 > 0 and x0 + w > ROW_WIDTH):
            x0, y0, row_h = 0.0, y0 - row_h, 0.0
        if bid == -1:
            x0 = (ROW_WIDTH - w) / 2
            y0 -= 0.35
        shift = np.array([x0 - lo[0] + GAP / 2, y0 - hi[1] - GAP / 2])
        for n, p in local.items():
            names.append(n)
            xy.append(p + shift)
            block_of.append(bid)
        x0 += w
        row_h = max(row_h, h)

    out = C.Q2_DIR / "q2_netlayout.npz"
    np.savez_compressed(
        out,
        node=np.array(names),
        xy=np.array(xy),
        module=np.array(block_of),
        n_single=int((nodes["is_candidate"] & (nodes["module_size"] == 1)).sum()),
        tau=C.TAU_MAIN,
        seed=C.SEED,
        meta=run_meta(),
    )
    print(f"多成员模块 {len(blocks) - 1} 个、节点 {len(names)} 个；写出 {out}")


if __name__ == "__main__":
    main()
