"""化合物共表达图的 MDS 布局（距离 1−|ρ|，附件 2）-> q2_layout.npz。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.meta import run_meta  # noqa: E402
from src.q2 import config as C  # noqa: E402

N_INIT = 4


def main():
    nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    corr = pd.read_csv(C.CORR_PATH, index_col=0)
    chem_cols = [
        c
        for c in pd.read_csv(C.CHEM_PATH, nrows=1).columns
        if c not in ("beer", "beer_id", "tasting_category_fine")
    ]
    corr = pd.DataFrame(corr.to_numpy(float), index=chem_cols, columns=chem_cols)
    corr = corr.loc[nodes["node"], nodes["node"]]
    D = 1.0 - corr.abs().to_numpy()
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=C.SEED,
        n_init=N_INIT,
        normalized_stress="auto",
    )
    xy = mds.fit_transform(D)
    out = C.Q2_DIR / "q2_layout.npz"
    np.savez_compressed(
        out,
        node=nodes["node"].to_numpy(str),
        xy=xy,
        stress=mds.stress_,
        tau=C.TAU_MAIN,
        seed=C.SEED,
        meta=run_meta(),
    )
    print(f"MDS 布局 {len(xy)} 个节点，stress {mds.stress_:.4f}；写出 {out}")


if __name__ == "__main__":
    main()
