"""对照 C1 的品评团与零点诊断：精确并列连边、品评团、组维零点与地板堆积 -> q1_panels.npz。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.data import DIMS, load_q1  # noqa: E402
from src.q1.meta import run_meta  # noqa: E402
from src.q1.panels import infer_panels, tie_pair_counts  # noqa: E402
from src.q1.paths import PROCESSED_DIR, SEED, ensure_dirs  # noqa: E402
from src.q1.twostage import group_zeros  # noqa: E402

K_TIES = 3


def main():
    ensure_dirs()
    data = load_q1()
    Z = data.Z.to_numpy(float)
    n, D = Z.shape
    cnt = tie_pair_counts(data.Z)
    edges = np.array(sorted((i, j, c) for (i, j), c in cnt.items() if c >= K_TIES), dtype=int)
    panels = infer_panels(data.Z, k=K_TIES)
    zeros = group_zeros(Z, panels, np.arange(n), DIMS)
    at_zero = ~np.isnan(zeros) & np.isclose(np.round(Z, 9), zeros)
    date_group = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=False)["date_group"]

    out = PROCESSED_DIR / "q1_panels.npz"
    np.savez_compressed(
        out,
        dims=np.array(DIMS),
        z=Z,
        edges=edges,
        panels=panels,
        zeros=zeros,
        at_zero=at_zero,
        coverage=(~np.isnan(zeros)).mean(axis=0),
        date_group=date_group,
        k_ties=K_TIES,
        seed=SEED,
        meta=run_meta(),
    )
    sizes = np.bincount(panels)
    print(
        f"并列连边 {len(edges)} 条；多啤酒品评团 {(sizes > 1).sum()} 个，覆盖 "
        f"{sizes[sizes > 1].sum()} 款；有零点的单元占比 {np.mean(~np.isnan(zeros)):.3f}"
    )
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
