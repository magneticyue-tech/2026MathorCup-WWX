"""尺度参数 m 的敏感性：C1 混合参照随 m 的还原误差，以及原始评分直接估计的 m -> q1_mscale.npz。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.data import DIMS, load_q1, load_raw_ratings  # noqa: E402
from src.q1.meta import run_meta  # noqa: E402
from src.q1.panels import infer_panels  # noqa: E402
from src.q1.paths import PROCESSED_DIR, SEED, ensure_dirs  # noqa: E402
from src.q1.twostage import (  # noqa: E402
    M_SCALE,
    group_zeros,
    linear_endpoints,
    local_scale,
    mixed_reference,
)

M_GRID = np.round(np.arange(2.0, 4.51, 0.25), 2)


def reference_by_m(Z, panels, fold_id, m):
    R_, n = fold_id.shape
    ref = np.full((R_, n, Z.shape[1]), -1, dtype=int)
    anchored = np.zeros_like(ref, dtype=bool)
    for r in range(R_):
        for k in np.unique(fold_id[r]):
            te = np.where(fold_id[r] == k)[0]
            tr = np.where(fold_id[r] != k)[0]
            lo, hi = linear_endpoints(Z, tr)
            sigma = local_scale(group_zeros(Z, panels, tr, DIMS), panels, tr, m)
            zb_te = group_zeros(Z, panels, te, DIMS)
            ref[r, te], anchored[r, te] = mixed_reference(Z[te], zb_te[te], sigma, lo, hi)
    return ref, anchored


def raw_m(ratings):
    x = ratings[DIMS].to_numpy(float)
    out = []
    for _, idx in ratings.groupby("taster").indices.items():
        v = x[idx]
        mean = v.mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(mean > 0, (v**2).mean(0) / mean, np.nan))
    return np.array(out)


def main():
    ensure_dirs()
    data = load_q1()
    Z = data.Z.to_numpy(float)
    panels = infer_panels(data.Z, k=3)
    G = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=False)["G"]
    fold_id = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=False)["fold_id"]
    ts = np.load(PROCESSED_DIR / "q1_twostage.npz", allow_pickle=False)

    stats = {k: np.full(len(M_GRID), np.nan) for k in ("mae_A", "exact_A", "bias_A", "mae_all")}
    share_changed = np.full(len(M_GRID), np.nan)
    ref_main, _ = reference_by_m(Z, panels, fold_id, M_SCALE)
    for i, m in enumerate(M_GRID):
        ref, anchored = reference_by_m(Z, panels, fold_id, m)
        e = ref - G[None]
        stats["mae_A"][i] = np.abs(e[anchored]).mean()
        stats["exact_A"][i] = (e[anchored] == 0).mean()
        stats["bias_A"][i] = e[anchored].mean()
        stats["mae_all"][i] = np.abs(e).mean()
        share_changed[i] = (ref[anchored] != ref_main[anchored]).mean()
    same = bool(np.array_equal(ref_main, ts["R_te"]))

    out = PROCESSED_DIR / "q1_mscale.npz"
    np.savez_compressed(
        out,
        m_grid=M_GRID,
        m_main=M_SCALE,
        share_changed=share_changed,
        m_raw=raw_m(load_raw_ratings()),
        dims=np.array(DIMS),
        matches_twostage=same,
        seed=SEED,
        meta=run_meta(),
        **stats,
    )
    print(f"m={M_SCALE} 的参照与 q1_twostage.npz 的 R_te 一致: {same}")
    for i, m in enumerate(M_GRID):
        print(f"m={m:.2f}：有锚 MAE {stats['mae_A'][i]:.3f}，偏差 {stats['bias_A'][i]:+.3f}")
    print(f"写出 {out}")
    if not same:
        sys.exit(1)


if __name__ == "__main__":
    main()
