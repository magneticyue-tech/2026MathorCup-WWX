"""各模型折外连续预测与 x̄ 的平均 Spearman 及按日期组 bootstrap 区间 -> q1_spearman.npz。"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.meta import run_meta  # noqa: E402
from src.q1.metrics import N_BOOT, bootstrap_weights, spearman_by_dim  # noqa: E402
from src.q1.paths import PROCESSED_DIR, SEED, ensure_dirs  # noqa: E402

MODELS = ("main", "b0", "b1", "lin10", "enet", "rf", "chemstyle", "c2")


def mean_spearman(mu, xbar, idx):
    ry = rankdata(xbar[idx], axis=0)
    ry = ry - ry.mean(0)
    vals = []
    for r in range(mu.shape[0]):
        rx = rankdata(mu[r][idx], axis=0)
        rx = rx - rx.mean(0)
        den = np.sqrt((rx**2).sum(0) * (ry**2).sum(0))
        with np.errstate(invalid="ignore", divide="ignore"):
            vals.append(np.where(den > 0, (rx * ry).sum(0) / den, np.nan))
    return float(np.nanmean(vals))


def main():
    ensure_dirs()
    d = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=False)
    xbar, groups = d["xbar"], d["date_group"]
    bl = np.load(PROCESSED_DIR / "q1_baselines.npz", allow_pickle=False)
    mu = {
        "main": np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=False)["oof_mu"],
        "c2": np.load(PROCESSED_DIR / "q1_twostage.npz", allow_pickle=False)["oof_c2"],
    }
    for m in MODELS:
        if m not in mu:
            mu[m] = bl[f"oof_{m}"]

    n = len(xbar)
    W = bootstrap_weights(groups, N_BOOT, SEED)
    boots = [np.repeat(np.arange(n), w.astype(int)) for w in W]
    point = np.array([mean_spearman(mu[m], xbar, np.arange(n)) for m in MODELS])
    boot = np.array([[mean_spearman(mu[m], xbar, idx) for m in MODELS] for idx in boots])
    diff = boot - boot[:, [0]]
    per_dim = np.stack([np.nanmean(spearman_by_dim(mu[m], xbar), axis=0) for m in MODELS])

    out = PROCESSED_DIR / "q1_spearman.npz"
    np.savez_compressed(
        out,
        models=np.array(MODELS),
        dims=d["dims"],
        point=point,
        lo=np.percentile(boot, 2.5, axis=0),
        hi=np.percentile(boot, 97.5, axis=0),
        diff_point=point - point[0],
        diff_lo=np.percentile(diff, 2.5, axis=0),
        diff_hi=np.percentile(diff, 97.5, axis=0),
        per_dim=per_dim,
        n_boot=N_BOOT,
        seed=SEED,
        meta=run_meta(),
    )
    for m, p in zip(MODELS, point):
        print(f"{m}: 平均 Spearman {p:.4f}")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
