"""用问题 1 全数据模型对同格式新化学数据评分；给出观测评分时计算与正式评价同口径的指标。"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.data import DIMS, grade  # noqa: E402
from src.q1.discretize import (  # noqa: E402
    N_GRADES,
    apply_cuts,
    dist_interval,
    dist_median,
    predictive_dist,
)
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.metrics import (  # noqa: E402
    coverage,
    exact,
    grade_mae,
    mean_qwk,
    qwk_by_dim,
    r2_by_dim,
    spearman_by_dim,
    within1,
)
from src.q1.paths import PROCESSED_DIR  # noqa: E402

RATING_RENAME = {
    "A_ester_fruity": "A_esters_fruity",
    "F_ester_fruity": "F_esters_fruity",
    "beer_name": "row",
}


def load_models(model_dir: Path):
    enc_path = model_dir / "encoder.joblib"
    if not enc_path.exists():
        raise FileNotFoundError(f"{enc_path} 不存在；须先运行 solve_q1_main.py")
    encoder = joblib.load(enc_path)
    models = [joblib.load(model_dir / f"{d}.joblib") for d in DIMS]
    return encoder, models


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def observed_means(path: Path, n_rows: int) -> np.ndarray:
    """逐品评员评分按行号取均分；行号为化学表中的 1 起行序，列名同原始评分文件。"""
    df = read_csv(path).rename(columns=RATING_RENAME)
    missing = [c for c in ["row", *DIMS] if c not in df.columns]
    if missing:
        raise KeyError(f"评分文件缺少列: {missing}")
    xbar = df.groupby("row")[DIMS].mean().reindex(np.arange(1, n_rows + 1))
    return xbar.to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chem", type=Path, required=True, help="附件 1 格式的化学表")
    ap.add_argument("--ratings", type=Path, help="原始评分格式的观测评分（可选）")
    ap.add_argument("--models", type=Path, default=PROCESSED_DIR / "q1_models")
    ap.add_argument("--scores", type=Path, default=PROCESSED_DIR / "q1_scores.npz")
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR / "new_data")
    args = ap.parse_args()
    require_sklearn()
    args.out.mkdir(parents=True, exist_ok=True)

    encoder, models = load_models(args.models)
    feats = list(encoder.fill_.index)
    chem = read_csv(args.chem)
    absent = [c for c in feats if c not in chem.columns]
    if absent:
        raise KeyError(f"化学表缺少 {len(absent)} 个特征列，例如 {absent[:5]}")
    Xe = encoder.transform(chem[feats].astype(float)).to_numpy(np.float64)
    mu = np.column_stack([m.predict(Xe) for m in models])

    z = np.load(args.scores, allow_pickle=False)
    resid, qcuts = z["final_resid"], z["final_qcuts"]
    p = np.stack([predictive_dist(mu[:, j], resid[j]) for j in range(len(DIMS))], axis=1)
    G_hat = dist_median(p)
    lo, hi = dist_interval(p)
    G_hat_Q = np.column_stack([apply_cuts(mu[:, j], qcuts[j]) for j in range(len(DIMS))])

    n, D = mu.shape
    rows = np.arange(1, n + 1)
    beer = chem["beer"].to_numpy() if "beer" in chem.columns else rows
    long = pd.DataFrame(
        {
            "row": np.repeat(rows, D),
            "beer": np.repeat(beer, D),
            "dim": np.tile(DIMS, n),
            "mu_hat": mu.ravel(),
            "G_hat": G_hat.ravel(),
            "G_hat_Q": G_hat_Q.ravel(),
            "lo": lo.ravel(),
            "hi": hi.ravel(),
            **{f"p{k}": p[:, :, k].ravel() for k in range(N_GRADES)},
        }
    )
    wide = pd.DataFrame(G_hat, columns=DIMS)
    wide.insert(0, "beer", beer)
    wide.insert(0, "row", rows)

    report = {"n_beers": int(n), "n_dims": int(D), "meta": json.loads(run_meta())}
    if args.ratings is not None:
        xbar = observed_means(args.ratings, n)
        ok = ~np.isnan(xbar).any(axis=1)
        G = grade(np.nan_to_num(xbar))
        long["xbar_obs"] = np.where(ok[:, None], xbar, np.nan).ravel()
        long["G_obs"] = np.where(ok[:, None], G, -1).ravel()
        Gs, Ghs, mus, xs = G[ok], G_hat[ok][None], mu[ok][None], xbar[ok]
        cov, width = coverage(Gs, lo[ok], hi[ok])
        report["metrics"] = {
            "n_scored": int(ok.sum()),
            "grade_mae": grade_mae(Gs, Ghs),
            "exact": exact(Gs, Ghs),
            "within1": within1(Gs, Ghs),
            "qwk_mean": mean_qwk(qwk_by_dim(Gs, Ghs))[0],
            "grade_mae_quant": grade_mae(Gs, G_hat_Q[ok][None]),
            "cont_mae": float(np.abs(mus[0] - xs).mean()),
            "r2_mean": float(np.nanmean(r2_by_dim(mus, xs))),
            "spearman_mean": float(np.nanmean(spearman_by_dim(mus, xs))),
            "coverage90": cov,
            "interval_width": width,
        }

    long.to_csv(args.out / "q1_new_scores_long.csv", index=False)
    wide.to_csv(args.out / "q1_new_scores.csv", index=False)
    (args.out / "q1_new_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"评分 {n} 款 × {D} 维，写出 {args.out}")
    if "metrics" in report:
        print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
