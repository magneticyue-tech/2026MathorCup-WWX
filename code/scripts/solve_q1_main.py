import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1_data import make_check  # noqa: E402
from src.q1.data import DIM_GROUPS, DIMS, load_q1  # noqa: E402
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.model import fit_outer_fold, merge_units, refit_full  # noqa: E402
from src.q1.paths import CHUNK_DIR, PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q1.splits import inner_splits, outer_splits, split_diagnostics  # noqa: E402


def save_unit(path, u):
    np.savez_compressed(
        path,
        train_idx=u["train_idx"],
        test_idx=u["test_idx"],
        oof=u["oof"],
        inner=u["inner"],
        chosen=json.dumps(u["chosen"]),
        groups=np.array(list(u["scores"])),
        **{f"scores__{g}": s for g, s in u["scores"].items()},
    )


def load_unit(path):
    z = np.load(path, allow_pickle=False)
    return {
        "train_idx": z["train_idx"],
        "test_idx": z["test_idx"],
        "oof": z["oof"],
        "inner": z["inner"],
        "chosen": json.loads(str(z["chosen"])),
        "scores": {g: z[f"scores__{g}"] for g in z["groups"]},
    }


def run_unit(path, X, Y, groups, style, tr, te, inner_seed):
    u = fit_outer_fold(X, Y, DIMS, DIM_GROUPS, groups, style, tr, te, inner_seed)
    save_unit(path, u)
    return path


def fold_metrics(oof_mu, xbar, repeats):
    mae, r2, rho = [], [], []
    for r in repeats:
        rows = ~np.isnan(oof_mu[r]).any(axis=1)
        p, y = oof_mu[r][rows], xbar[rows]
        mae.append(np.mean(np.abs(p - y)))
        ss_res = ((y - p) ** 2).sum(0)
        ss_tot = ((y - y.mean(0)) ** 2).sum(0)
        r2.append(1 - ss_res / ss_tot)
        rho.append([spearmanr(p[:, j], y[:, j])[0] for j in range(y.shape[1])])
    return float(np.mean(mae)), np.mean(r2, axis=0), np.mean(rho, axis=0)


def selection_summary(chosen):
    out = {}
    for g in DIM_GROUPS:
        picks = [json.dumps(v, sort_keys=True) for k, v in chosen.items() if k.endswith(f"_{g}")]
        out[g] = {
            "n_distinct": len(set(picks)),
            "counts": dict(Counter(picks).most_common()),
            "n_mono_true": sum(json.loads(p)["mono"] for p in picks),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers 必须在 1 到 8 之间")

    require_sklearn()
    ensure_dirs()
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    chunk_dir = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / "main"
    model_dir = out_dir / "q1_models"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=False)
    xbar, date_group, reliability = z["xbar"], z["date_group"], z["reliability"]
    data = load_q1()
    X, style = data.X, data.style
    splits = outer_splits(date_group, style)
    split_diag = split_diagnostics(splits, date_group, style)
    R, K = len(splits), len(splits[0])
    cells = [(0, 0), (0, 1)] if args.smoke else [(r, k) for r in range(R) for k in range(K)]

    t0 = time.time()
    todo = [c for c in cells if not (chunk_dir / f"r{c[0]}_k{c[1]}.npz").exists()]
    print(f"单元 {len(cells)} 个，待计算 {len(todo)} 个，进程 {args.workers}", flush=True)
    jobs = [delayed(refit_full)(X, xbar, DIMS, DIM_GROUPS, date_group, style)]
    jobs += [
        delayed(run_unit)(
            chunk_dir / f"r{r}_k{k}.npz", X, xbar, date_group, style, *splits[r][k], SEED + 100 + r
        )
        for r, k in todo
    ]
    results = Parallel(n_jobs=args.workers, verbose=10)(jobs)
    full = results[0]

    files = sorted(chunk_dir.glob("r*_k*.npz"))
    units = {}
    for f in files:
        r, k = (int(s[1:]) for s in f.stem.split("_"))
        if (r, k) in cells:
            units[(r, k)] = load_unit(f)
    assert len(units) == len(cells), f"单元个数 {len(units)}，应为 {len(cells)}"
    res = merge_units(units, (R, K, len(xbar), len(DIMS)))

    joblib.dump(full["encoder"], model_dir / "encoder.joblib")
    for d, m in full["models"].items():
        joblib.dump(m, model_dir / f"{d}.joblib")
    wall = time.time() - t0

    repeats = sorted({r for r, _ in cells})
    once = all((res.fold_id[r] >= 0).all() for r in repeats if sum(rr == r for rr, _ in cells) == K)
    isolated = all(
        not set(date_group[u["train_idx"]]) & set(date_group[u["test_idx"]]) for u in units.values()
    )
    inner_rows_ok = all(
        np.array_equal(np.where(~np.isnan(res.inner_mu[r, k]).any(1))[0], u["train_idx"])
        for (r, k), u in units.items()
    )
    n_finite = int(np.isfinite(res.inner_mu).sum())
    n_expected = int(sum(len(u["train_idx"]) for u in units.values()) * len(DIMS))
    mae, r2, rho = fold_metrics(res.oof_mu, xbar, repeats)
    n_over = int((r2 > reliability + 0.05).sum())

    checks = [
        make_check("外层单元个数", len(units), len(cells) if args.smoke else R * K),
        make_check("每次重复每款啤酒恰好测试一次", once, True),
        make_check("同一日期组不跨外层折", isolated, True),
        make_check("inner_mu[r,k] 的非 NaN 行等于训练行", inner_rows_ok, True),
        make_check(
            "inner_mu 有限值个数 / 含 inf",
            [n_finite, bool(np.isinf(res.inner_mu).any())],
            [n_expected, False],
        ),
    ]
    if not args.smoke:
        checks += [
            make_check("折外连续 MAE", mae, 0.479, 0.02),
            make_check("折外 R²", float(np.mean(r2)), 0.167, 0.04),
            make_check("折外 Spearman", float(np.mean(rho)), 0.368, 0.04),
            make_check("折外 R² 超过 reliability + 0.05 的维度数", n_over, 0),
        ]
    print(f"折外连续 MAE {mae:.4f}，R² {np.mean(r2):.4f}，Spearman {np.mean(rho):.4f}")
    excluded = {}
    for r, k in cells:
        for f, (_, b) in enumerate(
            inner_splits(np.sort(splits[r][k][0]), date_group, style, SEED + 100 + r)
        ):
            for j in np.where(np.std(xbar[b], axis=0) == 0)[0]:
                excluded.setdefault(f"{r}_{k}", []).append(f"内层折{f}:{DIMS[j]}")
    for f, (_, b) in enumerate(
        inner_splits(np.arange(len(xbar)), date_group, style, SEED + 200, 5)
    ):
        for j in np.where(np.std(xbar[b], axis=0) == 0)[0]:
            excluded.setdefault("full", []).append(f"内层折{f}:{DIMS[j]}")
    print(f"RMSE/SD 无定义而剔除的单元: {excluded}")

    out = out_dir / "q1_main_oof.npz"
    np.savez_compressed(
        out,
        oof_mu=res.oof_mu,
        inner_mu=res.inner_mu,
        fold_id=res.fold_id,
        chosen_params=json.dumps(res.chosen),
        inner_scores=json.dumps({k: v.tolist() for k, v in res.scores.items()}),
        final_params=json.dumps(full["chosen"]),
        split_diag=json.dumps(split_diag),
        inner_undefined=json.dumps(excluded, ensure_ascii=False),
        wall_time_s=wall,
        checks=json.dumps(checks, ensure_ascii=False),
        seed=SEED,
        meta=run_meta(),
    )

    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    for g, s in selection_summary(res.chosen).items():
        print(f"{g}: 不同配置 {s['n_distinct']} 种，mono=True {s['n_mono_true']} 次")
        for cfg, n in s["counts"].items():
            print(f"    {n:2d} × {cfg}")
    print(f"全数据重拟合选定: {json.dumps(full['chosen'])}")
    print(f"wall_time_s = {wall:.0f}")
    print(f"写出 {out}")

    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
