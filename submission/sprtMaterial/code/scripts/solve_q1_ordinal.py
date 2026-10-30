import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1_data import make_check  # noqa: E402
from src.q1.data import DIM_GROUPS, DIMS, load_q1  # noqa: E402
from src.q1.discretize import dist_median, qwk_score  # noqa: E402
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.ordinal import k2_fold  # noqa: E402
from src.q1.paths import CHUNK_DIR, PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q1.splits import outer_splits  # noqa: E402


def run_unit(path, X, xbar, tr, te, chosen):
    u = k2_fold(X, xbar, DIMS, DIM_GROUPS, tr, te, chosen)
    np.savez_compressed(path, test_idx=u["test_idx"], p=u["p"], fallback=u["fallback"])
    return path


def mean_qwk(G, Gh, reps):
    return float(
        np.mean(
            [np.nanmean([qwk_score(G[:, d], Gh[r, :, d]) for d in range(G.shape[1])]) for r in reps]
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers 必须在 1 到 8 之间")

    require_sklearn()
    ensure_dirs()
    io_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    chunk_dir = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / "ordinal"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    z = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=False)
    xbar, G, date_group = z["xbar"], z["G"], z["date_group"]
    m = np.load(io_dir / "q1_main_oof.npz", allow_pickle=False)
    chosen_params, fold_id = json.loads(str(m["chosen_params"])), m["fold_id"]
    grade_median = np.load(io_dir / "q1_scores.npz", allow_pickle=False)["grade_median"]
    data = load_q1()
    splits = outer_splits(date_group, data.style)
    R, K = len(splits), len(splits[0])
    cells = [(r, k) for r in range(R) for k in range(K) if (fold_id[r] == k).any()]
    split_ok = all(
        np.array_equal(np.sort(splits[r][k][1]), np.where(fold_id[r] == k)[0]) for r, k in cells
    )
    assert split_ok, "外层划分与 fold_id 不一致"

    def path(r, k):
        return chunk_dir / f"r{r}_k{k}.npz"

    todo = [(r, k) for r, k in cells if not path(r, k).exists()]
    print(f"单元 {len(cells)} 个，待计算 {len(todo)} 个", flush=True)
    Parallel(n_jobs=args.workers, verbose=10)(
        delayed(run_unit)(
            path(r, k),
            data.X,
            xbar,
            *splits[r][k],
            {g: chosen_params[f"{r}_{k}_{g}"] for g in DIM_GROUPS},
        )
        for r, k in todo
    )

    n, D = xbar.shape
    p_k2 = np.full((R, n, D, 7), np.nan)
    fallback = np.zeros((R, K, D, 6), dtype=bool)
    writes = np.zeros((R, n), dtype=int)
    units = sorted(f for f in chunk_dir.glob("r*_k*.npz"))
    loaded = 0
    for r, k in cells:
        u = np.load(path(r, k), allow_pickle=False)
        p_k2[r, u["test_idx"]] = u["p"]
        fallback[r, k] = u["fallback"]
        writes[r, u["test_idx"]] += 1
        loaded += 1
    assert loaded == len(cells) and len(units) >= len(cells)
    full_reps = [r for r in range(R) if all((r, k) in cells for k in range(K))]
    once = bool((writes.max() <= 1) and all((writes[r] == 1).all() for r in full_reps))
    assert once, "有啤酒在某次重复中未恰好测试一次"

    tested = fold_id >= 0
    grade_k2 = np.full((R, n, D), -1, dtype=int)
    grade_k2[tested] = dist_median(p_k2[tested])
    reps = sorted({r for r, _ in cells})

    checks = [
        make_check("外层划分与 q1_main_oof.npz 的 fold_id 一致", split_ok, True),
        make_check("K2 单元个数", len(cells), len(cells) if args.smoke else 25),
        make_check("每次重复每款啤酒恰好测试一次", once, True),
        make_check("p_k2 每行和为 1", float(np.abs(p_k2[tested].sum(-1) - 1).max()), 0.0, 1e-12),
        make_check("p_k2 无负值", bool((p_k2[tested] >= 0).all()), True),
        make_check(
            "所有等级数组取值在 0..6",
            bool((grade_k2[tested] >= 0).all() and (grade_k2[tested] <= 6).all()),
            True,
        ),
    ]
    mae_k2 = float(np.mean([np.abs(grade_k2[r] - G).mean() for r in reps]))
    mae_main = float(np.mean([np.abs(grade_median[r] - G).mean() for r in reps]))
    qwk_k2 = mean_qwk(G, grade_k2, reps) if not args.smoke else float("nan")
    if not args.smoke:
        checks += [
            make_check("K2 预测分布中位数 等级 MAE", mae_k2, 0.489, 0.02),
            make_check("K2 QWK", qwk_k2, 0.285, 0.03),
            make_check("K2 减主模型的等级 MAE", mae_k2 - mae_main, 0.005, 0.02),
        ]

    wall = time.time() - t0
    out = io_dir / "q1_ordinal.npz"
    np.savez_compressed(
        out,
        p_k2=p_k2,
        grade_k2_median=grade_k2,
        fallback=fallback,
        checks=json.dumps(checks, ensure_ascii=False),
        wall_time_s=wall,
        seed=SEED,
        meta=run_meta(),
    )
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    n_fb = int(fallback[[r for r, _ in cells], [k for _, k in cells]].sum())
    print(f"K2 回退阈值个数 {n_fb}；wall_time_s {wall:.0f}；写出 {out}")
    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
