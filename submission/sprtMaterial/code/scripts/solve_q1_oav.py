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
from solve_q1_metrics import has_string_numbers, sanitize  # noqa: E402
from src.q1.data import DIM_GROUPS, DIMS, ChemNAEncoder, load_q1  # noqa: E402
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.metrics import (  # noqa: E402
    PAIR_STATS,
    bootstrap_weights,
    cluster_bootstrap,
    pair_stat_fn,
    wins_by_dim,
)
from src.q1.oav import (  # noqa: E402
    OAV_CLASSES,
    exceed_count,
    fit_fold_oav,
    inner_ratio,
    load_thresholds,
    oav_class_sums,
    threshold_units,
)
from src.q1.paths import CHUNK_DIR, PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q1.sensitivity import changed_share, delivered, merge_arm, unit_grades  # noqa: E402
from src.q1.splits import outer_splits  # noqa: E402


def load(name):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def run_unit(path, X, xbar, groups, style, tr, te, inner_seed, chosen, thr, add):
    u = fit_fold_oav(X, xbar, DIMS, DIM_GROUPS, groups, style, tr, te, inner_seed, chosen, thr, add)
    np.savez_compressed(path, **u, chosen=json.dumps(chosen, sort_keys=True))
    return path


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
    chunk_dir = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / "oav"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_boot = 20 if args.smoke else 2000
    t0 = time.time()

    z = load("q1_data.npz")
    xbar, G, date_group, zero_heavy = z["xbar"], z["G"], z["date_group"], z["zero_heavy"]
    mo = load("q1_main_oof.npz")
    chosen_params = json.loads(str(mo["chosen_params"]))
    inner_main, oof_main, fold_id = mo["inner_mu"], mo["oof_mu"], mo["fold_id"]
    sc = load("q1_scores.npz")
    data = load_q1()
    X, style = data.X, data.style
    splits = outer_splits(date_group, style)
    R, K = len(splits), len(splits[0])
    split_ok = all(
        np.array_equal(np.sort(splits[r][k][1]), np.where(fold_id[r] == k)[0])
        for r in range(R)
        for k in range(K)
    )
    assert split_ok, "外层划分与 fold_id 不一致"
    thr = load_thresholds()
    cells = [(0, 0), (0, 1)] if args.smoke else [(r, k) for r in range(R) for k in range(K)]
    id_cells = [(0, 0), (0, 1)]

    def chosen_of(r, k):
        return {g: chosen_params[f"{r}_{k}_{g}"] for g in DIM_GROUPS}

    jobs = []
    plan = [(f"r{r}_k{k}", r, k, True) for r, k in cells]
    plan += [(f"identity_r{r}_k{k}", r, k, False) for r, k in id_cells]
    for name, r, k, add in plan:
        p = chunk_dir / f"{name}.npz"
        if not p.exists():
            tr, te = splits[r][k]
            jobs.append(
                delayed(run_unit)(
                    p, X, xbar, date_group, style, tr, te, SEED + 100 + r, chosen_of(r, k), thr, add
                )
            )
    print(f"待计算单元 {len(jobs)} 个", flush=True)
    if jobs:
        Parallel(n_jobs=args.workers, verbose=10)(jobs)

    units, chosen_ok = {}, True
    for r, k in cells:
        zz = np.load(chunk_dir / f"r{r}_k{k}.npz", allow_pickle=False)
        u = {key: zz[key] for key in ("train_idx", "test_idx", "oof", "inner")}
        chosen_ok &= json.loads(str(zz["chosen"])) == chosen_of(r, k)
        u.update(unit_grades(u["oof"], u["inner"], xbar[u["train_idx"]]))
        units[(r, k)] = u
    id_err = max(
        float(
            np.abs(
                np.load(chunk_dir / f"identity_r{r}_k{k}.npz")["oof"]
                - oof_main[r, np.sort(splits[r][k][1])]
            ).max()
        )
        for r, k in id_cells
    )
    merged = merge_arm(units, shape=(R, *xbar.shape), n_fold=K)
    reps = sorted({r for r, _ in cells})
    dv = delivered(merged["p"][reps])
    tested = merged["grade_median"][:, :, 0] >= 0
    rows = tested[reps].all(0)

    ratio_arm = np.full((R, K, len(DIMS)), np.nan)
    ratio_main = np.full((R, K, len(DIMS)), np.nan)
    for (r, k), u in units.items():
        tr = u["train_idx"]
        ratio_arm[r, k] = inner_ratio(u["inner"], xbar, tr)
        ratio_main[r, k] = inner_ratio(inner_main[r, k, tr], xbar, tr)
    diff = ratio_arm - ratio_main
    valid = ~np.isnan(diff)
    n_valid = valid.sum(-1)
    fold_mean = np.where(n_valid > 0, np.nansum(diff, -1) / np.maximum(n_valid, 1), np.nan)
    in_cells = np.zeros((R, K), bool)
    for r, k in cells:
        in_cells[r, k] = True
    inner_reduced_folds = int(((fold_mean < 0) & in_cells & (n_valid > 0)).sum())
    group_diff = {
        g: float(np.nanmean(diff[in_cells][:, [DIMS.index(d) for d in ds]]))
        for g, ds in DIM_GROUPS.items()
    }

    a = {
        k: merged[v][reps][:, rows]
        for k, v in (("g", "grade_median"), ("p", "p"), ("mu", "oof_mu"))
    }
    b = {
        "g": sc["grade_median"][reps][:, rows],
        "p": sc["p"][reps][:, rows],
        "mu": oof_main[reps][:, rows],
    }
    W = bootstrap_weights(date_group[rows], n_boot=n_boot)
    point, lo, hi = cluster_bootstrap(pair_stat_fn(G[rows], a, b, xbar[rows]), W)
    pair = {
        "point": dict(zip(PAIR_STATS, point.tolist())),
        "lo": dict(zip(PAIR_STATS, lo.tolist())),
        "hi": dict(zip(PAIR_STATS, hi.tolist())),
        "wins_a": wins_by_dim(G[rows], a["g"], b["g"]),
    }
    zh, other = np.where(zero_heavy)[0], np.where(~zero_heavy)[0]

    def sub_mae(g, idx):
        return float(np.abs(g[:, :, idx] - G[rows][None][:, :, idx]).mean())

    inner_reduced = inner_reduced_folds >= 13
    mae_ci_below_zero = bool(pair["hi"]["mae"] < 0)
    review_candidate = bool(inner_reduced and mae_ci_below_zero)

    enc_all = ChemNAEncoder("col_mean").fit(X).transform(X)
    oav_all = oav_class_sums(enc_all, thr)
    n_cols = enc_all.shape[1] + oav_all.shape[1]
    n_exceed = exceed_count(X, thr)
    units_map = threshold_units()
    classes = {
        name: [
            {"field": c, "threshold": thr[c][0], "unit": units_map[c], "log10": thr[c][1]}
            for c in cols
        ]
        for name, cols in OAV_CLASSES.items()
    }
    once_ok = all(tested[r].all() for r in reps) or args.smoke
    inner_ok = all(
        np.array_equal(
            np.where(~np.isnan(merged["inner_mu"][r, k]).any(1))[0], units[(r, k)]["train_idx"]
        )
        for r, k in cells
    )
    checks = [
        make_check("外层划分与 q1_main_oof.npz 的 fold_id 一致", split_ok, True),
        make_check("单元个数", len(units), len(cells) if args.smoke else 25),
        make_check("每次重复每款啤酒恰好测试一次", once_ok, True),
        make_check("inner_mu 的非 NaN 行恰为训练行", inner_ok, True),
        make_check("oof_mu 全部有限", bool(np.isfinite(merged["oof_mu"][tested]).all()), True),
        make_check(
            "p_final 每行和为 1", float(np.abs(dv["p_final"][rows].sum(-1) - 1).max()), 0.0, 1e-12
        ),
        make_check("实验臂超参数与主运行 chosen_params 逐字段相同", bool(chosen_ok), True),
        make_check("恒等单元的 oof 与 q1_main_oof.npz 的 oof_mu 逐元素相同", id_err, 0.0, 1e-10),
        make_check("阈值化合物个数", len(thr), 43),
        make_check(
            "酯类、硫化合物个数",
            [len(OAV_CLASSES["esters"]), len(OAV_CLASSES["sulfur"])],
            [13, 4],
        ),
        make_check("在样本内越过阈值的带阈值化合物个数", n_exceed, 29, 2),
        make_check("编码后化学表加 OAV 两列的列数", n_cols, 233),
    ]
    out_json = sanitize(
        {
            "classes": classes,
            "pair": pair,
            "changed_delivered": changed_share(dv["G_hat"][rows], sc["G_hat"][rows]),
            "changed_median": changed_share(a["g"], b["g"]),
            "zero_heavy_delta_mae": sub_mae(a["g"], zh) - sub_mae(b["g"], zh),
            "other_delta_mae": sub_mae(a["g"], other) - sub_mae(b["g"], other),
            "inner_valid_dims": np.where(in_cells, n_valid, 0).tolist(),
            "inner_fold_mean_diff": [
                [None if not in_cells[r, k] else float(fold_mean[r, k]) for k in range(K)]
                for r in range(R)
            ],
            "inner_fold_mean_diff_reason": "未运行的外层折或该折无有效维度（目标在该折训练集恒定）",
            "inner_group_mean_diff": group_diff,
            "inner_reduced_folds": inner_reduced_folds,
            "inner_reduced": bool(inner_reduced),
            "mae_ci_below_zero": mae_ci_below_zero,
            "review_candidate": review_candidate,
            "exceed_count": n_exceed,
            "seed": SEED,
            "meta": json.loads(run_meta()),
        },
        reason="无定义",
    )
    checks.append(
        make_check(
            "结果 JSON 中没有字符串形式的数值或状态（除 _reason 字段）",
            not has_string_numbers({k: v for k, v in out_json.items() if k != "classes"}),
            True,
        )
    )
    out_json["checks"] = checks
    wall = time.time() - t0
    with open(out_dir / "q1_oav.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=1, allow_nan=False)
    np.savez_compressed(
        out_dir / "q1_oav.npz",
        oof_mu=merged["oof_mu"],
        inner_mu=merged["inner_mu"],
        grade_median=merged["grade_median"],
        p_final=dv["p_final"],
        G_hat=dv["G_hat"],
        inner_ratio_arm=ratio_arm,
        inner_ratio_main=ratio_main,
        oav_esters=oav_all["oav_esters"].to_numpy(),
        oav_sulfur=oav_all["oav_sulfur"].to_numpy(),
        checks=json.dumps(checks, ensure_ascii=False),
        wall_time_s=wall,
        seed=SEED,
        meta=run_meta(),
    )
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    p_ = pair
    for k in ("mae", "qwk", "exact", "cont_mae"):
        print(f"{k}: {p_['point'][k]:+.4f} [{p_['lo'][k]:+.4f}, {p_['hi'][k]:+.4f}]")
    print(
        f"wins_a {p_['wins_a']}；changed_delivered {out_json['changed_delivered']:.4f}；"
        f"inner_reduced_folds {inner_reduced_folds}；review_candidate {review_candidate}；"
        f"组内 RMSE/SD 差 {json.dumps({g: round(v, 4) for g, v in group_diff.items()})}；"
        f"wall_time_s {wall:.0f}"
    )
    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
