import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1_data import make_check  # noqa: E402
from solve_q1_metrics import has_string_numbers, sanitize  # noqa: E402
from src.q1.data import DIM_GROUPS, DIMS, ChemNAEncoder, load_q1, load_raw_ratings  # noqa: E402
from src.q1.features import monotone_cst  # noqa: E402
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.metrics import (  # noqa: E402
    PAIR_STATS,
    bootstrap_weights,
    cluster_bootstrap,
    conf_weighted,
    exact_dim,
    mae_dim,
    mean_qwk,
    pair_stat_fn,
    qwk_dim,
    wins_by_dim,
)
from src.q1.model import PARAM_KEYS, make_hgb  # noqa: E402
from src.q1.paths import CHUNK_DIR, PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q1.sensitivity import (  # noqa: E402
    ARMS,
    SUM_COLS,
    alpha_target,
    arm_kwargs,
    changed_share,
    delivered,
    fit_fold_fixed,
    merge_arm,
    unit_grades,
)
from src.q1.splits import outer_splits  # noqa: E402

UNIT_KEYS = ("train_idx", "test_idx", "oof", "inner", "median", "round", "quant", "p")


def load(name):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def run_unit(path, arm, X_arm, kw, Y_fit, groups, style, tr, te, inner_seed, chosen):
    u = fit_fold_fixed(
        X_arm, Y_fit, DIMS, DIM_GROUPS, groups, style, tr, te, inner_seed, chosen, **kw
    )
    g = unit_grades(u["oof"], u["inner"], Y_fit[u["train_idx"]])
    np.savez_compressed(path, **u, **g, chosen=json.dumps(chosen, sort_keys=True))
    return path


def load_unit(path):
    z = np.load(path, allow_pickle=False)
    u = {k: z[k] for k in UNIT_KEYS}
    u["chosen"] = json.loads(str(z["chosen"]))
    return u


def sha12(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def subset_mae(G, g, idx):
    return float(np.abs(g[:, :, idx] - G[None][:, :, idx]).mean())


def fit_drop_sum_final(X, xbar, final_params, out_dir):
    X_use = X.drop(columns=list(SUM_COLS))
    enc = ChemNAEncoder("col_mean").fit(X_use)
    A_df = enc.transform(X_use)
    names = list(A_df.columns)
    A = A_df.to_numpy(np.float64)
    group_of = {d: g for g, ds in DIM_GROUPS.items() for d in ds}
    model_dir = out_dir / "q1_models_drop_sum"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(enc, model_dir / "encoder.joblib")
    used = {}
    for j, d in enumerate(DIMS):
        best = final_params[group_of[d]]
        params = {k: best[k] for k in PARAM_KEYS}
        m = make_hgb(params, monotone_cst(names, d, best["mono"])).fit(A, xbar[:, j])
        joblib.dump(m, model_dir / f"{d}.joblib")
        used[d] = {**params, "mono": bool(best["mono"])}
    manifest = {
        "input_columns": names,
        "n_input_columns": len(names),
        "params_by_dim": used,
        "params_by_group": final_params,
        "target": "q1_data.npz 的 xbar（品评小组均分）",
        "seed": SEED,
        "sources": {
            "q1_data.npz": sha12(PROCESSED_DIR / "q1_data.npz"),
            "q1_main_oof.npz": sha12(PROCESSED_DIR / "q1_main_oof.npz"),
        },
        "n_models": len(list(model_dir.glob("*.joblib"))) - 1,
        "has_encoder": (model_dir / "encoder.joblib").exists(),
        "meta": json.loads(run_meta()),
    }
    with open(out_dir / "q1_drop_sum_final.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1, allow_nan=False)
    return manifest, used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arm", action="append")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers 必须在 1 到 8 之间")
    require_sklearn()
    ensure_dirs()
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    chunk_dir = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / "sens"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_boot = 20 if args.smoke else 2000
    t0 = time.time()

    z = load("q1_data.npz")
    xbar, G, date_group, zero_heavy = z["xbar"], z["G"], z["date_group"], z["zero_heavy"]
    mo = load("q1_main_oof.npz")
    oof_main, fold_id = mo["oof_mu"], mo["fold_id"]
    chosen_params = json.loads(str(mo["chosen_params"]))
    final_params = json.loads(str(mo["final_params"]))
    sc = load("q1_scores.npz")
    data = load_q1()
    X, beer, style = data.X, data.beer, data.style
    ratings = load_raw_ratings()
    splits = outer_splits(date_group, style)
    R, K = len(splits), len(splits[0])
    split_ok = all(
        np.array_equal(np.sort(splits[r][k][1]), np.where(fold_id[r] == k)[0])
        for r in range(R)
        for k in range(K)
    )
    assert split_ok, "外层划分与 fold_id 不一致"

    if args.arm:
        arms = list(args.arm)
    elif args.smoke:
        arms = ["identity", "alpha"]
    else:
        arms = ["identity", *ARMS]
    for a in arms:
        if a != "identity" and a not in ARMS:
            raise ValueError(f"未知臂: {a}")
    all_cells = [(r, k) for r in range(R) for k in range(K)]
    id_cells = [(0, 0), (0, 1)]

    def cells_of(arm):
        return id_cells if (arm == "identity" or args.smoke) else all_cells

    def path(arm, r, k):
        return chunk_dir / f"{arm}_r{r}_k{k}.npz"

    arm_tables = {a: arm_kwargs(a, X, beer) for a in arms}
    alpha_cache = {}
    jobs = []
    for a in arms:
        X_arm, kw = arm_tables[a]
        for r, k in cells_of(a):
            if path(a, r, k).exists():
                continue
            tr, te = splits[r][k]
            if a == "alpha":
                if (r, k) not in alpha_cache:
                    alpha_cache[(r, k)] = alpha_target(ratings, tr)
                Y_fit = alpha_cache[(r, k)]
            else:
                Y_fit = xbar
            chosen = {g: chosen_params[f"{r}_{k}_{g}"] for g in DIM_GROUPS}
            jobs.append(
                delayed(run_unit)(
                    path(a, r, k),
                    a,
                    X_arm,
                    kw,
                    Y_fit,
                    date_group,
                    style,
                    tr,
                    te,
                    SEED + 100 + r,
                    chosen,
                )
            )
    print(f"待计算单元 {len(jobs)} 个", flush=True)
    if jobs:
        Parallel(n_jobs=args.workers, verbose=10)(jobs)
    if args.arm:
        print(f"已写出中间文件：{arms}")
        return

    units = {a: {(r, k): load_unit(path(a, r, k)) for r, k in cells_of(a)} for a in arms}
    chosen_ok = all(
        u["chosen"] == {g: chosen_params[f"{r}_{k}_{g}"] for g in DIM_GROUPS}
        for a in arms
        for (r, k), u in units[a].items()
    )
    id_oof_err = max(
        float(np.abs(u["oof"] - oof_main[r, u["test_idx"]]).max())
        for (r, k), u in units["identity"].items()
    )
    identity_oof = np.concatenate([units["identity"][c]["oof"] for c in id_cells])

    merged, deliv = {}, {}
    for a in arms:
        if a == "identity":
            continue
        merged[a] = merge_arm(units[a], shape=(R, *xbar.shape), n_fold=K)
        deliv[a] = delivered(merged[a]["p"][sorted({r for r, _ in cells_of(a)})])
    reps_by_arm = {a: sorted({r for r, _ in cells_of(a)}) for a in merged}
    tested = {a: merged[a]["grade_median"][:, :, 0] >= 0 for a in merged}
    rows_of = {a: tested[a][reps_by_arm[a]].all(0) for a in merged}

    inner_ok = all(
        np.array_equal(
            np.where(~np.isnan(merged[a]["inner_mu"][r, k]).any(1))[0],
            np.sort(units[a][(r, k)]["train_idx"]),
        )
        for a in merged
        for r, k in cells_of(a)
    )
    finite_ok = all(np.isfinite(merged[a]["oof_mu"][tested[a]]).all() for a in merged)
    once_ok = (
        all((tested[a][r]).all() for a in merged for r in sorted({r for r, _ in cells_of(a)}))
        or args.smoke
    )
    p_ok = max(float(np.abs(deliv[a]["p_final"].sum(-1) - 1)[rows_of[a]].max()) for a in merged)

    tr0, te0 = splits[0][0]
    r_mod = ratings.copy()
    r_mod.loc[r_mod["row"].isin(np.asarray(te0) + 1), DIMS] = (
        6 - r_mod.loc[r_mod["row"].isin(np.asarray(te0) + 1), DIMS]
    )
    a_ref, a_mod = alpha_target(ratings, tr0), alpha_target(r_mod, tr0)
    alpha_ok = bool(np.array_equal(a_ref[np.sort(tr0)], a_mod[np.sort(tr0)]))
    X_att8, _ = arm_kwargs("att8", X, beer)
    att8_cells = int((~((X_att8 == X) | (X_att8.isna() & X.isna()))).values.sum())
    att8_beers = int((~((X_att8 == X) | (X_att8.isna() & X.isna()))).any(axis=1).sum())
    n_drop = X.drop(columns=list(SUM_COLS)).shape[1]
    n_nan = int(X.isna().sum().sum())
    n_med = sum(c.endswith("_missing") for c in ChemNAEncoder("median_ind").fit(X).transform(X))
    n_flo = sum(c.endswith("_missing") for c in ChemNAEncoder("floor_ind").fit(X).transform(X))

    checks = [
        make_check("敏感性外层划分与 q1_main_oof.npz 的 fold_id 一致", split_ok, True),
        make_check("恒等臂的外层预测与主运行 oof_mu 逐元素相同", id_oof_err, 0.0, 1e-10),
        make_check("各臂 inner_mu 的非 NaN 行恰为训练行", inner_ok, True),
        make_check("各臂 oof_mu 全部有限", finite_ok, True),
        make_check("各臂每次重复每款啤酒恰好测试一次", once_ok, True),
        make_check("各臂 p_final 每行和为 1", p_ok, 0.0, 1e-12),
        make_check("各臂所用超参数与主运行 chosen_params 逐字段相同", chosen_ok, True),
        make_check("att8 臂替换单元数 / 啤酒数", [att8_cells, att8_beers], [80, 24]),
        make_check("drop_sum 臂化学表列数", n_drop, 227),
        make_check("na_native_nan 臂的 NaN 单元数", n_nan, 73),
        make_check("na_median_ind、na_floor_ind 臂的缺失指示列数", [n_med, n_flo], [9, 8]),
        make_check("alpha 臂训练目标只含训练行评分", alpha_ok, True),
    ]

    W = bootstrap_weights(date_group, n_boot=n_boot)
    b_main = {"g": sc["grade_median"], "p": sc["p"], "mu": oof_main}
    zh, other = np.where(zero_heavy)[0], np.where(~zero_heavy)[0]
    comp_arms = [a for a in merged if not args.smoke or a != "identity"]
    sides = []
    for a in comp_arms:
        reps = reps_by_arm[a]
        sa = {
            "g": merged[a]["grade_median"][reps],
            "p": merged[a]["p"][reps],
            "mu": merged[a]["oof_mu"][reps],
        }
        sb = {k: v[reps] for k, v in b_main.items()}
        if args.smoke:
            rows = tested[a][reps].all(0)
            sa = {k: v[:, rows] for k, v in sa.items()}
            sb = {k: v[:, rows] for k, v in sb.items()}
            sides.append((sa, sb, G[rows], xbar[rows], W[:, rows]))
        else:
            sides.append((sa, sb, G, xbar, W))
    res = Parallel(n_jobs=args.workers)(
        delayed(cluster_bootstrap)(pair_stat_fn(Gs, sa, sb, xs), Ws) for sa, sb, Gs, xs, Ws in sides
    )
    arms_out = {}
    for a, (sa, sb, Gs, xs, _), (point, lo, hi) in zip(comp_arms, sides, res):
        rows = rows_of[a]
        arms_out[a] = {
            "point": dict(zip(PAIR_STATS, point.tolist())),
            "lo": dict(zip(PAIR_STATS, lo.tolist())),
            "hi": dict(zip(PAIR_STATS, hi.tolist())),
            "wins_a": wins_by_dim(Gs, sa["g"], sb["g"]),
            "changed_delivered": changed_share(deliv[a]["G_hat"][rows], sc["G_hat"][rows]),
            "changed_median": changed_share(sa["g"], sb["g"]),
            "zero_heavy_delta_mae": subset_mae(Gs, sa["g"], zh) - subset_mae(Gs, sb["g"], zh),
            "other_delta_mae": subset_mae(Gs, sa["g"], other) - subset_mae(Gs, sb["g"], other),
        }
    alpha_flag = bool("alpha" in arms_out and arms_out["alpha"]["hi"]["mae"] < 0)

    jb = DIMS.index("barnyard")
    keep = [d for d in range(len(DIMS)) if d != jb]
    gm = sc["grade_median"]
    barnyard = {}
    for name, idx in (("dims_49", keep), ("dims_50", list(range(len(DIMS))))):
        conf = conf_weighted(G[:, idx], gm[:, :, idx])
        q, n_undef = mean_qwk(qwk_dim(conf))
        barnyard[name] = {
            "mae": float(mae_dim(conf).mean()),
            "exact": float(exact_dim(conf).mean()),
            "qwk": q,
            "qwk_undefined": n_undef,
        }
    barnyard_reason = (
        "超参数固定且其他维度的模型不依赖 barnyard，排除 barnyard 后其余 49 个维度的交付等级"
        "与主设定完全相同；此处只报告主设定在 49 维与 50 维上的指标，不做配对差与 CI"
    )

    if not args.smoke:
        mae_deltas = [abs(arms_out[a]["point"]["mae"]) for a in arms_out]
        changed = [arms_out[a]["changed_delivered"] for a in arms_out]
        mono_diff = abs(arms_out["mono_on"]["point"]["mae"] - arms_out["mono_off"]["point"]["mae"])
        manifest, used = fit_drop_sum_final(X, xbar, final_params, out_dir)
        group_of = {d: g for g, ds in DIM_GROUPS.items() for d in ds}
        params_same = all(
            used[d] == {k: final_params[group_of[d]][k] for k in (*PARAM_KEYS, "mono")}
            for d in DIMS
        )
        checks += [
            make_check(
                "Q3 drop_sum 全数据模型参数与 solve_q1_main.py final_params 逐字段相同",
                params_same,
                True,
            ),
            make_check(
                "Q3 drop_sum 全数据模型输入列数 / 维度模型数 / 编码器",
                [manifest["n_input_columns"], manifest["n_models"], manifest["has_encoder"]],
                [227, 50, True],
            ),
            make_check(
                "alpha - main 的 cont_mae 点估计",
                arms_out["alpha"]["point"]["cont_mae"],
                -0.0026,
                0.004,
            ),
            make_check("各臂等级 MAE 与主设定之差的绝对值 ≤ 0.03", max(mae_deltas) <= 0.03, True),
            make_check("各臂 changed_delivered ≤ 0.15", max(changed) <= 0.15, True),
            make_check(
                "mono_on 与 mono_off 的等级 MAE 之差的绝对值 ≤ 0.03", mono_diff <= 0.03, True
            ),
        ]

    out_json = sanitize(
        {
            "arms": arms_out,
            "barnyard_excluded": barnyard,
            "barnyard_reason": barnyard_reason,
            "alpha_flag": alpha_flag,
            "seed": SEED,
            "meta": json.loads(run_meta()),
        }
    )
    checks.append(
        make_check(
            "所有结果 JSON 中没有字符串形式的数值或状态（除 _reason 字段）",
            not has_string_numbers(out_json),
            True,
        )
    )
    out_json["checks"] = checks
    with open(out_dir / "q1_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=1, allow_nan=False)
    arrays = {}
    for a in merged:
        arrays[f"{a}__oof_mu"] = merged[a]["oof_mu"]
        arrays[f"{a}__grade_median"] = merged[a]["grade_median"]
        arrays[f"{a}__p_final"] = deliv[a]["p_final"]
        arrays[f"{a}__G_hat"] = deliv[a]["G_hat"]
    wall = time.time() - t0
    np.savez_compressed(
        out_dir / "q1_sensitivity.npz",
        **arrays,
        arms=np.array(arms),
        identity__oof=identity_oof,
        checks=json.dumps(checks, ensure_ascii=False),
        seed=SEED,
        meta=run_meta(),
        wall_time_s=wall,
    )
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    for a, v in arms_out.items():
        p, lo, hi = v["point"], v["lo"], v["hi"]
        print(
            f"{a}: mae {p['mae']:+.4f} [{lo['mae']:+.4f}, {hi['mae']:+.4f}] "
            f"qwk {p['qwk']:+.4f} cont {p['cont_mae']:+.4f} changed {v['changed_delivered']:.4f}"
        )
    print(f"alpha_flag={alpha_flag}；wall_time_s {wall:.0f}")
    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
