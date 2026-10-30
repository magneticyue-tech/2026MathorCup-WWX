import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1_data import make_check  # noqa: E402
from src.q1.baselines import (  # noqa: E402
    EN_CONFIGS,
    LIN_CONFIGS,
    RF_CONFIGS,
    RULES,
    b2_fit_dim,
    b2_log_columns,
    b2_prepare,
    chemstyle_input,
    fit_outer_fold_generic,
    fit_predict_enet,
    fit_predict_lin10,
    fit_predict_rf,
    merge_generic,
    prep_linear,
    prep_plain,
    simple_fold,
)
from src.q1.data import DIM_GROUPS, DIMS, load_q1  # noqa: E402
from src.q1.discretize import apply_rules, dist_interval, dist_median  # noqa: E402
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.model import fit_outer_fold  # noqa: E402
from src.q1.paths import CHUNK_DIR, PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q1.splits import outer_splits, paper_split  # noqa: E402

MODELS = ("b0", "b1", "lin10", "enet", "rf", "chemstyle")
PARALLEL_MODELS = ("rf", "chemstyle", "enet", "lin10")


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def run_unit(model, path, X, Xs, Y, groups, style, tr, te, inner_seed, rf_configs):
    t0 = time.time()
    if model in ("b0", "b1"):
        u = simple_fold(model, Y, style, groups, tr, te, inner_seed)
        u["chosen"], u["scores"] = {}, {}
    elif model == "chemstyle":
        u = fit_outer_fold(Xs, Y, DIMS, DIM_GROUPS, groups, style, tr, te, inner_seed)
    else:
        configs, prep, fp = {
            "lin10": (LIN_CONFIGS, prep_linear, fit_predict_lin10),
            "enet": (EN_CONFIGS, prep_linear, fit_predict_enet),
            "rf": (rf_configs, prep_plain, fit_predict_rf),
        }[model]
        u = fit_outer_fold_generic(
            X, Y, DIMS, DIM_GROUPS, groups, style, tr, te, inner_seed, configs, prep, fp
        )
    np.savez_compressed(
        path,
        train_idx=u["train_idx"],
        test_idx=u["test_idx"],
        oof=u["oof"],
        inner=u["inner"],
        chosen=json.dumps(_jsonable(u["chosen"])),
        scores=json.dumps(_jsonable(u["scores"])),
        wall_s=time.time() - t0,
    )
    return path


def load_unit(path):
    z = np.load(path, allow_pickle=False)
    return {
        "train_idx": z["train_idx"],
        "test_idx": z["test_idx"],
        "oof": z["oof"],
        "inner": z["inner"],
        "chosen": json.loads(str(z["chosen"])),
        "wall_s": float(z["wall_s"]),
    }


def rules_for_model(oof, inner, fold_id, xbar, cells):
    R, n, D = oof.shape
    grades = {rule: np.full((R, n, D), -1, dtype=int) for rule in RULES}
    p = np.full((R, n, D, 7), np.nan)
    for r, k in cells:
        te = np.where(fold_id[r] == k)[0]
        tr = np.where(fold_id[r] != k)[0]
        for d in range(D):
            out = apply_rules(oof[r, te, d], inner[r, k, tr, d], xbar[tr, d])
            p[r, te, d] = out["p"]
            for rule in RULES:
                grades[rule][r, te, d] = out[rule]
    return grades, p


def cont_metrics(mu, xbar, reps):
    mae, r2, rho = [], [], []
    for r in reps:
        p = mu[r]
        mae.append(np.mean(np.abs(p - xbar)))
        r2.append(np.mean(1 - ((xbar - p) ** 2).sum(0) / ((xbar - xbar.mean(0)) ** 2).sum(0)))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*constant.*")
            rho.append(np.nanmean([spearmanr(p[:, d], xbar[:, d])[0] for d in range(p.shape[1])]))
    return float(np.mean(mae)), float(np.mean(r2)), float(np.mean(rho))


def b2_dim(X_tr, y_tr, X_te, y_te):
    pred, params = b2_fit_dim(X_tr, y_tr, X_te)
    return pred, float(r2_score(y_te, pred)), params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", action="append", choices=MODELS)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers 必须在 1 到 8 之间")

    require_sklearn()
    ensure_dirs()
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    chunk_dir = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / "baselines"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rf_configs = [{**c, "n_estimators": 20} for c in RF_CONFIGS] if args.smoke else RF_CONFIGS
    t0 = time.time()

    z = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=False)
    xbar, G, date_group = z["xbar"], z["G"], z["date_group"]
    main_oof = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=False)
    fold_id_main, oof_main = main_oof["fold_id"], main_oof["oof_mu"]
    grade_const_main = np.load(PROCESSED_DIR / "q1_scores.npz", allow_pickle=False)["grade_const"]
    data = load_q1()
    X, Zdf, style = data.X, data.Z, data.style
    Xs = chemstyle_input(X, style)
    splits = outer_splits(date_group, style)
    R, K = len(splits), len(splits[0])
    split_ok = all(
        np.array_equal(np.sort(splits[r][k][1]), np.where(fold_id_main[r] == k)[0])
        for r in range(R)
        for k in range(K)
    )
    assert split_ok, "外层划分与 q1_main_oof.npz 的 fold_id 不一致"
    cells = [(0, 0), (0, 1)] if args.smoke else [(r, k) for r in range(R) for k in range(K)]
    models = args.only if args.only else list(MODELS)

    def path(m, r, k):
        return chunk_dir / f"{m}_r{r}_k{k}.npz"

    def job(m, r, k):
        tr, te = splits[r][k]
        return (
            m,
            path(m, r, k),
            X,
            Xs,
            xbar,
            date_group,
            style,
            tr,
            te,
            SEED + 100 + r,
            rf_configs,
        )

    todo = [(m, r, k) for m in models for r, k in cells if not path(m, r, k).exists()]
    for m, r, k in [t for t in todo if t[0] in ("b0", "b1")]:
        run_unit(*job(m, r, k))
    par = sorted(
        [t for t in todo if t[0] in PARALLEL_MODELS], key=lambda t: PARALLEL_MODELS.index(t[0])
    )
    print(f"待计算单元 {len(todo)} 个，其中并行 {len(par)} 个", flush=True)
    if par:
        Parallel(n_jobs=args.workers, verbose=10)(delayed(run_unit)(*job(*t)) for t in par)
    if args.only:
        print(f"已写出中间文件：{sorted(set(args.only))}")
        return

    merged, chosen, wall = {}, {}, {}
    for m in MODELS:
        units = {(r, k): load_unit(path(m, r, k)) for r, k in cells}
        assert len(units) == len(cells), f"{m} 单元个数不符"
        merged[m] = merge_generic(units, len(xbar), len(DIMS), R, K)
        chosen[m] = {
            f"{r}_{k}_{g}": v for (r, k), u in units.items() for g, v in u["chosen"].items()
        }
        wall[m] = float(sum(u["wall_s"] for u in units.values()))

    ruled = Parallel(n_jobs=min(args.workers, len(MODELS)))(
        delayed(rules_for_model)(merged[m][0], merged[m][1], merged[m][2], xbar, cells)
        for m in MODELS
    )
    grades = {m: g for m, (g, _) in zip(MODELS, ruled)}
    probs = {m: p for m, (_, p) in zip(MODELS, ruled)}

    p_cs = probs["chemstyle"]
    cs_p = p_cs.mean(0)
    cs_lo, cs_hi = dist_interval(cs_p) if not args.smoke else (None, None)

    tr_b2, te_b2 = paper_split(style, seed=SEED)
    tr_b2, te_b2 = np.sort(tr_b2), np.sort(te_b2)
    Xb_tr, Xb_te = b2_prepare(X.iloc[tr_b2], X.iloc[te_b2])
    log_cols = b2_log_columns(X.iloc[tr_b2])
    Zv = Zdf.to_numpy(float)
    b2_dims = range(3) if args.smoke else range(len(DIMS))
    b2 = Parallel(n_jobs=args.workers)(
        delayed(b2_dim)(Xb_tr, Zv[tr_b2, d], Xb_te, Zv[te_b2, d]) for d in b2_dims
    )
    b2_pred = np.full((len(te_b2), len(DIMS)), np.nan)
    b2_r2 = np.full(len(DIMS), np.nan)
    b2_params = {}
    for d, (pred, r2, params) in zip(b2_dims, b2):
        b2_pred[:, d], b2_r2[d], b2_params[DIMS[d]] = pred, r2, params

    tested = merged["b0"][2] >= 0
    reps = sorted({r for r, _ in cells})
    inner_ok = all(
        np.array_equal(
            np.where(~np.isnan(merged[m][1][r, k]).any(1))[0], np.where(merged[m][2][r] != k)[0]
        )
        for m in MODELS
        for r, k in cells
    )
    oof_finite = all(np.isfinite(merged[m][0][tested]).all() for m in MODELS)
    b0_same = bool((grades["b0"]["const"][tested] == grade_const_main[tested]).all())
    cs_diff = not np.array_equal(merged["chemstyle"][0][tested], oof_main[tested])
    checks = [
        make_check("外层划分与 q1_main_oof.npz 的 fold_id 一致", split_ok, True),
        make_check("6 个模型的 inner 非 NaN 行恰为训练行", inner_ok, True),
        make_check("6 个模型的 oof 全部有限", oof_finite, True),
        make_check("grade_b0_const 与 q1_scores.npz 的 grade_const 逐元素相同", b0_same, True),
        make_check("chemstyle 的 oof 与主模型 oof 不完全相同", cs_diff, True),
        make_check("B2 测试集款数", len(te_b2), 75),
    ]
    summary = {}
    if not args.smoke:
        for m in MODELS:
            mae, r2, rho = cont_metrics(merged[m][0], xbar, reps)
            gm = float(np.mean([np.abs(grades[m]["median"][r] - G).mean() for r in reps]))
            summary[m] = {
                "cont_mae": mae,
                "cont_r2": r2,
                "cont_spearman": rho,
                "grade_mae_median": gm,
            }
        b2_mean = float(np.mean(b2_r2))
        checks += [
            make_check("B1 连续 MAE", summary["b1"]["cont_mae"], 0.527, 0.02),
            make_check("B1 R²", summary["b1"]["cont_r2"], 0.053, 0.04),
            make_check("B1 Spearman", summary["b1"]["cont_spearman"], 0.177, 0.04),
            make_check("B0 R² ≤ 0.001", summary["b0"]["cont_r2"] <= 0.001, True),
            make_check(
                "B1 R² 大于 B0 R²", summary["b1"]["cont_r2"] > summary["b0"]["cont_r2"], True
            ),
            make_check("B2 50 维平均测试 R²", b2_mean, 0.189, 0.03),
        ]

    arrays = {}
    for m in MODELS:
        arrays[f"oof_{m}"] = merged[m][0]
        arrays[f"inner_{m}"] = merged[m][1]
        arrays[f"chosen_{m}"] = json.dumps(_jsonable(chosen[m]))
        arrays[f"p_{m}"] = probs[m]
        for rule in RULES:
            arrays[f"grade_{m}_{rule}"] = grades[m][rule]
    if not args.smoke:
        arrays.update(
            chemstyle_final_p=cs_p,
            chemstyle_final_G_hat=dist_median(cs_p),
            chemstyle_final_G_hat_Q=np.median(grades["chemstyle"]["quant"], axis=0).astype(int),
            chemstyle_final_lo=cs_lo,
            chemstyle_final_hi=cs_hi,
            chemstyle_final_mu_hat=merged["chemstyle"][0].mean(0),
        )
    wall_total = time.time() - t0
    out = out_dir / "q1_baselines.npz"
    np.savez_compressed(
        out,
        **arrays,
        b2_test_idx=te_b2,
        b2_pred=b2_pred,
        b2_r2=b2_r2,
        b2_params=json.dumps(_jsonable(b2_params)),
        b2_log_cols=json.dumps({DIMS[d]: len(log_cols) for d in b2_dims}),
        wall_time_s=json.dumps({**wall, "script": wall_total}),
        checks=json.dumps(checks, ensure_ascii=False),
        seed=SEED,
        meta=run_meta(),
    )

    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    for m, s in summary.items():
        print(m, {k: round(v, 4) for k, v in s.items()})
    for m in ("enet", "rf"):
        for g in DIM_GROUPS:
            picks = [json.dumps(v, sort_keys=True) for k, v in chosen[m].items() if k.endswith(g)]
            print(f"{m} {g}: 不同配置 {len(set(picks))} 种 {dict(Counter(picks).most_common(3))}")
    print(
        f"B2 对数列 {len(log_cols)} 个；wall_s {json.dumps({k: round(v) for k, v in wall.items()})}"
    )
    print(f"脚本耗时 {wall_total:.0f} 秒；写出 {out}")
    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
