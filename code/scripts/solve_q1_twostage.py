import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1_data import make_check  # noqa: E402
from solve_q1_main import load_unit, save_unit  # noqa: E402
from src.q1.data import DIM_GROUPS, DIMS, load_q1  # noqa: E402
from src.q1.discretize import qwk_score  # noqa: E402
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.model import fit_outer_fold  # noqa: E402
from src.q1.panels import infer_panels  # noqa: E402
from src.q1.paths import CHUNK_DIR, PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q1.splits import outer_splits  # noqa: E402
from src.q1.twostage import C2_RULES, c1_fold, c2_fold, select_c1_map  # noqa: E402


def run_unit(path, X, Z, groups, style, tr, te, inner_seed):
    u = fit_outer_fold(X, Z, DIMS, DIM_GROUPS, groups, style, tr, te, inner_seed)
    save_unit(path, u)
    return path


def rep_mean(fn, G, Gh, reps, rows):
    return float(np.mean([fn(G[rows[r]], Gh[r][rows[r]]) for r in reps]))


def mean_qwk(G, Gh, reps):
    return float(
        np.mean(
            [np.nanmean([qwk_score(G[:, d], Gh[r, :, d]) for d in range(G.shape[1])]) for r in reps]
        )
    )


def ref_stats(R, G, mask):
    r, g = R[mask], np.broadcast_to(G, R.shape)[mask]
    return {
        "cells_per_rep": float(mask.sum() / R.shape[0]),
        "exact": float((r == g).mean()),
        "within1": float((np.abs(r - g) <= 1).mean()),
        "mae": float(np.abs(r - g).mean()),
        "bias": float((r - g).mean()),
    }


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
    chunk_dir = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / "twostage_z"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    z = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=False)
    xbar, G, date_group = z["xbar"], z["G"], z["date_group"]
    m = np.load(io_dir / "q1_main_oof.npz", allow_pickle=False)
    fold_id, oof_mu = m["fold_id"], m["oof_mu"]
    s = np.load(io_dir / "q1_scores.npz", allow_pickle=False)
    grade_round_main = s["grade_round"]
    data = load_q1()
    Zdf, style = data.Z, data.style
    Z = Zdf.to_numpy(float)
    splits = outer_splits(date_group, style)
    R, K = len(splits), len(splits[0])
    cells = [(r, k) for r in range(R) for k in range(K) if (fold_id[r] == k).any()]
    split_ok = all(
        np.array_equal(np.sort(splits[r][k][1]), np.where(fold_id[r] == k)[0]) for r, k in cells
    )
    assert split_ok, "外层划分与 fold_id 不一致"

    def path(r, k):
        return chunk_dir / f"r{r}_k{k}.npz"

    todo = [(r, k) for r, k in cells if not path(r, k).exists()]
    print(f"z 模型单元 {len(cells)} 个，待计算 {len(todo)} 个", flush=True)
    Parallel(n_jobs=args.workers, verbose=10)(
        delayed(run_unit)(path(r, k), data.X, Z, date_group, style, *splits[r][k], SEED + 100 + r)
        for r, k in todo
    )
    units = {(r, k): load_unit(path(r, k)) for r, k in cells}
    assert len(units) == len(cells)

    panels = infer_panels(Zdf, k=3)
    n, D = Z.shape
    oof_z = np.full((R, n, D), np.nan)
    inner_z = np.full((R, K, n, D), np.nan)
    sigma = np.full((R, K, D), np.nan)
    gc1 = {k: np.full((R, n, D), -1, dtype=int) for k in ("round_T", "qwk_T", "quant_T")}
    R_te = np.full((R, n, D), -1, dtype=int)
    I_te = np.zeros((R, n, D), dtype=bool)
    oof_c2 = np.full((R, n, D), np.nan)
    inner_c2 = np.full((R, K, n, D), np.nan)
    coef_c2 = np.full((R, K, D, 2), np.nan)
    p_c2 = np.full((R, n, D, 7), np.nan)
    gc2 = {rule: np.full((R, n, D), -1, dtype=int) for rule in C2_RULES}
    chosen_z = {}
    for (r, k), u in units.items():
        tr, te = u["train_idx"], u["test_idx"]
        oof_z[r, te] = u["oof"]
        inner_z[r, k, tr] = u["inner"]
        for g, v in u["chosen"].items():
            chosen_z[f"{r}_{k}_{g}"] = v
        c1 = c1_fold(Z, xbar, panels, DIMS, tr, te, u["oof"], u["inner"])
        sigma[r, k] = c1["sigma"]
        R_te[r, te], I_te[r, te] = c1["R_te"], c1["I_te"]
        for key in gc1:
            gc1[key][r, te] = c1[key]
        c2 = c2_fold(xbar, Z, tr, te, u["oof"], u["inner"])
        oof_c2[r, te], inner_c2[r, k, tr], coef_c2[r, k] = c2["mu"], c2["inner_mu"], c2["coef"]
        p_c2[r, te] = c2["p"]
        for rule in C2_RULES:
            gc2[rule][r, te] = c2["grades"][rule]

    tested = fold_id >= 0
    reps = sorted({r for r, _ in cells})
    if tested[reps].all():
        c1_sel, c1_scores = select_c1_map(
            R_te[reps], {k: gc1[k][reps] for k in ("round_T", "qwk_T")}
        )
    else:
        c1_sel, c1_scores = "round_T", {"reason": "冒烟运行只含部分外层折，不做选型"}
    grade_c1_sel = gc1[c1_sel]

    labels, counts = np.unique(panels, return_counts=True)
    multi = labels[counts > 1]
    panels_ok = all(len(np.unique(fold_id[r][panels == g])) == 1 for r in reps for g in multi)
    inner_ok = all(
        np.array_equal(np.where(~np.isnan(a[r, k]).any(1))[0], np.where(fold_id[r] != k)[0])
        for a in (inner_z, inner_c2)
        for r, k in cells
    )
    all_grades = [*gc1.values(), *gc2.values()]
    grades_ok = all(((g[tested] >= 0) & (g[tested] <= 6)).all() for g in all_grades)
    checks = [
        make_check("外层划分与 q1_main_oof.npz 的 fold_id 一致", split_ok, True),
        make_check("z 模型单元个数", len(units), len(cells) if args.smoke else 25),
        make_check(
            "每次重复每款啤酒恰好测试一次",
            all(tested[r].all() for r in reps) or args.smoke,
            True,
        ),
        make_check("p_c2 每行和为 1", float(np.abs(p_c2[tested].sum(-1) - 1).max()), 0.0, 1e-12),
        make_check("所有等级数组取值在 0..6", bool(grades_ok), True),
        make_check(
            "R_te 取值在 0..6", bool(((R_te[tested] >= 0) & (R_te[tested] <= 6)).all()), True
        ),
        make_check(f"{len(multi)} 个多啤酒品评团（k=3）不跨外层折", panels_ok, True),
        make_check("inner_z、inner_c2 的非 NaN 行恰为训练行", inner_ok, True),
    ]

    def mae(Gh):
        return rep_mean(lambda a, b: np.abs(a - b).mean(), G, Gh, reps, tested)

    Rb = R_te[reps]
    Ib = I_te[reps]
    rows_b = np.broadcast_to(tested[reps][:, :, None], Rb.shape)
    ref = {
        "all": ref_stats(Rb, G, rows_b),
        "A": ref_stats(Rb, G, Ib & rows_b),
        "L": ref_stats(Rb, G, ~Ib & rows_b),
    }
    summary = {
        "c1_round_mae": mae(gc1["round_T"]),
        "c1_quant_mae": mae(gc1["quant_T"]),
        "c1_sel_mae": mae(grade_c1_sel),
        "main_round_mae": mae(grade_round_main),
        "I_share": float(Ib[rows_b].mean()),
        "c2_round_mae": mae(gc2["round"]),
        "c2_quant_mae": mae(gc2["quant"]),
        "c2_cont_mae": float(
            np.mean([np.abs(oof_c2[r][tested[r]] - xbar[tested[r]]).mean() for r in reps])
        ),
        "main_cont_mae": float(
            np.mean([np.abs(oof_mu[r][tested[r]] - xbar[tested[r]]).mean() for r in reps])
        ),
    }
    if not args.smoke:
        summary["c1_quant_qwk"] = mean_qwk(G, gc1["quant_T"], reps)
        summary["c2_quant_qwk"] = mean_qwk(G, gc2["quant"], reps)
        checks += [
            make_check("C1 四舍五入 等级 MAE", summary["c1_round_mae"], 0.737, 0.05),
            make_check("C1 分位数切点 等级 MAE", summary["c1_quant_mae"], 0.760, 0.05),
            make_check("C1 分位数切点 QWK", summary["c1_quant_qwk"], 0.333, 0.03),
            make_check("C1 选定映射 等级 MAE", summary["c1_sel_mae"], 0.786, 0.05),
            make_check(
                "C1 四舍五入的 MAE 大于主模型四舍五入的 MAE",
                summary["c1_round_mae"] > summary["main_round_mae"],
                True,
            ),
            make_check("局部参照单元占比", summary["I_share"], 0.307, 0.03),
            make_check("R 对 G 的完全一致率（全部）", ref["all"]["exact"], 0.629, 0.03),
            make_check("R 对 G 的等级 MAE（全部）", ref["all"]["mae"], 0.427, 0.04),
            make_check("R 对 G 的平均偏差（全部）", ref["all"]["bias"], 0.295, 0.06),
            make_check("局部参照单元的完全一致率", ref["A"]["exact"], 0.886, 0.04),
            make_check("线性参照单元的完全一致率", ref["L"]["exact"], 0.515, 0.04),
            make_check("C2 四舍五入 等级 MAE", summary["c2_round_mae"], 0.493, 0.015),
            make_check("C2 分位数切点 等级 MAE", summary["c2_quant_mae"], 0.569, 0.02),
            make_check("C2 分位数切点 QWK", summary["c2_quant_qwk"], 0.362, 0.02),
            make_check(
                "C2 连续 MAE 与主模型连续 MAE 之差的绝对值 ≤ 0.01",
                abs(summary["c2_cont_mae"] - summary["main_cont_mae"]) <= 0.01,
                True,
            ),
        ]

    wall = time.time() - t0
    out = io_dir / "q1_twostage.npz"
    np.savez_compressed(
        out,
        grade_c1_round=gc1["round_T"],
        grade_c1_qwk=gc1["qwk_T"],
        grade_c1_quant=gc1["quant_T"],
        grade_c1_sel=grade_c1_sel,
        R_te=R_te,
        I_te=I_te,
        c1_selected=c1_sel,
        c1_select_scores=json.dumps(c1_scores, ensure_ascii=False),
        oof_c2=oof_c2,
        inner_c2=inner_c2,
        coef_c2=coef_c2,
        p_c2=p_c2,
        **{f"grade_c2_{rule}": gc2[rule] for rule in C2_RULES},
        oof_z=oof_z,
        inner_z=inner_z,
        chosen_params_z=json.dumps(chosen_z),
        sigma=sigma,
        panels=panels,
        checks=json.dumps(checks, ensure_ascii=False),
        wall_time_s=wall,
        seed=SEED,
        meta=run_meta(),
    )
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    print({k: round(v, 4) for k, v in summary.items()})
    print(f"C1 选型 {c1_sel} {c1_scores}")
    print("参照误差", json.dumps(ref))
    for g in DIM_GROUPS:
        picks = [json.dumps(v, sort_keys=True) for k, v in chosen_z.items() if k.endswith(g)]
        print(f"z 模型 {g}: 不同配置 {len(Counter(picks))} 种")
    print(f"wall_time_s {wall:.0f}；写出 {out}")
    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
