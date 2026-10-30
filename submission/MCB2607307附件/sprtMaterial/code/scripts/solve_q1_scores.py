import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1_data import make_check  # noqa: E402
from src.q1.discretize import (  # noqa: E402
    apply_rules,
    dist_interval,
    dist_median,
    quantile_cuts,
    qwk_score,
)
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.paths import PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402

RULES = ("const", "round", "median", "quant", "qwk_cut", "median_knn")


def rep_mean(fn, G, Gh, cells):
    vals = [fn(G[cells[r]], Gh[r][cells[r]]) for r in range(len(Gh)) if cells[r].any()]
    return float(np.mean(vals))


def mean_qwk(G, Gh, rows):
    out = []
    for r in range(len(Gh)):
        if not rows[r].any():
            continue
        q = [qwk_score(G[rows[r], d], Gh[r, rows[r], d]) for d in range(G.shape[1])]
        out.append(np.nanmean(q))
    return float(np.mean(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    require_sklearn()
    ensure_dirs()
    io_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    t0 = time.time()

    z = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=False)
    xbar, G = z["xbar"], z["G"]
    m = np.load(io_dir / "q1_main_oof.npz", allow_pickle=False)
    oof_mu, inner_mu, fold_id = m["oof_mu"], m["inner_mu"], m["fold_id"]
    R, K, n, D = inner_mu.shape

    units = []
    for r in range(R):
        for k in range(K):
            te = np.where(fold_id[r] == k)[0]
            if len(te) == 0:
                continue
            tr = np.where(fold_id[r] != k)[0]
            filled = np.where(~np.isnan(inner_mu[r, k]).any(1))[0]
            assert np.array_equal(filled, tr), f"inner_mu[{r},{k}] 的非 NaN 行不等于训练行"
            units.append((r, k, tr, te))
    print(f"外层折 {len(units)} 个", flush=True)

    p = np.full((R, n, D, 7), np.nan)
    p_knn = np.full((R, n, D, 7), np.nan)
    grades = {rule: np.full((R, n, D), -1, dtype=int) for rule in RULES}
    for r, k, tr, te in units:
        for d in range(D):
            out = apply_rules(oof_mu[r, te, d], inner_mu[r, k, tr, d], xbar[tr, d])
            p[r, te, d], p_knn[r, te, d] = out["p"], out["p_knn"]
            for rule in RULES:
                grades[rule][r, te, d] = out[rule]

    tested = fold_id >= 0
    full_reps = [r for r in range(R) if tested[r].all()]
    deliver_reps = full_reps if full_reps else sorted({u[0] for u in units})
    deliver_rows = tested[deliver_reps].all(0)
    p_final = p[deliver_reps].mean(axis=0)
    G_hat = np.full((n, D), -1, dtype=int)
    lo = np.full((n, D), -1, dtype=int)
    hi = np.full((n, D), -1, dtype=int)
    G_hat[deliver_rows] = dist_median(p_final[deliver_rows])
    lo_, hi_ = dist_interval(p_final[deliver_rows])
    lo[deliver_rows], hi[deliver_rows] = lo_, hi_
    G_hat_Q = np.full((n, D), -1, dtype=int)
    G_hat_Q[deliver_rows] = np.median(
        grades["quant"][deliver_reps][:, deliver_rows], axis=0
    ).astype(int)

    if full_reps == list(range(R)):
        final_resid = np.stack([(xbar[None, :, d] - oof_mu[:, :, d]).ravel() for d in range(D)])
        final_qcuts = np.stack(
            [quantile_cuts(oof_mu[:, :, d].mean(axis=0), G[:, d]) for d in range(D)]
        )
    else:
        final_resid = np.full((D, R * n), np.nan)
        final_qcuts = np.full((D, 6), np.nan)

    dr = deliver_rows
    p_rows = p[:, :, :, :][tested]
    all_grades_ok = all(
        g[tested].min() >= 0 and g[tested].max() <= 6 for g in grades.values()
    ) and bool(((G_hat[dr] >= 0) & (G_hat[dr] <= 6)).all())
    checks = [
        make_check(
            "p_final 每行和为 1",
            float(np.abs(p_final[dr].sum(-1) - 1).max()),
            0.0,
            1e-12,
        ),
        make_check(
            "lo ≤ G_hat ≤ hi 处处成立",
            bool(((lo[dr] <= G_hat[dr]) & (G_hat[dr] <= hi[dr])).all()),
            True,
        ),
        make_check(
            "G_hat 等于由保存的 p_final 重算的中位数",
            bool((dist_median(p_final[dr]) == G_hat[dr]).all()),
            True,
        ),
        make_check("各等级数组取值在 0..6", bool(all_grades_ok), True),
        make_check(
            "grade_qwk_cut 与 grade_quant 逐元素相同的占比小于 1",
            bool((grades["qwk_cut"][tested] == grades["quant"][tested]).mean() < 1),
            True,
        ),
    ]
    assert np.isfinite(p_rows).all()

    mae = {
        rule: rep_mean(lambda a, b: np.abs(a - b).mean(), G, grades[rule], tested) for rule in RULES
    }
    ex_med = rep_mean(lambda a, b: (a == b).mean(), G, grades["median"], tested)
    qwk_round = mean_qwk(G, grades["round"], tested)
    qwk_quant = mean_qwk(G, grades["quant"], tested)
    s56_quant = rep_mean(lambda a, b: (b >= 5).mean(), G, grades["quant"], tested)
    s56_med = rep_mean(lambda a, b: (b >= 5).mean(), G, grades["median"], tested)
    cover = float(((G[dr] >= lo[dr]) & (G[dr] <= hi[dr])).mean())
    width = float((hi[dr] - lo[dr]).mean())
    print(
        "等级 MAE "
        + "，".join(f"{k} {v:.4f}" for k, v in mae.items())
        + f"；中位数完全一致 {ex_med:.4f}；QWK 四舍五入 {qwk_round:.4f}、分位数 {qwk_quant:.4f}"
        + f"；5–6 级占比 分位数 {s56_quant:.4f}、中位数 {s56_med:.4f}"
        + f"；覆盖率 {cover:.4f}，宽度 {width:.4f}"
    )
    if not args.smoke:
        checks += [
            make_check("常量规则等级 MAE", mae["const"], 0.550, 0.015),
            make_check("四舍五入等级 MAE", mae["round"], 0.495, 0.015),
            make_check("预测分布中位数等级 MAE", mae["median"], 0.484, 0.015),
            make_check("近邻残差中位数等级 MAE", mae["median_knn"], 0.477, 0.015),
            make_check("分位数切点等级 MAE", mae["quant"], 0.571, 0.02),
            make_check("预测分布中位数完全一致率", ex_med, 0.574, 0.015),
            make_check("四舍五入 QWK", qwk_round, 0.310, 0.02),
            make_check("分位数切点 QWK", qwk_quant, 0.370, 0.02),
            make_check("分位数切点 5–6 级占比", s56_quant, 0.0146, 0.005),
            make_check("预测分布中位数 5–6 级占比 ≤ 0.001", s56_med <= 0.001, True),
            make_check("中位数 MAE 低于四舍五入 MAE", mae["median"] < mae["round"], True),
            make_check("90% 区间覆盖率", cover, 0.969, 0.03),
            make_check("90% 区间平均宽度", width, 1.95, 0.10),
        ]

    out = io_dir / "q1_scores.npz"
    np.savez_compressed(
        out,
        p=p,
        p_knn=p_knn,
        **{f"grade_{rule}": grades[rule] for rule in RULES},
        p_final=p_final,
        G_hat=G_hat,
        lo=lo,
        hi=hi,
        G_hat_Q=G_hat_Q,
        final_resid=final_resid,
        final_qcuts=final_qcuts,
        checks=json.dumps(checks, ensure_ascii=False),
        seed=SEED,
        meta=run_meta(),
    )
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    print(f"耗时 {time.time() - t0:.0f} 秒；写出 {out}")

    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
