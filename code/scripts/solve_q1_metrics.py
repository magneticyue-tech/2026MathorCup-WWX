import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1_data import make_check  # noqa: E402
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.metrics import (  # noqa: E402
    ALL_RULES,
    CTRL_MODELS,
    N_BOOT,
    PAIR_STATS,
    balanced_accuracy,
    bootstrap_weights,
    climatology_p,
    cluster_bootstrap,
    conf_weighted,
    confusion_pooled,
    coverage,
    exact_dim,
    grade_freq,
    logloss_cells,
    mae_dim,
    mean_qwk,
    pair_stat_fn,
    qwk_dim,
    r2_by_dim,
    recall_by_class,
    reference_error,
    rmse_over_sd,
    rps_cells,
    share_high,
    spearman_by_dim,
    subset_report,
    wins_by_dim,
    wins_by_dim_cont,
    within1_dim,
)
from src.q1.paths import PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402

CTRL_VS_MAIN = (*CTRL_MODELS, "k2", "c1", "c2")
REASONS = {
    "rps": "该模型或规则没有预测分布",
    "logloss": "该模型或规则没有预测分布",
    "cont": "该模型没有 x̄ 尺度的连续预测",
    "qwk": "全部单元的 QWK 无定义",
    "pair": "至少一方缺少该统计量所需的预测分布或连续预测",
}


def load(name):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def entry_metrics(G, g, p, mu_stats, zh_idx, other_idx):
    conf = conf_weighted(G, g)
    q, n_nan = mean_qwk(qwk_dim(conf))
    e = {
        "mae": float(mae_dim(conf).mean()),
        "exact": float(exact_dim(conf).mean()),
        "within1": float(within1_dim(conf).mean()),
        "qwk": q,
        "qwk_undefined": n_nan,
        "balanced_acc": balanced_accuracy(confusion_pooled(G, g)),
        "share56": share_high(g),
    }
    if p is not None:
        e["rps"] = float(rps_cells(p, G).mean())
        e["logloss"] = float(logloss_cells(p, G).mean())
    else:
        e["rps"] = e["logloss"] = None
        e["rps_reason"] = e["logloss_reason"] = REASONS["rps"]
    if mu_stats is not None:
        e.update(mu_stats)
    else:
        for k in ("cont_mae", "cont_r2", "cont_spearman", "cont_rmse_over_sd"):
            e[k] = None
            e[f"{k}_reason"] = REASONS["cont"]
    e["zero_heavy"] = subset_report(G, g, zh_idx)
    e["other"] = subset_report(G, g, other_idx)
    return e


def cont_stats(mu, xbar):
    rho = spearman_by_dim(mu, xbar)
    return {
        "cont_mae": float(np.abs(mu - xbar[None]).mean()),
        "cont_r2": float(r2_by_dim(mu, xbar).mean()),
        "cont_spearman": float(np.nanmean(rho)),
        "cont_spearman_undefined": int(np.isnan(rho).sum()),
        "cont_rmse_over_sd": float(rmse_over_sd(mu, xbar).mean()),
    }


def run_pair(G, a, b, xbar, W):
    point, lo, hi = cluster_bootstrap(pair_stat_fn(G, a, b, xbar), W)
    return point, lo, hi


def side(GR, PR, MU, model, rule):
    s = {"g": GR[(model, rule)]}
    if (model, rule) in PR:
        s["p"] = PR[(model, rule)]
    if model in MU:
        s["mu"] = MU[model]
    return s


def sanitize(obj, reason=None):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, float) and math.isnan(v):
                out[k] = None
                out.setdefault(f"{k}_reason", reason or REASONS.get(k.split("_")[0], "无定义"))
            else:
                out[k] = sanitize(v, reason)
        return out
    if isinstance(obj, (list, tuple)):
        return [
            None if isinstance(v, float) and math.isnan(v) else sanitize(v, reason) for v in obj
        ]
    if isinstance(obj, np.generic):
        return sanitize(obj.item(), reason)
    return obj


def has_string_numbers(obj, key=""):
    if isinstance(obj, dict):
        return any(has_string_numbers(v, k) for k, v in obj.items())
    if isinstance(obj, list):
        return any(has_string_numbers(v, key) for v in obj)
    if isinstance(obj, str) and not key.endswith("_reason"):
        if obj.strip().upper() in {"PASSED", "FAILED", "PASS", "FAIL", "SKIPPED", "TRUE", "FALSE"}:
            return True
        try:
            float(obj)
            return True
        except ValueError:
            return False
    return False


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
    n_boot = 20 if args.smoke else N_BOOT
    t0 = time.time()

    z = load("q1_data.npz")
    xbar, G, date_group = z["xbar"], z["G"], z["date_group"]
    reliability, zero_heavy, dims = z["reliability"], z["zero_heavy"], [str(d) for d in z["dims"]]
    main_oof = load("q1_main_oof.npz")
    sc = load("q1_scores.npz")
    bl = load("q1_baselines.npz")
    od = load("q1_ordinal.npz")
    ts = load("q1_twostage.npz")
    fold_id, oof_mu = main_oof["fold_id"], main_oof["oof_mu"]
    zh_idx, other_idx = np.where(zero_heavy)[0], np.where(~zero_heavy)[0]

    GR, PR, MU = {}, {}, {}
    for rule in ALL_RULES:
        GR[("main", rule)] = sc[f"grade_{rule}"]
        for m in CTRL_MODELS:
            GR[(m, rule)] = bl[f"grade_{m}_{rule}"]
        GR[("c2", rule)] = ts[f"grade_c2_{rule}"]
    GR[("k2", "median")] = od["grade_k2_median"]
    for key, name in (("sel", "sel"), ("round", "round"), ("qwk", "qwk"), ("quant", "quant")):
        GR[("c1", key)] = ts[f"grade_c1_{name}"]
    PR[("main", "median")], PR[("main", "median_knn")] = sc["p"], sc["p_knn"]
    for m in CTRL_MODELS:
        PR[(m, "median")] = bl[f"p_{m}"]
    PR[("k2", "median")], PR[("c2", "median")] = od["p_k2"], ts["p_c2"]
    MU["main"] = oof_mu
    for m in CTRL_MODELS:
        MU[m] = bl[f"oof_{m}"]
    MU["c2"] = ts["oof_c2"]

    shape = (len(fold_id), *G.shape)
    grades_ok = all(g.shape == shape and g.min() >= 0 and g.max() <= 6 for g in GR.values())
    probs_ok = max(float(np.abs(p.sum(-1) - 1).max()) for p in PR.values())

    cont = {m: cont_stats(mu, xbar) for m, mu in MU.items()}
    models = {}
    conf_sums_ok = True
    for (m, rule), g in GR.items():
        conf_sums_ok &= bool((conf_weighted(G, g).sum(axis=(1, 2, 3)) == G.size).all())
        models[f"{m}.{rule}"] = entry_metrics(
            G, g, PR.get((m, rule)), cont.get(m), zh_idx, other_idx
        )

    clim = climatology_p(G, fold_id)
    climatology = {
        "rps": float(rps_cells(clim, G).mean()),
        "logloss": float(logloss_cells(clim, G).mean()),
    }

    delivered = {}
    for key in ("G_hat", "G_hat_Q"):
        g1 = sc[key][None]
        conf = conf_weighted(G, g1)
        q, n_nan = mean_qwk(qwk_dim(conf))
        delivered[key] = {
            "mae": float(mae_dim(conf).mean()),
            "exact": float(exact_dim(conf).mean()),
            "within1": float(within1_dim(conf).mean()),
            "qwk": q,
            "qwk_undefined": n_nan,
            "balanced_acc": balanced_accuracy(confusion_pooled(G, g1)),
            "share56": share_high(g1),
        }
    cov, width = coverage(G, sc["lo"], sc["hi"])
    delivered["interval"] = {"coverage": cov, "width": width}
    pf = sc["p_final"][None]
    delivered["p_final"] = {
        "rps": float(rps_cells(pf, G).mean()),
        "logloss": float(logloss_cells(pf, G).mean()),
    }

    conf_keys = {
        "main_median": GR[("main", "median")],
        "k2_median": GR[("k2", "median")],
        "c1_sel": GR[("c1", "sel")],
        "G_hat": sc["G_hat"][None],
    }
    confusion = {}
    for k, g in conf_keys.items():
        cm = confusion_pooled(G, g)
        confusion[k] = {"matrix": cm.tolist(), "recall": recall_by_class(cm).tolist()}
    freq = {
        "truth": grade_freq(G).tolist(),
        "main.median": grade_freq(GR[("main", "median")]).tolist(),
        "main.quant": grade_freq(GR[("main", "quant")]).tolist(),
        "G_hat": grade_freq(sc["G_hat"]).tolist(),
        "G_hat_Q": grade_freq(sc["G_hat_Q"]).tolist(),
    }
    ref = reference_error(ts["R_te"], ts["I_te"], G)

    r2_main = r2_by_dim(oof_mu, xbar).mean(0)
    leak_dims = [dims[d] for d in range(len(dims)) if r2_main[d] > reliability[d]]

    pairs = [(("main", "median"), ("b1", "median"))]
    pairs += [((m, "sel" if m == "c1" else "median"), ("main", "median")) for m in CTRL_VS_MAIN]
    pairs += [
        (("main", "median"), ("main", "round")),
        (("main", "quant"), ("main", "round")),
        (("main", "quant"), ("main", "qwk_cut")),
        (("main", "median_knn"), ("main", "median")),
        (("main", "round"), ("c1", "round")),
        (("main", "quant"), ("c1", "quant")),
        (("main", "quant"), ("c1", "sel")),
        (("main", "quant"), ("c2", "quant")),
    ]
    W = bootstrap_weights(date_group, n_boot=n_boot)
    sides = [(side(GR, PR, MU, *a), side(GR, PR, MU, *b)) for a, b in pairs]
    res = Parallel(n_jobs=args.workers)(delayed(run_pair)(G, sa, sb, xbar, W) for sa, sb in sides)
    paired = []
    for (a, b), (sa, sb), (point, lo, hi) in zip(pairs, sides, res):
        item = {
            "a": f"{a[0]}.{a[1]}",
            "b": f"{b[0]}.{b[1]}",
            "point": dict(zip(PAIR_STATS, point.tolist())),
            "lo": dict(zip(PAIR_STATS, lo.tolist())),
            "hi": dict(zip(PAIR_STATS, hi.tolist())),
            "wins_a": wins_by_dim(G, sa["g"], sb["g"]),
        }
        if "mu" in sa and "mu" in sb and not {a[0], b[0]} & {"c1", "k2"} and a[0] != b[0]:
            item["wins_a_cont"] = wins_by_dim_cont(sa["mu"], sb["mu"], xbar)
        paired.append(item)
    by_pair = {(p["a"], p["b"]): p for p in paired}
    mb1 = by_pair[("main.median", "b1.median")]

    success_detail = {
        "main_mae": models["main.median"]["mae"],
        "b1_mae": models["b1.median"]["mae"],
        "diff_mae_hi": mb1["hi"]["mae"],
        "wins_a": mb1["wins_a"],
    }
    success = bool(
        success_detail["main_mae"] < success_detail["b1_mae"]
        and success_detail["diff_mae_hi"] < 0
        and success_detail["wins_a"] >= 25
    )
    linear_overturn = bool(models["enet.median"]["mae"] <= models["main.median"]["mae"])
    beaters = [
        p["a"].split(".")[0]
        for p in paired
        if p["b"] == "main.median" and p["a"].split(".")[0] in CTRL_VS_MAIN and p["hi"]["mae"] < 0
    ]
    control_beats_main = bool(beaters)
    leak_flag = bool(leak_dims)

    gm = GR[("main", "median")]
    conf_m = conf_weighted(G, gm)
    mae_direct = float(np.abs(gm - G[None]).mean())
    q_m = qwk_dim(conf_m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q_ref = np.array(
            [
                [
                    cohen_kappa_score(G[:, d], gm[r, :, d], weights="quadratic", labels=range(7))
                    for d in range(G.shape[1])
                ]
                for r in range(gm.shape[0])
            ]
        )
    qwk_ok = bool(
        np.array_equal(np.isnan(q_m), np.isnan(q_ref)) and np.nanmax(np.abs(q_m - q_ref)) <= 1e-10
    )
    pair_direct = max(
        abs(
            p["point"]["mae"]
            - (np.abs(sa["g"] - G[None]).mean() - np.abs(sb["g"] - G[None]).mean())
        )
        for p, (sa, sb) in zip(paired, sides)
    )
    _, lo2, hi2 = run_pair(G, *sides[0], xbar, bootstrap_weights(date_group, n_boot=n_boot))
    ci_same = bool(
        np.array_equal(lo2, res[0][1], equal_nan=True)
        and np.array_equal(hi2, res[0][2], equal_nan=True)
    )
    b0_same = bool((GR[("b0", "const")] == sc["grade_const"]).all())

    out_json = {
        "models": models,
        "climatology": climatology,
        "delivered": delivered,
        "confusion": confusion,
        "grade_freq": freq,
        "reference_error": ref,
        "paired": paired,
        "success": success,
        "success_detail": success_detail,
        "linear_overturn": linear_overturn,
        "control_beats_main": control_beats_main,
        "beaters": beaters,
        "leak_flag": leak_flag,
        "leak_dims": leak_dims,
        "r2_main": r2_main.tolist(),
        "reliability": reliability.tolist(),
        "seed": SEED,
        "meta": json.loads(run_meta()),
    }
    out_json = sanitize(out_json)
    for p in out_json["paired"]:
        for part in ("point", "lo", "hi"):
            for k in PAIR_STATS:
                if p[part][k] is None:
                    p[part][f"{k}_reason"] = REASONS["pair"]

    rj = out_json
    sd = rj["success_detail"]
    redo_success = sd["main_mae"] < sd["b1_mae"] and sd["diff_mae_hi"] < 0 and sd["wins_a"] >= 25
    redo_lin = rj["models"]["enet.median"]["mae"] <= rj["models"]["main.median"]["mae"]
    redo_beat = any(
        p["b"] == "main.median"
        and p["a"].split(".")[0] in CTRL_VS_MAIN
        and p["hi"]["mae"] is not None
        and p["hi"]["mae"] < 0
        for p in rj["paired"]
    )
    redo_leak = any(r > s for r, s in zip(rj["r2_main"], rj["reliability"]))
    decisions_ok = (redo_success, redo_lin, redo_beat, redo_leak) == (
        success,
        linear_overturn,
        control_beats_main,
        leak_flag,
    )

    checks = [
        make_check("所有等级数组形状为 (5, 250, 50)，取值在 0..6", grades_ok, True),
        make_check("所有概率数组每行和为 1", probs_ok, 0.0, 1e-12),
        make_check("每个 (模型, 规则) 的混淆矩阵每次重复总和为 12500", conf_sums_ok, True),
        make_check(
            "混淆矩阵 MAE 与逐单元直接计算一致（main.median）",
            abs(float(mae_dim(conf_m).mean()) - mae_direct),
            0.0,
            1e-12,
        ),
        make_check("逐维 QWK 与 cohen_kappa_score 一致（main.median）", qwk_ok, True),
        make_check("pair_stat_fn 全 1 权重点估计与直接计算一致", float(pair_direct), 0.0, 1e-12),
        make_check("同一种子重复计算 main - b1 的 CI 完全相同", ci_same, True),
        make_check("b0 与 grade_const 在规则 const 下逐元素相同", b0_same, True),
        make_check("判定布尔值由 JSON 中的数值重新推出，与记录一致", decisions_ok, True),
    ]
    mm = models["main.median"]

    def pt(a, b, k):
        return by_pair[(a, b)]["point"][k]

    if not args.smoke:
        rec = confusion["main_median"]["recall"]
        zh_const = models["main.const"]["zero_heavy"]["exact"]
        zh_round = models["main.round"]["zero_heavy"]["exact"]
        checks += [
            make_check("main.median 等级 MAE", mm["mae"], 0.484, 0.015),
            make_check("main.median 完全一致率", mm["exact"], 0.574, 0.015),
            make_check("main.median QWK", mm["qwk"], 0.309, 0.02),
            make_check("main.median 平衡准确率", mm["balanced_acc"], 0.351, 0.03),
            make_check("main.median 5–6 级占比 ≤ 0.001", mm["share56"] <= 0.001, True),
            make_check("main.median 的 RPS", mm["rps"], 0.343, 0.02),
            make_check("main.median 的对数损失", mm["logloss"], 0.986, 0.05),
            make_check("等级频率气候学基线的对数损失", climatology["logloss"], 1.064, 0.03),
            make_check(
                "主模型对数损失低于气候学基线", mm["logloss"] < climatology["logloss"], True
            ),
            make_check("main.const 等级 MAE", models["main.const"]["mae"], 0.550, 0.015),
            make_check("main.round 等级 MAE", models["main.round"]["mae"], 0.495, 0.015),
            make_check("main.quant QWK", models["main.quant"]["qwk"], 0.370, 0.02),
            make_check("main.quant 5–6 级占比", models["main.quant"]["share56"], 0.0146, 0.005),
            make_check("混淆矩阵 main.median 的 0 级召回率", rec[0], 0.744, 0.04),
            make_check("混淆矩阵 main.median 的 5 级召回率 ≤ 0.05", rec[5] <= 0.05, True),
            make_check(
                "median - round 的 mae 点估计",
                pt("main.median", "main.round", "mae"),
                -0.011,
                0.01,
            ),
            make_check(
                "quant - round 的 mae 点估计", pt("main.quant", "main.round", "mae"), 0.076, 0.02
            ),
            make_check(
                "quant - round 的 qwk 点估计", pt("main.quant", "main.round", "qwk"), 0.059, 0.02
            ),
            make_check(
                "median_knn - median 的 qwk 点估计为负",
                pt("main.median_knn", "main.median", "qwk") < 0,
                True,
            ),
            make_check(
                "k2 - main 的 rps 点估计", pt("k2.median", "main.median", "rps"), 0.027, 0.02
            ),
            make_check(
                "k2 - main 的 logloss 点估计为正",
                pt("k2.median", "main.median", "logloss") > 0,
                True,
            ),
            make_check("main 连续 MAE", mm["cont_mae"], 0.479, 0.02),
            make_check("main 连续 R²", mm["cont_r2"], 0.167, 0.04),
            make_check("main 连续 Spearman", mm["cont_spearman"], 0.368, 0.04),
            make_check("b1 连续 MAE", models["b1.median"]["cont_mae"], 0.527, 0.02),
            make_check("main - b1 的 cont_mae 点估计", mb1["point"]["cont_mae"], -0.047, 0.02),
            make_check("main 对 b1 的连续 MAE 逐维胜出维度数", mb1["wins_a_cont"], 38, 6),
            make_check(
                "22 个零分为主维度上 main.const 完全一致率高于 main.round",
                zh_const > zh_round,
                True,
            ),
            make_check("90% 区间覆盖率", cov, 0.969, 0.03),
            make_check("90% 区间平均宽度", width, 1.95, 0.10),
            make_check("leak_flag", leak_flag, False),
        ]
    out_json["checks"] = checks
    out_json["wall_time_s"] = time.time() - t0
    checks.append(
        make_check(
            "JSON 中除 _reason 字段外没有字符串形式的数值或状态",
            not has_string_numbers({k: v for k, v in out_json.items() if k != "checks"}),
            True,
        )
    )

    with open(out_dir / "q1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=1, allow_nan=False)

    arrays = {"dims": np.array(dims)}
    for (m, rule), g in GR.items():
        conf = conf_weighted(G, g)
        arrays[f"{m}__{rule}__mae_dim"] = mae_dim(conf).mean(0)
        arrays[f"{m}__{rule}__exact_dim"] = exact_dim(conf).mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            arrays[f"{m}__{rule}__qwk_dim"] = np.nanmean(qwk_dim(conf), axis=0)
    arrays["r2_dim"] = r2_main
    arrays["reliability"] = reliability
    for k, src in (
        ("main_median", "main_median"),
        ("k2_median", "k2_median"),
        ("c1_sel", "c1_sel"),
    ):
        arrays[f"conf_{k}"] = np.array(confusion[src]["matrix"])
    np.savez_compressed(out_dir / "q1_metrics.npz", **arrays, seed=SEED, meta=run_meta())

    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    print(
        f"success={success} linear_overturn={linear_overturn} "
        f"control_beats_main={control_beats_main} beaters={beaters} leak_flag={leak_flag}"
    )
    print(f"wall_time_s {out_json['wall_time_s']:.0f}；写出 {out_dir / 'q1_metrics.json'}")
    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
