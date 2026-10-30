import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1_data import make_check  # noqa: E402
from solve_q1_metrics import has_string_numbers, sanitize  # noqa: E402
from src.q1.baselines import simple_fold  # noqa: E402
from src.q1.data import DIM_GROUPS, DIMS, load_q1  # noqa: E402
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
    r2_by_dim,
    spearman_by_dim,
    within1_dim,
)
from src.q1.model import fit_outer_fold  # noqa: E402
from src.q1.paths import CHUNK_DIR, PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q1.sensitivity import unit_grades  # noqa: E402
from src.q1.splits import leave_one_style_splits, paper_split  # noqa: E402

MODELS = ("main", "b0", "b1")
GKEYS = ("median", "round", "quant")


def load(name):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def run_split(path, X, xbar, dims, groups, date_group, style, tr, te):
    tr, te = np.sort(tr), np.sort(te)
    out = {"train_idx": tr, "test_idx": te}
    u = fit_outer_fold(X, xbar, dims, groups, date_group, style, tr, te, SEED + 100)
    res = {"main": (u["oof"], u["inner"])}
    for kind in ("b0", "b1"):
        s = simple_fold(kind, xbar, style, date_group, tr, te, SEED + 100)
        res[kind] = (s["oof"], s["inner"])
    for m, (oof, inner) in res.items():
        g = unit_grades(oof, inner, xbar[tr])
        out[f"{m}__oof"] = oof
        for k in (*GKEYS, "p"):
            out[f"{m}__{k}"] = g[k]
    out["chosen"] = json.dumps(u["chosen"])
    np.savez_compressed(path, **out)
    return path


def grade_stats(G, g):
    conf = conf_weighted(G, g[None])
    q, n_undef = mean_qwk(qwk_dim(conf))
    return {
        "mae": float(mae_dim(conf).mean()),
        "exact": float(exact_dim(conf).mean()),
        "within1": float(within1_dim(conf).mean()),
        "qwk": q,
        "qwk_undefined": n_undef,
    }


def cont_stats(mu, xbar):
    rho = spearman_by_dim(mu[None], xbar)
    return {
        "cont_mae": float(np.abs(mu - xbar).mean()),
        "cont_r2": float(r2_by_dim(mu[None], xbar).mean()),
        "cont_spearman": float(np.nanmean(rho)) if (~np.isnan(rho)).any() else float("nan"),
        "spearman_const_dims": int(np.isnan(rho).sum()),
    }


def style_spearman(mu, x):
    vals, n_const = [], 0
    for d in range(mu.shape[1]):
        if np.ptp(mu[:, d]) == 0 or np.ptp(x[:, d]) == 0:
            n_const += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals.append(spearmanr(mu[:, d], x[:, d])[0])
    return (float(np.nanmean(vals)) if vals else float("nan")), n_const


def merged_report(G, xbar, arrs, rows, date_group, n_boot, workers):
    rep = {}
    for m in MODELS:
        rep[m] = {
            **grade_stats(G[rows], arrs[m]["median"][rows]),
            **cont_stats(arrs[m]["oof"][rows], xbar[rows]),
        }
    W = bootstrap_weights(date_group[rows], n_boot=n_boot)

    def side(m):
        return {
            "g": arrs[m]["median"][rows][None],
            "p": arrs[m]["p"][rows][None],
            "mu": arrs[m]["oof"][rows][None],
        }

    res = Parallel(n_jobs=min(workers, 2))(
        delayed(cluster_bootstrap)(pair_stat_fn(G[rows], side("main"), side(b), xbar[rows]), W)
        for b in ("b1", "b0")
    )
    for b, (point, lo, hi) in zip(("b1", "b0"), res):
        rep[f"main_minus_{b}"] = {
            "point": dict(zip(PAIR_STATS, point.tolist())),
            "lo": dict(zip(PAIR_STATS, lo.tolist())),
            "hi": dict(zip(PAIR_STATS, hi.tolist())),
        }
    return rep


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
    chunk_dir = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / "stress"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_boot = 20 if args.smoke else 2000
    t0 = time.time()

    z = load("q1_data.npz")
    xbar, G, date_group = z["xbar"], z["G"], z["date_group"]
    sc = load("q1_scores.npz")
    oof_main = load("q1_main_oof.npz")["oof_mu"]
    data = load_q1()
    X, style = data.X, data.style
    s_arr = style.to_numpy()

    dims = DIMS[:3] if args.smoke else DIMS
    groups = {g: [d for d in ds if d in dims] for g, ds in DIM_GROUPS.items()}
    groups = {g: ds for g, ds in groups.items() if ds}
    dj = [DIMS.index(d) for d in dims]
    Xd, Gd = xbar[:, dj], G[:, dj]

    loso = leave_one_style_splits(style)
    disjoint = sorted(np.concatenate([te for _, te, _ in loso]).tolist()) == list(range(len(s_arr)))
    absent = all(name not in set(s_arr[tr]) for tr, _, name in loso)
    order = list(range(len(loso)))
    if args.smoke:
        order = sorted(order, key=lambda i: len(loso[i][1]))[:2]
    p_tr, p_te = paper_split(style, seed=SEED)

    jobs = []
    for i in order:
        tr, te, _ = loso[i]
        path = chunk_dir / f"loso_{i}.npz"
        if not path.exists():
            jobs.append(delayed(run_split)(path, X, Xd, dims, groups, date_group, style, tr, te))
    paper_path = chunk_dir / "paper.npz"
    if not paper_path.exists():
        jobs.append(
            delayed(run_split)(paper_path, X, Xd, dims, groups, date_group, style, p_tr, p_te)
        )
    print(f"待计算划分 {len(jobs)} 个", flush=True)
    if jobs:
        Parallel(n_jobs=args.workers, verbose=10)(jobs)

    n, D = Xd.shape

    def empty():
        return {
            m: {
                "oof": np.full((n, D), np.nan),
                **{k: np.full((n, D), -1, dtype=int) for k in GKEYS},
                "p": np.full((n, D, 7), np.nan),
            }
            for m in MODELS
        }

    def fill(arrs, u):
        te = u["test_idx"]
        for m in MODELS:
            arrs[m]["oof"][te] = u[f"{m}__oof"]
            arrs[m]["p"][te] = u[f"{m}__p"]
            for k in GKEYS:
                arrs[m][k][te] = u[f"{m}__{k}"]

    loso_arr = empty()
    per_style = []
    for i in order:
        u = np.load(chunk_dir / f"loso_{i}.npz", allow_pickle=False)
        fill(loso_arr, u)
        tr, te, name = loso[i]
        te = np.sort(te)
        entry = {"style": name, "n": int(len(te)), "small": bool(len(te) <= 5)}
        for m in MODELS:
            gm = u[f"{m}__median"]
            mu = u[f"{m}__oof"]
            if m == "main":
                rho, n_const = style_spearman(mu, Xd[te])
                entry.update(
                    mae_median=float(np.abs(gm - Gd[te]).mean()),
                    mae_round=float(np.abs(u["main__round"] - Gd[te]).mean()),
                    exact=float((gm == Gd[te]).mean()),
                    within1=float((np.abs(gm - Gd[te]) <= 1).mean()),
                    cont_mae=float(np.abs(mu - Xd[te]).mean()),
                    spearman=rho,
                    spearman_const_dims=n_const,
                )
                if len(te) < 10:
                    entry["qwk"] = None
                    entry["qwk_reason"] = "测试风格少于 10 款，QWK 不稳定，不报告"
                else:
                    entry["qwk"] = mean_qwk(qwk_dim(conf_weighted(Gd[te], gm[None])))[0]
            else:
                entry[f"{m}_mae"] = float(np.abs(gm - Gd[te]).mean())
                entry[f"{m}_cont_mae"] = float(np.abs(mu - Xd[te]).mean())
        per_style.append(entry)

    paper_arr = empty()
    fill(paper_arr, np.load(paper_path, allow_pickle=False))
    p_rows = np.zeros(n, bool)
    p_rows[p_te] = True
    loso_rows = loso_arr["main"]["median"][:, 0] >= 0

    combined = merged_report(Gd, Xd, loso_arr, loso_rows, date_group, n_boot, args.workers)
    paper = merged_report(Gd, Xd, paper_arr, p_rows, date_group, n_boot, args.workers)

    exposure = 0
    for i in order:
        tr, te, _ = loso[i]
        tr_groups = set(date_group[tr])
        exposure += int(sum(date_group[b] in tr_groups for b in te))
    small = [e["style"] for e in per_style if e["small"]]
    small_rows = np.isin(s_arr, small)
    gm_main = sc["grade_median"][:, :, dj]
    main_split_small = {
        "styles": small,
        "n": int(small_rows.sum()),
        "mae": (
            float(np.abs(gm_main[:, small_rows] - Gd[small_rows][None]).mean())
            if small_rows.any()
            else float("nan")
        ),
        "cont_mae": (
            float(np.abs(oof_main[:, small_rows][:, :, dj] - Xd[small_rows][None]).mean())
            if small_rows.any()
            else float("nan")
        ),
    }

    checks = [
        make_check("留一风格：22 个测试集互不相交，并集为 250 款", bool(disjoint), True),
        make_check("留一风格：测试风格在对应训练集中不出现", bool(absent), True),
        make_check("论文划分：训练 / 测试款数", [len(p_tr), len(p_te)], [175, 75]),
    ]
    if not args.smoke:
        checks.append(
            make_check(
                "留一风格合并后主模型 cont_mae 低于 b0 的 cont_mae",
                combined["main"]["cont_mae"] < combined["b0"]["cont_mae"],
                True,
            )
        )
    report = sanitize(
        {
            "per_style": per_style,
            "combined": combined,
            "paper": paper,
            "same_date_group_exposure": exposure,
            "main_split_small_styles": main_split_small,
            "seed": SEED,
            "meta": json.loads(run_meta()),
        }
    )
    checks.append(
        make_check(
            "所有结果 JSON 中没有字符串形式的数值或状态（除 _reason 字段）",
            not has_string_numbers({k: v for k, v in report.items() if k != "per_style"})
            and not any(
                has_string_numbers({k: v for k, v in e.items() if k != "style"})
                for e in report["per_style"]
            ),
            True,
        )
    )
    report["checks"] = checks
    wall = time.time() - t0
    report["wall_time_s"] = wall
    with open(out_dir / "q1_stress.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, allow_nan=False)

    arrays = {}
    for prefix, arrs in (("loso", loso_arr), ("paper", paper_arr)):
        for m in MODELS:
            tag = prefix if m == "main" else f"{prefix}_{m}"
            arrays[f"{tag}_oof_mu"] = arrs[m]["oof"]
            arrays[f"{tag}_p"] = arrs[m]["p"]
            for k in GKEYS:
                arrays[f"{tag}_grade_{k}"] = arrs[m][k]
    np.savez_compressed(
        out_dir / "q1_stress.npz",
        loso_styles=np.array([loso[i][2] for i in order]),
        loso_n=np.array([len(loso[i][1]) for i in order]),
        paper_test_idx=np.sort(p_te),
        **arrays,
        checks=json.dumps(checks, ensure_ascii=False),
        seed=SEED,
        meta=run_meta(),
        wall_time_s=wall,
    )
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    for name, rep in (("留一风格合并", combined), ("论文划分", paper)):
        print(
            name,
            {m: (round(rep[m]["mae"], 4), round(rep[m]["cont_mae"], 4)) for m in MODELS},
            "main-b1 mae",
            round(rep["main_minus_b1"]["point"]["mae"], 4),
            [
                round(rep["main_minus_b1"]["lo"]["mae"], 4),
                round(rep["main_minus_b1"]["hi"]["mae"], 4),
            ],
        )
    print(f"同日期组暴露 {exposure} 款；wall_time_s {wall:.0f}")
    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
