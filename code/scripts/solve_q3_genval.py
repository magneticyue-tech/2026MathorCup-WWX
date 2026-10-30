import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import N_BOOT, N_PERM, NAMED_STYLES  # noqa: E402
from src.q3.data import load_chem  # noqa: E402
from src.q3.gen_metrics import (  # noqa: E402
    GEN_NAMES,
    NEAR_DUP,
    REAL_SCALARS,
    SCALARS,
    acceptance,
    auc_from_oof,
    c2st,
    c2st_perm,
    dispersion_ratio,
    module_pairs,
    pvalue,
    real_vs_real,
)
from src.q3.genval import gen_rep, scalar_metrics  # noqa: E402
from src.q3.meta import dump_json, run_meta  # noqa: E402
from src.q3.modules import modules_from_q2  # noqa: E402
from src.q3.paths import CHUNK_DIR, PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402

BOOT_SEED = SEED


def load_select(path):
    z = np.load(path, allow_pickle=False)
    out = {}
    for r, f, kind, k, kap in zip(
        z["outer_rep"], z["outer_fold"], z["chosen_kind"], z["chosen_k"], z["chosen_kappa"]
    ):
        out[(int(r), int(f))] = {"kind": str(kind), "k": int(k), "kappa": float(kap)}
    return out


def prerun_select():
    return {(r, f): {"kind": "famod", "k": 15, "kappa": 2.0} for r in range(5) for f in range(5)}


def _pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def _unpkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def bootstrap_indices(style, n_boot=N_BOOT, seed=BOOT_SEED):
    style = np.asarray(style)
    styles = sorted(set(style))
    rng = np.random.default_rng(seed)
    idx_all = rng.integers(0, len(style), size=(n_boot, len(style)))
    ids, by_style = [], []
    for st in styles:
        b = np.sort(np.where(style == st)[0])
        ids.append(b)
        by_style.append(rng.integers(0, len(b), size=(n_boot, len(b))))
    return idx_all, ids, by_style


# ---------------- stage gen ----------------
def stage_gen(X, S, styles, sel, chunk, reps, which, opts, workers, smoke_folds=None):
    def one(rep):
        p = chunk / f"pool_{rep}.pkl"
        if not p.exists():
            _pkl(p, gen_rep(X, S, styles, rep, sel, opts, which, smoke_folds))
        return rep

    Parallel(n_jobs=min(workers, len(reps)), verbose=5)(delayed(one)(r) for r in reps)


# ---------------- stage metrics ----------------
def _c2st_unit(path, pool, S, styles, gi, name, var, n_perm):
    real, gen = pool["real_x"], pool["gen_x"][gi]
    rows = ~np.isnan(real).any(1)
    real, gen, s = real[rows], gen[rows], np.asarray(S)[rows]
    sr = s if var == "cond" else None
    res = c2st(real, gen, sr, styles)
    null = c2st_perm(real, gen, sr, styles, n_perm) if n_perm else np.array([])
    _pkl(path, {"auc": res["auc"], "oof": res["oof"], "y": res["y"], "null": null, "rows": rows})


def _style_unit(path, pool, S, styles, gi, st, n_perm):
    s = np.asarray(S)
    rows = np.where((s == st) & ~np.isnan(pool["real_x"]).any(1))[0]
    k = min(5, len(rows))
    res = c2st(pool["real_x"][rows], pool["gen_x"][gi, rows], n_splits=k)
    null = c2st_perm(pool["real_x"][rows], pool["gen_x"][gi, rows], None, None, n_perm, n_splits=k)
    _pkl(path, {"auc": res["auc"], "oof": res["oof"], "y": res["y"], "rows": rows, "null": null})


def _rr_unit(path, pool, S, styles, rep):
    rows = np.where(~np.isnan(pool["real_x"]).any(1))[0]
    out = real_vs_real(pool["real_x"][rows], np.asarray(S)[rows], styles, seed=SEED + rep)
    out["idx_a"], out["idx_b"] = rows[out["idx_a"]], rows[out["idx_b"]]
    _pkl(path, out)


def stage_metrics(S, styles, chunk, reps, which, workers, n_perm, style_perm, per_style=True):
    jobs = []
    pools = {r: _unpkl(chunk / f"pool_{r}.pkl") for r in reps}
    for r in reps:
        p = chunk / f"rr_{r}.pkl"
        if not p.exists():
            jobs.append(delayed(_rr_unit)(p, pools[r], S, styles, r))
        for gi, name in enumerate(which):
            for var in ("marg", "cond"):
                p = chunk / f"c2st_{r}_{name}_{var}.pkl"
                if not p.exists():
                    jobs.append(delayed(_c2st_unit)(p, pools[r], S, styles, gi, name, var, n_perm))
            if not per_style:
                continue
            for st in styles:
                p = chunk / f"style_{r}_{name}_{styles.index(st):02d}.pkl"
                if not p.exists():
                    npm = style_perm if name == "main" else 0
                    jobs.append(delayed(_style_unit)(p, pools[r], S, styles, gi, st, npm))
    print(f"metrics 单元 {len(jobs)}", flush=True)
    jobs.sort(key=lambda j: 0 if "c2st" in str(j[1][0]) else 1)
    Parallel(n_jobs=workers, verbose=5)(jobs)


# ---------------- stage merge ----------------
def _boot_rep(pool, S, styles, mins, inmod, oof, point, idx_all, which):
    est = scalar_metrics(pool, np.arange(len(S)), S, mins, inmod, oof, which, point)
    boots = []
    for b in range(len(idx_all)):
        boots.append(scalar_metrics(pool, idx_all[b], S, mins, inmod, oof, which))
    return est, boots


def _flatten(m, which):
    out = {f"real.{k}": m["real"][k] for k in REAL_SCALARS if k in m["real"]}
    for name in which:
        for k in SCALARS:
            if k in m[name]:
                out[f"{name}.{k}"] = m[name][k]
    return out


def summarize_interval(est_reps, boot_means):
    est = np.array(est_reps, float)
    bm = np.array(boot_means, float)
    lo, hi = np.nanpercentile(bm, [2.5, 97.5]) if np.isfinite(bm).any() else (np.nan, np.nan)
    return {
        "estimate": float(np.mean(est)),
        "ci95": {"lo": float(lo), "hi": float(hi)},
        "repeat_mean": float(np.mean(est)),
        "repeat_min": float(np.min(est)),
        "repeat_max": float(np.max(est)),
        "bootstrap_unit": "beer_id",
        "n_boot": int(len(bm)),
        "interval_scope": "conditional_on_fitted_models_and_splits",
    }


def per_style_tables(S, styles, chunk, reps, which, pools, ids, by_style, weights):
    """逐风格 C2ST AUC 与风格离散比（主模型）及其 bootstrap。"""
    s = np.asarray(S)
    nb = by_style[0].shape[0]
    res = {}
    for name in which:
        auc_rep = np.full((len(reps), len(styles)), np.nan)
        boot = np.full((len(reps), nb, len(styles)), np.nan)
        null_p = np.full((len(reps), len(styles)), np.nan)
        null_q = np.full((len(reps), len(styles)), np.nan)
        for ri, r in enumerate(reps):
            for si, st in enumerate(styles):
                p = chunk / f"style_{r}_{name}_{si:02d}.pkl"
                if not p.exists():
                    continue
                u = _unpkl(p)
                auc_rep[ri, si] = u["auc"]
                if len(u["null"]):
                    null_p[ri, si] = pvalue(u["auc"], u["null"])
                    null_q[ri, si] = np.quantile(u["null"], 0.95)
                rows = list(u["rows"])
                pos = {v: i for i, v in enumerate(rows)}
                n = len(rows)
                beer_ids = ids[si]
                for b in range(nb):
                    take = [pos[beer_ids[j]] for j in by_style[si][b] if beer_ids[j] in pos]
                    sel = np.r_[take, np.array(take) + n].astype(int)
                    boot[ri, b, si] = auc_from_oof(u["y"][sel], u["oof"][sel])
        bm = np.nanmean(boot, 0)
        res[name] = {
            "auc": np.nanmean(auc_rep, 0),
            "auc_rep": auc_rep,
            "ci_lo": np.nanpercentile(bm, 2.5, axis=0),
            "ci_hi": np.nanpercentile(bm, 97.5, axis=0),
            "perm_p": np.nanmean(null_p, 0),
            "perm_q95": np.nanmean(null_q, 0),
            "macro": float(np.nanmean(np.nanmean(auc_rep, 0))),
            "macro_ci": np.nanpercentile(np.nanmean(bm, 1), [2.5, 97.5]),
            "weighted": float(np.nansum(np.nanmean(auc_rep, 0) * weights)),
            "weighted_ci": np.nanpercentile(np.nansum(bm * weights, 1), [2.5, 97.5]),
            "boot": bm,
        }
    # 离散比
    disp_rep = np.full((len(reps), len(styles)), np.nan)
    dboot = np.full((len(reps), nb, len(styles)), np.nan)
    for ri, r in enumerate(reps):
        v = pools[r]["disp"]
        dr = dispersion_ratio(v, s, styles)
        disp_rep[ri] = [dr.get(st, np.nan) for st in styles]
        for b in range(nb):
            means = np.array([v[ids[si][by_style[si][b]]].mean() for si in range(len(styles))])
            dboot[ri, b] = means / np.nanmedian(means)
    dm = np.nanmean(dboot, 0)
    ratio = np.nanmean(disp_rep, 0)
    res["dispersion"] = {
        "ratio": ratio,
        "ratio_rep": disp_rep,
        "ci_lo": np.nanpercentile(dm, 2.5, axis=0),
        "ci_hi": np.nanpercentile(dm, 97.5, axis=0),
        "flag": np.maximum(ratio, 1 / ratio) > 1.5,
        "macro": float(np.nanmean(ratio)),
        "macro_ci": np.nanpercentile(np.nanmean(dm, 1), [2.5, 97.5]),
        "weighted": float(np.nansum(ratio * weights)),
        "weighted_ci": np.nanpercentile(np.nansum(dm * weights, 1), [2.5, 97.5]),
        "boot": dm,
    }
    return res


def stage_merge(
    X, S, styles, chunk, reps, which, out_npz, out_json, workers, per_style=True, n_boot=N_BOOT
):
    s = np.asarray(S)
    chem_mins = np.nanmin(X, 0)
    mod = modules_from_q2(load_chem().gen_cols)
    inmod = module_pairs(mod)
    idx_all, ids, by_style = bootstrap_indices(s, n_boot)
    weights = np.array([np.mean(s == st) for st in styles])
    pools = {r: _unpkl(chunk / f"pool_{r}.pkl") for r in reps}
    oofs, points, nulls = {}, {}, {}
    for r in reps:
        oof, point = {}, {}
        for name in which:
            for var in ("marg", "cond"):
                u = _unpkl(chunk / f"c2st_{r}_{name}_{var}.pkl")
                assert u["rows"].all(), "池中存在缺失行"
                oof[(name, var)] = (u["oof"], u["y"])
                point[(name, var)] = u["auc"]
                nulls[(r, name, var)] = (u["auc"], u["null"])
        rr = _unpkl(chunk / f"rr_{r}.pkl")
        for var in ("marg", "cond"):
            oof[f"rr_{var}"] = (rr[var]["oof"], rr[var]["y"], rr["idx_a"], rr["idx_b"])
            point[f"rr_{var}"] = rr[var]["auc"]
        oofs[r], points[r] = oof, point
    res = Parallel(n_jobs=min(workers, len(reps)), verbose=5)(
        delayed(_boot_rep)(
            pools[r], s, styles, chem_mins, inmod, oofs[r], points[r], idx_all, which
        )
        for r in reps
    )
    est = [_flatten(e, which) for e, _ in res]
    keys = list(est[0])
    boot = {
        k: np.mean([[_flatten(bb, which)[k] for bb in b] for _, b in res], axis=0) for k in keys
    }
    summary = {k: summarize_interval([e[k] for e in est], boot[k]) for k in keys}

    # 置换检验与阳性对照
    perm = {}
    for name in which:
        for var in ("marg", "cond"):
            items = [nulls[(r, name, var)] for r in reps]
            if len(items[0][1]) == 0:
                continue
            perm[f"{name}.{var}"] = {
                "perm_p": float(np.mean([pvalue(a, n) for a, n in items])),
                "perm_q95": float(np.mean([np.quantile(n, 0.95) for _, n in items])),
                "perm_p_rep": [pvalue(a, n) for a, n in items],
                "detected_frac": float(np.mean([a > np.quantile(n, 0.95) for a, n in items])),
            }
    rep_means = {}
    for name in which:
        rep_means[name] = {k: summary[f"{name}.{k}"]["estimate"] for k in SCALARS}
    for k in REAL_SCALARS:
        rep_means[k] = summary[f"real.{k}"]["estimate"]
    acc = acceptance(rep_means) if {"main", "G0", "G1"} <= set(which) else {}

    ps = (
        per_style_tables(s, styles, chunk, reps, which, pools, ids, by_style, weights)
        if per_style
        else {}
    )
    n_s = np.array([(s == st).sum() for st in styles])
    save = dict(
        styles=np.array(styles),
        gen_names=np.array(which),
        n_style=n_s,
        evidence_limited=n_s <= 5,
        bootstrap_indices_all=idx_all,
        metric_keys=np.array(keys),
        metric_estimate=np.array([summary[k]["estimate"] for k in keys]),
        metric_boot=np.array([boot[k] for k in keys]),
        metric_rep=np.array([[e[k] for e in est] for k in keys]),
        seed=SEED,
        meta=run_meta(),
    )
    for si in range(len(styles)):
        save[f"style_beer_ids_{si:02d}"] = ids[si]
        save[f"bootstrap_indices_style_{si:02d}"] = by_style[si]
    for name, d in ps.items():
        for k, v in d.items():
            save[f"ps_{name}_{k}"] = np.asarray(v)
    np.savez_compressed(out_npz, **save)

    js = {"metrics": summary, "permutation": perm, "acceptance": acc}
    js["acceptance_basis"] = rep_means
    js["near_duplicate_note"] = {
        g: "生成池含训练啤酒的近重复样本，AUC 低于 0.5 与 TRTS 不可比" for g in NEAR_DUP
    }
    js["pure_equals_main"] = [bool(pools[r]["pure_equals_main"].all()) for r in reps]
    if ps:
        js["per_style"] = {
            name: {
                st: {
                    "auc": float(d["auc"][i]),
                    "ci95": [float(d["ci_lo"][i]), float(d["ci_hi"][i])],
                    "perm_p": float(d["perm_p"][i]),
                    "perm_q95": float(d["perm_q95"][i]),
                    "n": int(n_s[i]),
                    "evidence_limited": bool(n_s[i] <= 5),
                }
                for i, st in enumerate(styles)
            }
            for name, d in ps.items()
            if name != "dispersion"
        }
        js["per_style_summary"] = {
            name: {
                "macro": d["macro"],
                "macro_ci95": list(d["macro_ci"]),
                "weighted": d["weighted"],
                "weighted_ci95": list(d["weighted_ci"]),
            }
            for name, d in ps.items()
        }
        dd = ps["dispersion"]
        js["dispersion"] = {
            st: {
                "ratio": float(dd["ratio"][i]),
                "ci95": [float(dd["ci_lo"][i]), float(dd["ci_hi"][i])],
                "flag": bool(dd["flag"][i]),
            }
            for i, st in enumerate(styles)
        }
        js["dispersion_flagged"] = [st for i, st in enumerate(styles) if dd["flag"][i]]
        js["named_styles"] = {st: js["per_style"]["main"][st] for st in NAMED_STYLES}
    dump_json(js, out_json)
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "metrics", "merge", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preset", choices=["final", "prerun"], default="final")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ensure_dirs()
    t0 = time.time()
    chem = load_chem()
    X, S, styles = chem.XG, chem.style, chem.styles
    if args.preset == "prerun":
        sel, reps, n_perm, sperm = prerun_select(), [0, 1], 0, 0
        out_dir, chunk = SMOKE_DIR, SMOKE_DIR / "_chunks" / "genval_prerun"
    elif args.smoke:
        sel, reps, n_perm, sperm = load_select(PROCESSED_DIR / "q3_select.npz"), [0], 10, 10
        out_dir, chunk = SMOKE_DIR, SMOKE_DIR / "_chunks" / "genval"
    else:
        sel, reps, n_perm, sperm = (
            load_select(PROCESSED_DIR / "q3_select.npz"),
            list(range(5)),
            N_PERM,
            N_PERM,
        )
        out_dir, chunk = PROCESSED_DIR, CHUNK_DIR / "genval"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk.mkdir(parents=True, exist_ok=True)
    which = GEN_NAMES
    per_style = args.preset != "prerun"
    if args.stage in ("gen", "all"):
        stage_gen(X, S, styles, sel, chunk, reps, which, {}, args.workers)
    if args.stage in ("metrics", "all"):
        stage_metrics(S, styles, chunk, reps, which, args.workers, n_perm, sperm, per_style)
    if args.stage in ("merge", "all"):
        name = "q3_genval_prerun" if args.preset == "prerun" else "q3_genval"
        js = stage_merge(
            X,
            S,
            styles,
            chunk,
            reps,
            which,
            out_dir / f"{name}.npz",
            out_dir / f"{name}.json",
            args.workers,
            per_style,
            N_BOOT if (args.preset == "final" and not args.smoke) else 20,
        )
        m = js["metrics"]
        for k in (
            "main.auc_marg",
            "main.dcr_ratio",
            "main.trts",
            "main.tstr",
            "main.floor_err",
            "main.floor_row",
            "main.corr_err",
            "main.module_rho",
            "G0.auc_marg",
            "G2.auc_marg",
            "real.trtr",
            "real.real_module_rho",
            "main.auc_cond",
            "main.near_copy",
            "main.wcorr_err",
            "G1.wcorr_err",
            "G1.floor_err",
        ):
            print(k, round(m[k]["estimate"], 4), m[k]["ci95"])
        print("acceptance", js["acceptance"])
        print("wall", round(time.time() - t0))
        dump_json(
            {"wall_time_s": time.time() - t0, "stage": args.stage}, out_dir / f"{name}_time.json"
        )


if __name__ == "__main__":
    main()
