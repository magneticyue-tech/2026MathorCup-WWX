import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import ALPHA_GRID, DEFECT_SENS, SRC_STYLE, TGT_STYLE  # noqa: E402
from src.q3.data import concentration, load_dictionary, load_q2_top3  # noqa: E402
from src.q3.defects import build_defect_spec, violation  # noqa: E402
from src.q3.evaluate import (  # noqa: E402
    CATEGORIES,
    StyleDiscriminator,
    category,
    change_amount,
    n_changed_cols,
    retention,
)
from src.q3.fullmodel import (  # noqa: E402
    VARIANTS,
    fit_full,
    load_final_config,
    variant_data,
    x_filled,
)  # noqa: E402
from src.q3.meta import dump_json, make_check, run_meta  # noqa: E402
from src.q3.modules import modules_from_q2  # noqa: E402
from src.q3.paths import (  # noqa: E402
    CHUNK_DIR,
    PROCESSED_DIR,
    Q1_DIR,
    SEED,
    SMOKE_DIR,
    ensure_dirs,  # noqa: E402
)
from src.q3.q1_bridge import grade_median, load_main_oof, load_q1_bundle, predict_mu  # noqa: E402
from src.q3.sens_search import (  # noqa: E402
    DEFECT_SENS_IDX,
    RULES,
    StylePosterior,
    alpha_fit,
    path_search,
    sens_feasible,
    sens_values,
)
from src.q3.source_input import InputBuilder  # noqa: E402
from src.q3.transfer import shift_raw, translate  # noqa: E402

CTX = {}


def setup(args):
    dic = load_dictionary()
    chem, _ = variant_data(args.variant, dic)
    cfg = load_final_config(PROCESSED_DIR / "q3_select.json")
    gen, comp = fit_full(chem, cfg, args.variant)
    chem_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    cz = np.load(chem_dir / f"q3_transfer_chem{args.chem_suffix}.npz", allow_pickle=False)
    cz = {k: cz[k] for k in cz.files}
    thr = "threshold_median" if "_median" in args.chem_suffix else "threshold_mean"
    spec = build_defect_spec(chem.gen_cols, dic, thr)
    styles = chem.styles
    S = chem.style
    own = np.array([styles.index(s) for s in S])
    assert np.allclose(cz["u_star"][np.arange(len(S)), own], gen.U), "化学层产物与生成器不一致"
    model_dir = Q1_DIR / ("q1_models" if args.q1_models == "main" else "q1_models_drop_sum")
    bundle = load_q1_bundle(model_dir, chem.chem_cols)
    XGf = x_filled(chem, gen)
    disc = StyleDiscriminator(XGf, S)
    prior = np.log(np.array([np.mean(S == s) for s in styles]))
    post = StylePosterior(gen, styles, prior)
    builder = InputBuilder(gen, comp, chem)
    Cinv = np.linalg.inv(gen.C)
    CTX.update(
        dict(
            dic=dic,
            chem=chem,
            cfg=cfg,
            gen=gen,
            comp=comp,
            cz=cz,
            spec=spec,
            styles=styles,
            S=S,
            bundle=bundle,
            XGf=XGf,
            disc=disc,
            post=post,
            builder=builder,
            Cinv=Cinv,
        )
    )
    return CTX


def eval_beer(ctx, i, targets, rule, keep_resid, controls):
    gen, builder, bundle, spec, disc = (
        ctx[k] for k in ("gen", "builder", "bundle", "spec", "disc")
    )
    S, styles, cz, Cinv = ctx["S"], ctx["styles"], ctx["cz"], ctx["Cinv"]
    s0 = S[i]
    sr = builder.build_source(i)
    b = cz["b"][i]
    src_in = builder.source_input231(sr)
    src_grade = sens_values(bundle, src_in, "grade")[0]
    if not keep_resid:
        # 只用岭回归派生综合指标时，α=0 的输入与源输入不同；感官基准取同一口径下的 α=0 输入
        src_in = builder.transfer_input231(sr, sr.u[None], keep_resid=False)
    base = sens_values(bundle, src_in, rule)
    base_grade = sens_values(bundle, src_in, "grade")[0]
    out = {"i": i, "base_grade": base_grade, "src_grade": src_grade, "targets": {}}
    for t in targets:
        if t == s0:
            continue
        ti = styles.index(t)
        u_star = cz["u_star"][i, ti]
        ut = translate(sr.u, gen.mu[s0], gen.mu[t])
        ps = path_search(
            builder,
            bundle,
            sr,
            b,
            spec,
            u_star,
            rule,
            None if base is None else base[0],
            keep_resid,
        )
        af, af_i = alpha_fit(ctx["post"], ps["feasible"], ps["U"], t)
        u_main = ps["U"][ps["alpha_idx"]]
        x_main = ps["X225"][ps["alpha_idx"]]
        x_t0 = gen.mg.inverse(ut[None])[0]
        t0_in = builder.transfer_input231(sr, ut[None], keep_resid)
        g_t0 = sens_values(bundle, t0_in, "grade")[0]
        x_fit = ps["X225"][af_i]
        P = disc.p_target(i, np.vstack([sr.x_src, x_main, x_t0, x_fit]), t)
        r = {
            "alpha_star": ps["alpha_star"],
            "alpha_fit": af,
            "feasible": ps["feasible"],
            "ok_sens": ps["ok_sens"],
            "ok_chem": ps["ok_chem_orig"],
            "discontinuous": ps["discontinuous"],
            "n_recheck_fail": ps["n_recheck_fail"],
            "first_infeasible": ps["first_infeasible"],
            "violated_dims": ps["violated_dims"],
            "p_src": float(P[0]),
            "p_main": float(P[1]),
            "p_t0": float(P[2]),
            "p_fit": float(P[3]),
            "t0_chem_viol": bool(cz["viol_t0"][i, ti].any()),
            "t0_sens_viol": bool((g_t0 > base_grade).any()),
            "n_active": int(cz["active"][i, ti].sum()),
            "d2_main": float(gen.mahal2(u_main, t)[0]),
            "d2_src": float(gen.mahal2(sr.u, t)[0]),
            "d2_t0": float(gen.mahal2(ut, t)[0]),
            "retention": retention(sr.u, gen.mu[s0], u_main, gen.mu[t]),
            "change_mahal": float(change_amount(sr.u, u_main, Cinv)[0]),
            "change_cols": int(n_changed_cols(sr.x_src, x_main)[0]),
            "main_chem_viol": bool(violation(x_main, b, spec).any()),
        }
        if controls and ps["first_infeasible"] >= 0:
            r["x_first"] = ps["X225"][ps["first_infeasible"]]
            r["u_first"] = ps["U"][ps["first_infeasible"]]
        if rule != "none":
            fin = sens_values(
                bundle, builder.transfer_input231(sr, u_main[None], keep_resid), rule
            )[0]
            r["main_sens_viol"] = bool(~sens_feasible(fin[None], base[0], rule)[0])
        else:
            r["main_sens_viol"] = False
        if controls:
            r.update(control_eval(ctx, i, t, sr, b, base_grade, ut, ps, keep_resid))
            r["x_main"], r["u_main"], r["x_t0"] = x_main, u_main, x_t0
            r["grade_main"] = sens_values(
                bundle, builder.transfer_input231(sr, u_main[None], keep_resid), "grade"
            )[0]
            r["mu_main_full"] = predict_mu(
                bundle, builder.transfer_input231(sr, u_main[None], keep_resid)
            )[0]
            r["mu_src_full"] = predict_mu(bundle, src_in)[0]
            r["grade_t0"] = g_t0
            r["path_vals"] = ps["vals"]
        out["targets"][t] = r
    return out


def control_eval(ctx, i, t, sr, b, base_grade, ut, ps, keep_resid):
    gen, builder, bundle, spec, disc, cz = (
        ctx[k] for k in ("gen", "builder", "bundle", "spec", "disc", "cz")
    )
    S, styles, XGf, Cinv = ctx["S"], ctx["styles"], ctx["XGf"], ctx["Cinv"]
    s0 = S[i]
    ti = styles.index(t)
    out = {}
    # T1 截断终点
    u_tr = cz["u_trunc"][i, ti]
    x_t1 = gen.mg.inverse(u_tr[None])[0]
    g_t1 = sens_values(bundle, builder.transfer_input231(sr, u_tr[None], keep_resid), "grade")[0]
    out["t1_chem_viol"] = bool(violation(x_t1, b, spec).any())
    out["t1_sens_viol"] = bool((g_t1 > base_grade).any())
    # T3 原尺度均值平移
    chem = ctx["chem"]
    mu_raw = {st: np.nanmean(chem.XG[S == st], 0) for st in (s0, t)}
    x_t3 = shift_raw(XGf[i], mu_raw[s0], mu_raw[t])
    g_t3 = sens_values(bundle, builder.raw_input231(sr, x_t3), "grade")[0]
    out["t3_chem_viol"] = bool(violation(x_t3, b, spec).any())
    out["t3_sens_viol"] = bool((g_t3 > base_grade).any())
    # T5 截断加感官搜索
    p5 = path_search(builder, bundle, sr, b, spec, u_tr, "grade", base_grade, keep_resid)
    u_t5 = p5["U"][p5["alpha_idx"]]
    x_t5 = p5["X225"][p5["alpha_idx"]]
    out["alpha_t5"] = p5["alpha_star"]
    # T2 检索
    cand = []
    for tb in np.where(S == t)[0]:
        if np.all(XGf[tb][spec.idx] <= b + 1e-12):
            g = grade_median(
                bundle, predict_mu(bundle, chem.X.iloc[[tb]], DEFECT_SENS_IDX), DEFECT_SENS_IDX
            )[0]
            if np.all(g <= base_grade):
                cand.append(tb)
    if cand:
        dd = [float((gen.U[c] - ut) @ Cinv @ (gen.U[c] - ut)) for c in cand]
        t2 = cand[int(np.argmin(dd))]
        out["t2_beer_row"] = int(t2)
        out["p_t2"] = float(disc.p_target(t2, XGf[t2], t)[0])
        out["d2_t2"] = float(gen.mahal2(gen.U[t2], t)[0])
        out["t2_reason"] = ""
        out["t2_n_candidates"] = len(cand)
    else:
        out.update(
            t2_beer_row=-1, p_t2=np.nan, d2_t2=np.nan, t2_reason="no_feasible", t2_n_candidates=0
        )
    P = disc.p_target(i, np.vstack([x_t1, x_t3, x_t5]), t)
    out.update(
        p_t1=float(P[0]),
        p_t3=float(P[1]),
        p_t5=float(P[2]),
        p_t4=None,
        d2_t1=float(gen.mahal2(u_tr, t)[0]),
        d2_t5=float(gen.mahal2(u_t5, t)[0]),
        change_t5=float(change_amount(sr.u, u_t5, Cinv)[0]),
        t5_chem_viol=bool(violation(x_t5, b, spec).any()),
        t5_sens_viol=bool(
            (
                sens_values(bundle, builder.transfer_input231(sr, u_t5[None], keep_resid), "grade")[
                    0
                ]
                > base_grade
            ).any()
        ),
        x_t1=x_t1,
        x_t3=x_t3,
        x_t5=x_t5,
    )
    return out


def run_style(path, ctx, s0, targets, rule, keep_resid, controls_pairs):
    rows = np.where(ctx["S"] == s0)[0]
    res = []
    for i in rows:
        ctl = [t for t in targets if (s0, t) in controls_pairs]
        rest = [t for t in targets if (s0, t) not in controls_pairs]
        r = eval_beer(ctx, i, rest, rule, keep_resid, False)
        if ctl:
            r2 = eval_beer(ctx, i, ctl, rule, keep_resid, True)
            r["targets"].update(r2["targets"])
        res.append(r)
    with open(path, "wb") as f:
        pickle.dump(res, f, protocol=4)


def out_suffix(args):
    s = args.chem_suffix
    if args.rule != "grade":
        s += f"_{args.rule}"
    if args.comp_mode != "resid":
        s += f"_{args.comp_mode}"
    if args.q1_models != "main":
        s += f"_{args.q1_models}"
    if args.variant != "base" and args.variant not in args.chem_suffix:
        s += f"_{args.variant}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=RULES, default="grade")
    ap.add_argument("--chem-suffix", default="")
    ap.add_argument("--variant", choices=VARIANTS, default="base")
    ap.add_argument("--comp-mode", choices=["resid", "ridge"], default="resid")
    ap.add_argument("--q1-models", choices=["main", "drop_sum"], default="main")
    ap.add_argument("--all-controls", action="store_true")
    ap.add_argument("--pils-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ensure_dirs()
    t0 = time.time()
    ctx = setup(args)
    styles = ctx["styles"]
    sfx = out_suffix(args)
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    chunk = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / f"transfer{sfx}"
    chunk.mkdir(parents=True, exist_ok=True)
    keep_resid = args.comp_mode == "resid"
    if args.smoke:
        src_styles, tgt_styles = styles[:2], styles[:3]
        if SRC_STYLE not in src_styles:
            src_styles = list(src_styles) + [SRC_STYLE]
            tgt_styles = list(tgt_styles) + [TGT_STYLE]
    elif args.pils_only:
        src_styles, tgt_styles = [SRC_STYLE], [TGT_STYLE]
    else:
        src_styles, tgt_styles = styles, styles
    ctrl = (
        {(a, c) for a in src_styles for c in tgt_styles if a != c} if args.all_controls else set()
    )
    if SRC_STYLE in src_styles and TGT_STYLE in tgt_styles:
        ctrl.add((SRC_STYLE, TGT_STYLE))
    jobs = []
    for s0 in src_styles:
        p = chunk / f"src_{styles.index(s0):02d}.pkl"
        if not p.exists():
            jobs.append(delayed(run_style)(p, ctx, s0, tgt_styles, args.rule, keep_resid, ctrl))
    print(f"待计算源风格 {len(jobs)}", flush=True)
    Parallel(n_jobs=args.workers, verbose=5)(jobs)
    res = []
    for s0 in src_styles:
        with open(chunk / f"src_{styles.index(s0):02d}.pkl", "rb") as f:
            res += pickle.load(f)
    merge(ctx, args, res, src_styles, tgt_styles, out_dir, sfx, t0)


def merge(ctx, args, res, src_styles, tgt_styles, out_dir, sfx, t0):
    styles, S, disc, chem, cz, spec, dic = (
        ctx[k] for k in ("styles", "S", "disc", "chem", "cz", "spec", "dic")
    )
    n, T = len(S), len(styles)
    ref = {t: disc.ref_q25(t) for t in tgt_styles}
    keys_f = [
        "alpha_star",
        "alpha_fit",
        "p_src",
        "p_main",
        "p_t0",
        "p_fit",
        "d2_main",
        "d2_src",
        "d2_t0",
        "retention",
        "change_mahal",
        "change_cols",
        "n_active",
        "n_recheck_fail",
    ]
    keys_b = ["t0_chem_viol", "t0_sens_viol", "discontinuous", "main_chem_viol", "main_sens_viol"]
    A = {k: np.full((n, T), np.nan) for k in keys_f + keys_b}
    cat = np.full((n, T), "", dtype=object)
    cat_fit = np.full((n, T), "", dtype=object)
    base_grade = np.full((n, 6), -1)
    src_grade = np.full((n, 6), -1)
    by_i = {}
    for r in res:
        i = r["i"]
        by_i[i] = r
        base_grade[i] = r["base_grade"]
        src_grade[i] = r.get("src_grade", r["base_grade"])
        for t, v in r["targets"].items():
            ti = styles.index(t)
            for k in keys_f + keys_b:
                A[k][i, ti] = float(v[k])
            cat[i, ti] = category(v["p_main"], v["p_src"], v["alpha_star"], ref[t])
            cat_fit[i, ti] = category(v["p_fit"], v["p_src"], v["alpha_fit"], ref[t])
    src_missing = cz["src_missing_defect"].any(1)
    pairs = []
    for s0 in src_styles:
        rows = np.where(S == s0)[0]
        for t in tgt_styles:
            if t == s0:
                continue
            ti = styles.index(t)
            a = A["alpha_star"][rows, ti]
            c = cat[rows, ti]
            pairs.append(
                {
                    "src": s0,
                    "tgt": t,
                    "n_src": len(rows),
                    "alpha_star_mean": a.mean(),
                    "alpha_star_median": np.median(a),
                    "alpha_star_q25": np.quantile(a, 0.25),
                    "alpha_star_q75": np.quantile(a, 0.75),
                    "frac_alpha1": np.mean(a == 1),
                    "frac_alpha0": np.mean(a == 0),
                    "frac_t0_chem_viol": A["t0_chem_viol"][rows, ti].mean(),
                    "frac_t0_sens_viol": A["t0_sens_viol"][rows, ti].mean(),
                    "frac_constraint_active": np.mean(
                        (A["t0_chem_viol"][rows, ti] > 0) | (A["t0_sens_viol"][rows, ti] > 0)
                    ),
                    "p_main_median": np.median(A["p_main"][rows, ti]),
                    "p_t4_median": np.median(A["p_src"][rows, ti]),
                    "p_fit_median": np.median(A["p_fit"][rows, ti]),
                    "alpha_fit_median": np.median(A["alpha_fit"][rows, ti]),
                    "frac_ref": np.mean(c == "reach"),
                    "frac_improved": np.mean(c == "improved"),
                    "frac_unchanged": np.mean(c == "kept"),
                    "frac_no_improve": np.mean(c == "not_improved"),
                    "frac_discontinuous": A["discontinuous"][rows, ti].mean(),
                    "n_src_missing_defect": int(src_missing[rows].sum()),
                    "ref_q25": ref[t],
                }
            )
    pdf = pd.DataFrame(pairs)
    pdf.to_csv(out_dir / f"q3_transfer_pairs{sfx}.csv", index=False, encoding="utf-8-sig")

    checks = []
    done = ~np.isnan(A["alpha_star"])
    checks.append(
        make_check("α* 全在网格上", bool(np.isin(A["alpha_star"][done], ALPHA_GRID).all()), True)
    )
    checks.append(make_check("主方法原尺度化学违约数", int(np.nansum(A["main_chem_viol"])), 0))
    checks.append(make_check("主方法原尺度感官违约数", int(np.nansum(A["main_sens_viol"])), 0))
    checks.append(
        make_check(
            "风格对数",
            len(pdf),
            len(src_styles) * len(tgt_styles) - len(set(src_styles) & set(tgt_styles)),
        )
    )
    four = pdf[["frac_ref", "frac_improved", "frac_unchanged", "frac_no_improve"]].sum(1)
    checks.append(make_check("四类比例之和为 1", float(np.abs(four - 1).max()), 0.0, 1e-9))
    bundle = ctx["bundle"]
    raw_grade = grade_median(bundle, predict_mu(bundle, chem.X, DEFECT_SENS_IDX), DEFECT_SENS_IDX)
    got = base_grade[:, 0] >= 0
    n_diff = int((raw_grade[got] != base_grade[got]).any(1).sum())
    extra = {
        "n_beers_baseline_differs_from_raw_input": n_diff,
        "n_recheck_fail_total": int(np.nansum(A["n_recheck_fail"])),
        "n_beers_baseline_differs_from_source_input": int(
            (base_grade[got] != src_grade[got]).any(1).sum()
        ),
        "n_discontinuous": int(np.nansum(A["discontinuous"])),
        "pair_summary": {
            "alpha_star_mean_median": float(pdf["alpha_star_mean"].median()),
            "frac_t0_chem_viol_median": float(pdf["frac_t0_chem_viol"].median()),
            "p_main_median_median": float(pdf["p_main_median"].median()),
            "frac_alpha0_mean": float(pdf["frac_alpha0"].mean()),
            "frac_constraint_active_mean": float(pdf["frac_constraint_active"].mean()),
        },
    }
    # 迁移中综合指标重构误差
    builder = ctx["builder"]
    rc = np.array([builder.resid_comp(i) for i in range(n)])
    extra["resid_comp"] = {
        c: {
            "median_abs": float(np.median(np.abs(rc[:, k]))),
            "max_abs": float(np.abs(rc[:, k]).max()),
        }
        for k, c in enumerate(chem.composite_cols)
    }
    if SRC_STYLE in src_styles and TGT_STYLE in tgt_styles:
        extra.update(pils_reports(ctx, by_i, A, cat, cat_fit, ref, out_dir, sfx, checks))
    np.savez_compressed(
        out_dir / f"q3_transfer{sfx}.npz",
        styles=np.array(styles),
        categories=np.array(CATEGORIES),
        category=cat.astype(str),
        category_fit=cat_fit.astype(str),
        base_grade=base_grade,
        ref_q25=np.array([ref.get(t, np.nan) for t in styles]),
        **{k: v for k, v in A.items()},
        config=json.dumps(
            {
                **ctx["cfg"],
                "rule": args.rule,
                "chem_suffix": args.chem_suffix,
                "variant": args.variant,
                "comp_mode": args.comp_mode,
                "q1_models": args.q1_models,
            }
        ),
        seed=SEED,
        meta=run_meta(),
    )
    extra["wall_time_s"] = time.time() - t0
    dump_json({"checks": checks, **extra}, out_dir / f"q3_transfer{sfx}.json")
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["name"], c["observed"], c["expected"])
    print(
        json.dumps(extra.get("pair_summary"), ensure_ascii=False), f"wall {time.time() - t0:.0f}s"
    )


def top3_table():
    return load_q2_top3(DEFECT_SENS)


def pils_reports(ctx, by_i, A, cat, cat_fit, ref, out_dir, sfx, checks):
    styles, S, chem, cz, spec, dic, gen = (
        ctx[k] for k in ("styles", "S", "chem", "cz", "spec", "dic", "gen")
    )
    ti = styles.index(TGT_STYLE)
    src = np.where(S == SRC_STYLE)[0]
    na = np.isnan(chem.XG)
    mods = modules_from_q2(chem.gen_cols)
    top3 = top3_table()
    rows, drows, crows, prow = [], [], [], []
    oof = load_main_oof()
    tgt_rows = np.where(S == TGT_STYLE)[0]
    real_tgt = np.median(oof["oof_mu"].mean(0)[tgt_rows], 0)
    sd = np.sqrt(np.diag(gen.C))
    builder = ctx["builder"]
    bundle = ctx["bundle"]
    t4_viol = 0
    for i in src:
        v = by_i[i]["targets"][TGT_STYLE]
        sr = builder.build_source(i)
        b = cz["b"][i]
        miss_d = bool(na[i, chem.gen_cols.index("X2.3.butanedione")])
        fr = ""
        if cat[i, ti] in ("kept", "not_improved"):
            fr = failure_reason(v, sr, gen, chem, top3)
        rows.append(
            {
                "beer": int(chem.beer[i]),
                "row_index": int(i),
                "alpha_star": v["alpha_star"],
                "alpha_fit": v["alpha_fit"],
                "n_active": v["n_active"],
                "active_cols": "|".join(np.array(spec.cols)[cz["active"][i, ti]]),
                "binding_sens_dims": "|".join(v["violated_dims"]),
                "feasible_path": "".join("1" if f else "0" for f in v["feasible"]),
                "discontinuous": v["discontinuous"],
                "n_recheck_fail": v["n_recheck_fail"],
                "category": cat[i, ti],
                "category_alpha_fit": cat_fit[i, ti],
                "failure_reason": fr,
                "p_src": v["p_src"],
                "p_t0": v["p_t0"],
                "p_t1": v["p_t1"],
                "p_t3": v["p_t3"],
                "p_main": v["p_main"],
                "p_fit": v["p_fit"],
                "p_t4": v["p_src"],
                "p_t5": v["p_t5"],
                "p_t2": v["p_t2"],
                "t2_reason": v["t2_reason"],
                "t2_n_candidates": v["t2_n_candidates"],
                "alpha_t5": v["alpha_t5"],
                "d2_src": v["d2_src"],
                "d2_t0": v["d2_t0"],
                "d2_t1": v["d2_t1"],
                "d2_main": v["d2_main"],
                "d2_t5": v["d2_t5"],
                "d2_t2": v["d2_t2"],
                "retention": v["retention"],
                "change_mahal": v["change_mahal"],
                "change_cols": v["change_cols"],
                "change_mahal_t5": v["change_t5"],
                "t0_chem_viol": v["t0_chem_viol"],
                "t0_sens_viol": v["t0_sens_viol"],
                "t1_chem_viol": v["t1_chem_viol"],
                "t1_sens_viol": v["t1_sens_viol"],
                "t3_chem_viol": v["t3_chem_viol"],
                "t3_sens_viol": v["t3_sens_viol"],
                "t5_chem_viol": v["t5_chem_viol"],
                "t5_sens_viol": v["t5_sens_viol"],
                "main_chem_viol": v["main_chem_viol"],
                "main_sens_viol": v["main_sens_viol"],
                "src_missing_diacetyl": miss_d,
                "chem_fully_decidable": not cz["src_missing_defect"][i].any(),
                "ref_q25_tripel": ref[TGT_STYLE],
            }
        )
        t4_viol += int(violation(sr.x_src, b, spec).any())
        for k, c in enumerate(spec.cols):
            j = spec.idx[k]
            th = spec.theta[k]
            before, after = sr.x_src[j], v["x_main"][j]
            oav_b = oav_a = np.nan
            if spec.has_threshold[k]:
                th_raw = dic.loc[
                    c, "threshold_mean" if "_median" not in sfx else "threshold_median"
                ]
                oav_b = float(concentration(before, c, dic) / th_raw)
                oav_a = float(concentration(after, c, dic) / th_raw)
            drows.append(
                {
                    "beer": int(chem.beer[i]),
                    "layer": "chem",
                    "item": c,
                    "before": before,
                    "after": after,
                    "oav_before": oav_b,
                    "oav_after": oav_a,
                    "bound": b[k],
                    "theta": th,
                    "active": bool(cz["active"][i, ti, k]),
                    "source_missing": bool(cz["src_missing_defect"][i, k]),
                    "recheck_ok": bool(after <= b[k] + 1e-9),
                }
            )
        for k, d in enumerate(DEFECT_SENS):
            drows.append(
                {
                    "beer": int(chem.beer[i]),
                    "layer": "sens",
                    "item": d,
                    "before": float(by_i[i]["base_grade"][k]),
                    "after": float(v["grade_main"][k]),
                    "oav_before": np.nan,
                    "oav_after": np.nan,
                    "bound": float(by_i[i]["base_grade"][k]),
                    "theta": np.nan,
                    "active": d in v["violated_dims"],
                    "source_missing": bool(d == "diacetyl" and miss_d),
                    "recheck_ok": bool(v["grade_main"][k] <= by_i[i]["base_grade"][k]),
                }
            )
        ch = np.abs(v["u_main"] - sr.u) / sd
        for rk, j in enumerate(np.argsort(-ch)[:15]):
            crows.append(
                {
                    "beer": int(chem.beer[i]),
                    "rank": rk + 1,
                    "compound": chem.gen_cols[j],
                    "std_latent_change": ch[j],
                    "before": sr.x_src[j],
                    "after": v["x_main"][j],
                    "q2_module": int(mods[j]),
                }
            )
        for kind, vec in (
            ("before", v["mu_src_full"]),
            ("after", v["mu_main_full"]),
            ("target_real_oof", real_tgt),
        ):
            prow.append({"beer": int(chem.beer[i]), "profile": kind, **dict(zip(bundle.dims, vec))})
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / f"q3_transfer_pils_tripel{sfx}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(drows).to_csv(
        out_dir / f"q3_transfer_defects{sfx}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(crows).to_csv(
        out_dir / f"q3_transfer_top_changes{sfx}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(prow).to_csv(
        out_dir / f"q3_transfer_portraits{sfx}.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        out_dir / f"q3_transfer_pils_paths{sfx}.npz",
        rows=src,
        alphas=ALPHA_GRID,
        feasible=np.array([by_i[i]["targets"][TGT_STYLE]["feasible"] for i in src]),
        ok_sens=np.array([by_i[i]["targets"][TGT_STYLE]["ok_sens"] for i in src]),
        ok_chem=np.array([by_i[i]["targets"][TGT_STYLE]["ok_chem"] for i in src]),
        alpha_star=tab["alpha_star"].to_numpy(),
        alpha_fit=tab["alpha_fit"].to_numpy(),
        seed=SEED,
        meta=run_meta(),
    )
    full = tab["chem_fully_decidable"]
    checks.append(make_check("T4 两层违约率为 0", t4_viol, 0))
    checks.append(
        make_check(
            "T5 原尺度两层违约为 0", int((tab["t5_chem_viol"] | tab["t5_sens_viol"]).sum()), 0
        )
    )
    checks.append(make_check("化学层可完全判定款数", int(full.sum()), 9))
    viol_rates = {}
    for m in ("t0", "t1", "t3", "t5", "main"):
        both = tab[f"{m}_chem_viol"] | tab[f"{m}_sens_viol"]
        viol_rates[m] = {
            "chem_all": float(tab[f"{m}_chem_viol"].mean()),
            "sens_all": float(tab[f"{m}_sens_viol"].mean()),
            "chem_decidable": float(tab.loc[full, f"{m}_chem_viol"].mean()),
            "sens_decidable": float(tab.loc[full, f"{m}_sens_viol"].mean()),
            "any_all": float(both.mean()),
        }
    summ = {
        c: float(tab[c].median())
        for c in (
            "p_src",
            "p_t0",
            "p_t1",
            "p_main",
            "p_fit",
            "p_t5",
            "p_t2",
            "d2_src",
            "d2_t0",
            "d2_main",
            "d2_t5",
            "retention",
            "alpha_star",
            "alpha_fit",
            "n_active",
        )
    }
    p_real = ctx["disc"].p_real_oof(TGT_STYLE)
    summ["p_real_tripel_median"] = float(np.median(p_real))
    summ["d2_real_tripel_median"] = float(np.median(gen.mahal2(gen.U[tgt_rows], TGT_STYLE)))
    return {
        "pils_tripel": {
            "summary_median": summ,
            "violation_rates": viol_rates,
            "category_counts": tab["category"].value_counts().to_dict(),
            "n_alpha1": int((tab["alpha_star"] == 1).sum()),
            "n_alpha0": int((tab["alpha_star"] == 0).sum()),
            "n_discontinuous": int(tab["discontinuous"].sum()),
            "src_missing_beers": tab.loc[tab["src_missing_diacetyl"], "beer"].tolist(),
        }
    }


def failure_reason(v, sr, gen, chem, top3):
    k = v["first_infeasible"]
    if k < 0:
        return "路径全程可行；P_t 未达到改善阈值"
    dims = v["violated_dims"]
    parts = []
    if not v["ok_chem"][k]:
        parts.append("原尺度化学复查不通过")
    def seg(c, tag=""):
        if c not in chem.gen_cols:
            return f"{c}{tag}(非生成列)"
        j = chem.gen_cols.index(c)
        du = v["u_first"][j] - sr.u[j]
        return f"{c}{tag}:{sr.x_src[j]:.4g}→{v['x_first'][j]:.4g}(Δu={du:+.3f})"

    for d in dims:
        t = top3.get(d)
        if t is None or t["merged"] is None:
            parts.append(f"{d}: NaN（{t['reason'] if t else 'Q2 冻结产物中没有该维度'}）")
            continue
        merged = [seg(c, f"[{s}{',共变' if cov else ''}]") for c, s, cov in t["merged"]]
        within = [seg(c) for c in t["within"]]
        parts.append(f"{d} 合并Top3[" + ";".join(merged) + "] 风格内Top3[" + ";".join(within) + "]")
    return " / ".join(parts) if parts else "首个不可行点无感官维度违约"


if __name__ == "__main__":
    main()
