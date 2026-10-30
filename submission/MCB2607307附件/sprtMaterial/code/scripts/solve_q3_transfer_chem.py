import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import SRC_STYLE, TGT_STYLE  # noqa: E402
from src.q3.data import is_linear, load_dictionary  # noqa: E402
from src.q3.defects import LatentBounder, build_defect_spec, chem_bounds, violation  # noqa: E402
from src.q3.fullmodel import (  # noqa: E402
    PRERUN_CONFIG,
    VARIANTS,
    fit_full,
    load_final_config,
    variant_data,
    x_filled,
)
from src.q3.meta import dump_json, make_check, run_meta  # noqa: E402
from src.q3.paths import PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q3.transfer import project, shift_raw, translate, truncate  # noqa: E402


def suffix_of(rule, thr_kind, variant):
    s = ""
    if rule == "strict":
        s += "_strict"
    if thr_kind == "median":
        s += "_median"
    if variant != "base":
        s += f"_{variant}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["prerun", "final"], default="final")
    ap.add_argument("--rule", choices=["threshold", "strict"], default="threshold")
    ap.add_argument("--thr-kind", choices=["mean", "median"], default="mean")
    ap.add_argument("--variant", choices=VARIANTS, default="base")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    dic = load_dictionary()
    chem, extra = variant_data(args.variant, dic)
    if args.config == "final":
        cfg = load_final_config(PROCESSED_DIR / "q3_select.json")
    else:
        cfg = PRERUN_CONFIG
    gen, _ = fit_full(chem, cfg, args.variant)
    S, styles = chem.style, chem.styles
    U = gen.U
    x_src = gen.mg.inverse(U)
    XGf = x_filled(chem, gen)
    na = np.isnan(chem.XG)
    spec = build_defect_spec(chem.gen_cols, dic, f"threshold_{args.thr_kind}")
    bounder = LatentBounder(gen.mg, spec)
    n, K, T = len(S), len(spec.cols), len(styles)
    b = np.array([chem_bounds(x_src[i], spec, args.rule) for i in range(n)])
    beta = np.array([bounder.beta(b[i], U[i]) for i in range(n)])
    src_missing = na[:, spec.idx]

    src_styles = styles[:2] if args.smoke else styles
    tgt_styles = styles[:3] if args.smoke else styles
    active = np.zeros((n, T, K), bool)
    viol = np.zeros((n, T, K), bool)
    lam_all = np.zeros((n, T, K))
    u_star = np.repeat(U[:, None, :], T, axis=1)
    u_trunc = np.repeat(U[:, None, :], T, axis=1)
    pairs = []
    for s0 in src_styles:
        rows = np.where(S == s0)[0]
        for s1 in tgt_styles:
            if s0 == s1:
                continue
            ti = styles.index(s1)
            for i in rows:
                ut = translate(U[i], gen.mu[s0], gen.mu[s1])
                viol[i, ti] = violation(gen.mg.inverse(ut[None])[0], b[i], spec)
                us, act, lam = project(ut, beta[i], gen.C, spec.idx)
                u_star[i, ti], active[i, ti], lam_all[i, ti] = us, act, lam
                u_trunc[i, ti] = truncate(ut, beta[i], spec.idx)
            pairs.append(
                (
                    s0,
                    s1,
                    len(rows),
                    float(viol[rows, ti].any(1).mean()),
                    float(active[rows, ti].sum(1).mean()),
                    int(src_missing[rows].any(1).sum()),
                )
            )

    checks = []
    res = {}
    if SRC_STYLE in src_styles and TGT_STYLE in tgt_styles:
        ti = styles.index(TGT_STYLE)
        src = np.where(S == SRC_STYLE)[0]
        ut = np.array([translate(U[i], gen.mu[SRC_STYLE], gen.mu[TGT_STYLE]) for i in src])
        vt = viol[src, ti]
        nact = active[src, ti].sum(1)
        res["pils_viol_count"] = dict(zip(spec.cols, vt.sum(0).tolist()))
        res["pils_n_active"] = nact.tolist()
        mu_raw = {st: np.nanmean(chem.XG[S == st], 0) for st in styles}
        raw = shift_raw(XGf[src], mu_raw[SRC_STYLE], mu_raw[TGT_STYLE])
        lo, hi = np.nanmin(chem.XG, 0), np.nanmax(chem.XG, 0)
        mins = XGf.min(0)
        tied = (XGf == mins).sum(0) >= 2
        lin = np.array([is_linear(c, dic) for c in chem.gen_cols])
        t3 = {
            "n_cells": int(raw.size),
            "out_of_range": int(((raw < lo) | (raw > hi)).sum()),
            "below_floor": int((raw[:, tied] < mins[tied] - 1e-12).sum()),
            "linear_negative": int((raw[:, lin] < 0).sum()),
        }
        res["t3"] = t3
        np.savez_compressed(
            out_dir
            / f"q3_transfer_chem_pils{suffix_of(args.rule, args.thr_kind, args.variant)}.npz",
            src_rows=src,
            u_tilde=ut,
            u_star=u_star[src, ti],
            u_trunc=u_trunc[src, ti],
            x_t0=gen.mg.inverse(ut),
            x_proj=gen.mg.inverse(u_star[src, ti]),
            x_trunc=gen.mg.inverse(u_trunc[src, ti]),
            x_t3=raw,
            b=b[src],
            beta=beta[src],
            active=active[src, ti],
            lam=lam_all[src, ti],
            viol_t0=vt,
            seed=SEED,
            meta=run_meta(),
        )
        if args.rule == "threshold" and args.thr_kind == "mean" and args.variant == "base":
            vc = res["pils_viol_count"]
            checks += [
                make_check("Pils/Lager 款数", len(src), 11),
                make_check(
                    "T0 违反 H2S/MeSH/E-2-nonenal",
                    [vc["H2S"], vc["MeSH"], vc["X.E..2.nonenal"]],
                    [11, 11, 11],
                ),
                make_check("T0 违反 DMS", vc["DMS"], 10),
                make_check("T0 违反 DES", vc["DES"], 2, 1),
                make_check("T0 违反 双乙酰", vc["X2.3.butanedione"], 3, 2),
                make_check("有效约束数最小", int(nact.min()), 3, 1),
                make_check("有效约束数最大", int(nact.max()), 5, 1),
                make_check("有效约束数中位", float(np.median(nact)), 4, 1),
                make_check("T3 单元格总数", t3["n_cells"], 2475),
                make_check("T3 超出观测范围", t3["out_of_range"], 163),
                make_check("T3 低于下限", t3["below_floor"], 102),
                make_check("T3 线性列负值", t3["linear_negative"], 1),
            ]
    pv = np.array([q[3] for q in pairs])
    pa = np.array([q[4] for q in pairs])
    if not args.smoke and args.rule == "threshold" and args.thr_kind == "mean":
        if args.variant == "base":
            checks += [
                make_check("风格对数", len(pairs), 462),
                make_check("pair_viol_frac 中位", float(np.median(pv)), 1.0, 0.02),
                make_check("pair_viol_frac 最小", float(pv.min()), 0.0, 0.02),
                make_check("pair_viol_frac 最大", float(pv.max()), 1.0, 0.02),
                make_check("pair_mean_active 中位", float(np.median(pa)), 2.0, 0.3),
                make_check("src_missing_defect 个数", int(src_missing.sum()), 9),
                make_check(
                    "src_missing 只在双乙酰",
                    bool(src_missing[:, spec.cols.index("X2.3.butanedione")].sum() == 9),
                    True,
                ),
                make_check(
                    "Pils/Lager 源缺失 beer",
                    sorted(chem.beer[(S == SRC_STYLE) & src_missing.any(1)].tolist()),
                    [28, 119],
                ),
            ]
    x0 = gen.mg.inverse(U)
    zero_ok = not violation(x0, b, spec).any()
    checks.append(make_check("零迁移 α=0 化学精确可行", bool(zero_ok), True))
    checks.append(make_check("beta ≥ u[idx]", bool((beta >= U[:, spec.idx]).all()), True))

    suffix = suffix_of(args.rule, args.thr_kind, args.variant)
    np.savez_compressed(
        out_dir / f"q3_transfer_chem{suffix}.npz",
        styles=np.array(styles),
        spec_cols=np.array(spec.cols),
        spec_idx=spec.idx,
        theta=spec.theta,
        b=b,
        beta=beta,
        active=active,
        viol_t0=viol,
        lam=lam_all,
        u_star=u_star,
        u_trunc=u_trunc,
        src_missing_defect=src_missing,
        pair_src=np.array([q[0] for q in pairs]),
        pair_tgt=np.array([q[1] for q in pairs]),
        pair_n_src=np.array([q[2] for q in pairs]),
        pair_viol_frac=pv,
        pair_mean_active=pa,
        pair_n_src_missing_defect=np.array([q[5] for q in pairs]),
        config=json.dumps(
            {**cfg, "rule": args.rule, "thr_kind": args.thr_kind, "variant": args.variant}
        ),
        seed=SEED,
        meta=run_meta(),
    )
    wall = time.time() - t0
    dump_json(
        {"config": cfg, "checks": checks, "results": res, "wall_time_s": wall},
        out_dir / f"q3_transfer_chem{suffix}.json",
    )
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["name"], c["observed"], c["expected"])
    print(f"wall {wall:.0f}s")


if __name__ == "__main__":
    main()
