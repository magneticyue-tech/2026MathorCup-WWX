import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import N_SYNTH, NAMED_STYLES  # noqa: E402
from src.q3.data import load_dictionary  # noqa: E402
from src.q3.fullmodel import fit_full, load_final_config, variant_data, x_filled  # noqa: E402
from src.q3.meta import dump_json, run_meta  # noqa: E402
from src.q3.modules import modules_from_q2  # noqa: E402
from src.q3.paths import PROCESSED_DIR, Q1_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q3.portrait import (  # noqa: E402
    chem_fingerprint,
    fingerprint_table,
    portrait_samples,
    sensory_fingerprint,
    summarize,
)
from src.q3.q1_bridge import load_main_oof, load_q1_bundle  # noqa: E402

FP_BOOT = 1000


def weak_dims():
    z = np.load(Q1_DIR / "q1_metrics.npz", allow_pickle=False)
    return z["main__median__mae_dim"] >= z["b1__median__mae_dim"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dic = load_dictionary()
    chem, _ = variant_data("base", dic)
    cfg = load_final_config(PROCESSED_DIR / "q3_select.json")
    gen, comp = fit_full(chem, cfg)
    bundle = load_q1_bundle(chem_cols=chem.chem_cols)
    istd = float(chem.X[chem.internal_std_col].median())
    cols = (chem.chem_cols, chem.gen_cols, chem.composite_cols, chem.internal_std_col)
    styles = chem.styles
    n = 50 if args.smoke else N_SYNTH
    weak = weak_dims()
    mods = modules_from_q2(chem.gen_cols)
    XGf = x_filled(chem, gen)

    S = len(styles)
    D = len(bundle.dims)
    mu_all = np.zeros((S, n, D), np.float32)
    synth = np.zeros((S, n, len(chem.gen_cols)), np.float32)
    summ = {k: [] for k in ("mu_q10", "mu_q50", "mu_q90", "pi", "pi_tilde", "mode")}
    delta, dlo, dhi = np.zeros((S, 225)), np.zeros((S, 225)), np.zeros((S, 225))
    fp_tabs = []
    mu_by = {}
    for si, st in enumerate(styles):
        rng = np.random.default_rng([SEED, si])
        r = portrait_samples(gen, comp, bundle, st, n, rng, cols, istd)
        mu_all[si] = r["mu"]
        synth[si] = r["x"][:, [chem.chem_cols.index(c) for c in chem.gen_cols]]
        mu_by[st] = r["mu"]
        sm = summarize(r["mu"], r["G"], r["p"])
        for k in summ:
            summ[k].append(sm[k])
        fp = chem_fingerprint(gen, st, FP_BOOT, seed=SEED + si)
        delta[si], dlo[si], dhi[si] = fp["delta"], fp["lo"], fp["hi"]
        fp_tabs.append(fingerprint_table(fp, st, chem.gen_cols, mods, dic, XGf, chem.style))
    dev, sens_tab = sensory_fingerprint(styles, mu_by, bundle.dims, weak)
    fp_df = pd.concat(fp_tabs, ignore_index=True)
    fp_df["n_style"] = fp_df["style"].map(pd.Series(chem.style).value_counts())
    fp_df["small_style_shrunk"] = fp_df["n_style"] <= 5
    fp_df.to_csv(out_dir / "q3_fingerprint.csv", index=False, encoding="utf-8-sig")
    sens_tab.to_csv(out_dir / "q3_sensory_fingerprint.csv", index=False, encoding="utf-8-sig")
    arr = {k: np.array(v) for k, v in summ.items()}
    oof_mu = load_main_oof()["oof_mu"].mean(0)
    real_q50 = np.array([np.median(oof_mu[chem.style == st], 0) for st in styles])
    real_q10 = np.array([np.quantile(oof_mu[chem.style == st], 0.1, 0) for st in styles])
    real_q90 = np.array([np.quantile(oof_mu[chem.style == st], 0.9, 0) for st in styles])
    np.savez_compressed(
        out_dir / "q3_portraits.npz",
        styles=np.array(styles),
        dims=np.array(bundle.dims),
        mu=mu_all,
        **arr,
        delta=delta,
        delta_lo=dlo,
        delta_hi=dhi,
        sens_dev=dev,
        real_oof_q10=real_q10,
        real_oof_q50=real_q50,
        real_oof_q90=real_q90,
        weak_dim=weak,
        n_style=np.array([(chem.style == s).sum() for s in styles]),
        gen_cols=np.array(chem.gen_cols),
        config=json.dumps(cfg),
        seed=SEED,
        meta=run_meta(),
    )
    np.savez_compressed(
        out_dir / "q3_synthetic_profiles.npz",
        styles=np.array(styles),
        gen_cols=np.array(chem.gen_cols),
        profiles=synth,
        seed=SEED,
        meta=run_meta(),
    )
    checks = {
        "mu_no_nan": bool(np.isfinite(mu_all).all()),
        "pi_sum_max_err": float(np.abs(arr["pi"].sum(-1) - 1).max()),
        "pi_tilde_sum_max_err": float(np.abs(arr["pi_tilde"].sum(-1) - 1).max()),
        "n_styles": len(styles),
    }
    report = {"checks": checks, "weak_dims": [d for d, w in zip(bundle.dims, weak) if w]}
    for st in NAMED_STYLES:
        si = styles.index(st)
        report[st] = {
            "fingerprint_top10": fp_tabs[si][["compound", "delta", "ci_covers_zero"]].to_dict(
                "records"
            ),
            "sensory_top5": [
                (bundle.dims[j], float(dev[si, j])) for j in np.argsort(-np.abs(dev[si]))[:5]
            ],
        }
    report["wall_time_s"] = time.time() - t0
    report["config"] = cfg
    dump_json(report, out_dir / "q3_portraits.json")
    print(json.dumps(checks), f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
