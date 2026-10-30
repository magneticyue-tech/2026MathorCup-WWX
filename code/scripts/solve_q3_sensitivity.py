import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q3_genval import load_select, stage_gen, stage_merge, stage_metrics  # noqa: E402
from src.q3.config import GRID_K, GRID_KAPPA, N_BOOT, N_SYNTH, SRC_STYLE, TGT_STYLE  # noqa: E402
from src.q3.data import apply_floor_na, load_dictionary  # noqa: E402
from src.q3.fullmodel import fit_full, load_final_config, variant_data  # noqa: E402
from src.q3.meta import dump_json, run_meta, sha12  # noqa: E402
from src.q3.paths import (  # noqa: E402
    CHUNK_DIR,
    PROCESSED_DIR,
    Q1_DIR,
    SEED,
    SENS_DIR,
    SMOKE_DIR,
    ensure_dirs,  # noqa: E402
)
from src.q3.portrait import assemble_input  # noqa: E402
from src.q3.q1_bridge import load_q1_bundle, predict_mu  # noqa: E402
from src.q3.sensitivity import (  # noqa: E402
    ARMS,
    compare_generator,
    compare_portrait,
    compare_transfer,
)  # noqa: E402

PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent


def _run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run([PY, "-W", "ignore", *cmd], check=True, cwd=SCRIPTS.parents[1])


def load_genval(path):
    z = np.load(path, allow_pickle=False)
    styles = list(z["styles"])
    n = np.array([int(x) for x in z["n_style"]])
    out = {
        "keys": list(z["metric_keys"]),
        "estimate": z["metric_estimate"],
        "boot": z["metric_boot"],
        "weights": n / n.sum(),
        "per_style": {},
        "styles": styles,
    }
    if "ps_main_boot" in z.files:
        out["per_style"]["main"] = {"value": z["ps_main_auc"], "boot": z["ps_main_boot"]}
    if "ps_dispersion_boot" in z.files:
        out["per_style"]["dispersion"] = {
            "value": z["ps_dispersion_ratio"],
            "boot": z["ps_dispersion_boot"],
        }
    return out


def gen_arm_specs(base_cfg_by_fold):
    """返回 [(子臂名, opts, cfg_by_fold, data_variant)]。"""
    out = []
    lw = {k: {**v, "kind": "lw", "k": None} for k, v in base_cfg_by_fold.items()}
    out.append(("G-S2", {}, lw, "base"))
    nb = []
    ks = sorted({v["k"] for v in base_cfg_by_fold.values()})
    kaps = sorted({v["kappa"] for v in base_cfg_by_fold.values()})
    ref = base_cfg_by_fold[(0, 0)]
    ki, ai = GRID_K.index(ref["k"]), GRID_KAPPA.index(ref["kappa"])
    for name, dk, da in (("k-", -1, 0), ("k+", 1, 0), ("kappa-", 0, -1), ("kappa+", 0, 1)):
        if not (0 <= ki + dk < len(GRID_K) and 0 <= ai + da < len(GRID_KAPPA)):
            continue
        sel = {}
        for key, v in base_cfg_by_fold.items():
            kk, aa = GRID_K.index(v["k"]), GRID_KAPPA.index(v["kappa"])
            kk = min(max(kk + dk, 0), len(GRID_K) - 1)
            aa = min(max(aa + da, 0), len(GRID_KAPPA) - 1)
            sel[key] = {**v, "k": GRID_K[kk], "kappa": float(GRID_KAPPA[aa])}
        nb.append((f"G-S3_{name}", {}, sel, "base"))
    out += nb
    out.append(("G-S4", {"style_scale": True}, base_cfg_by_fold, "base"))
    out.append(("G-S5", {"tie_scheme": "mean_rank"}, base_cfg_by_fold, "base"))
    out.append(("G-S6", {"na_fill": "min"}, base_cfg_by_fold, "na_floor"))
    out.append(("G-S7", {}, base_cfg_by_fold, "att8"))
    del ks, kaps
    return out


def run_generator_arm(arm, args, base):
    sel = load_select(PROCESSED_DIR / "q3_select.npz")
    reps = [0] if args.smoke else list(range(5))
    nb = 20 if args.smoke else N_BOOT
    results, files = {}, {}
    if arm == "G-S1":
        bk = base["keys"]
        est = {
            k.replace("pure_fa.", "main."): base["estimate"][i]
            for i, k in enumerate(bk)
            if k.startswith("pure_fa.")
        }
        boot = {
            k.replace("pure_fa.", "main."): base["boot"][i]
            for i, k in enumerate(bk)
            if k.startswith("pure_fa.")
        }
        a = {
            "keys": list(est),
            "estimate": np.array(list(est.values())),
            "boot": np.array(list(boot.values())),
            "per_style": {},
        }
        z = np.load(PROCESSED_DIR / "q3_genval.npz", allow_pickle=False)
        if "ps_pure_fa_boot" in z.files:
            a["per_style"]["main"] = {"value": z["ps_pure_fa_auc"], "boot": z["ps_pure_fa_boot"]}
        results["G-S1"] = compare_generator(base, a)
        return results, {"q3_genval.npz": sha12(PROCESSED_DIR / "q3_genval.npz")}
    specs = [s for s in gen_arm_specs(sel) if s[0].split("_")[0] == arm]
    dic = load_dictionary()
    for name, opts, cfg, variant in specs:
        chem, extra = variant_data(variant, dic)
        X = chem.XG
        if variant == "att8":
            extra["replacements"].to_csv(
                SENS_DIR / "G-S7_replacements.csv", index=False, encoding="utf-8-sig"
            )
        if variant == "na_floor":
            na = np.isnan(X)
            filled = apply_floor_na(X)
            rows, cols = np.where(na)
            pd.DataFrame(
                {
                    "beer": chem.beer[rows],
                    "field": np.array(chem.gen_cols)[cols],
                    "replacement": filled[rows, cols],
                }
            ).to_csv(SENS_DIR / "G-S6_replacements.csv", index=False, encoding="utf-8-sig")
        chunk = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / f"sens_{name}"
        chunk.mkdir(parents=True, exist_ok=True)
        stage_gen(X, chem.style, chem.styles, cfg, chunk, reps, ("main",), opts, args.workers)
        stage_metrics(chem.style, chem.styles, chunk, reps, ("main",), args.workers, 0, 0, True)
        if base is None:
            continue
        out_npz = (SMOKE_DIR if args.smoke else SENS_DIR) / f"{name}_genval.npz"
        stage_merge(
            X,
            chem.style,
            chem.styles,
            chunk,
            reps,
            ("main",),
            out_npz,
            out_npz.with_suffix(".json"),
            args.workers,
            True,
            nb,
        )
        a = load_genval(out_npz)
        results[name] = compare_generator(base, a)
        files[out_npz.name] = sha12(out_npz)
    return results, files


def portrait_q50(chem, gen, comp, bundle, n, istd_mode="median"):
    styles = chem.styles
    cols = (chem.chem_cols, chem.gen_cols, chem.composite_cols, chem.internal_std_col)
    istd_obs = chem.X[chem.internal_std_col].to_numpy(float)
    out = np.zeros((len(styles), len(bundle.dims)))
    for si, st in enumerate(styles):
        rng = np.random.default_rng([SEED, si])
        u = gen.sample_latent(st, n, rng)
        if istd_mode == "median":
            iv = float(np.median(istd_obs))
        else:
            iv = np.random.default_rng([SEED, 900, si]).choice(istd_obs, n)
        mu = predict_mu(bundle, assemble_input(gen, comp, u, iv, *cols))
        out[si] = np.median(mu, 0)
    return out


def load_transfer(sfx):
    z = np.load(PROCESSED_DIR / f"q3_transfer{sfx}.npz", allow_pickle=False)
    act = (z["t0_chem_viol"] > 0) | (z["t0_sens_viol"] > 0)
    act = np.where(np.isnan(z["alpha_star"]), np.nan, act.astype(float))
    return {
        "alpha": z["alpha_star"],
        "p": z["p_main"],
        "active": act,
        "alpha_fit": z["alpha_fit"],
        "p_fit": z["p_fit"],
        "styles": list(z["styles"]),
    }


def transfer_compare(base, arm, chem):
    styles = base["styles"]
    S = chem.style
    src = np.where(S == SRC_STYLE)[0]
    ti = styles.index(TGT_STYLE)
    out = {"pils_tripel": compare_transfer(base, arm, src, ti)}
    if not np.isnan(arm["alpha"]).all() and np.isfinite(arm["alpha"]).sum() > len(src):
        out["all_pairs"] = compare_transfer(base, arm)
    out["pils_alpha_base"] = base["alpha"][src, ti].tolist()
    out["pils_alpha_arm"] = arm["alpha"][src, ti].tolist()
    return out


def run_transfer_arm(arm, args, chem):
    w = ["--workers", str(args.workers)]
    sm = ["--smoke"] if args.smoke else []
    base = load_transfer("")
    files = {"q3_transfer.npz": sha12(PROCESSED_DIR / "q3_transfer.npz")}
    if arm == "T-S2":
        a = {**base, "alpha": base["alpha_fit"], "p": base["p_fit"]}
        return {"T-S2": transfer_compare(base, a, chem)}, files
    plan = {
        "T-S1": ([], ["--comp-mode", "ridge"], "_ridge"),
        "T-S3": (["--thr-kind", "median"], ["--chem-suffix", "_median"], "_median"),
        "T-S4": (["--rule", "strict"], ["--chem-suffix", "_strict"], "_strict"),
        "T-S5a": ([], ["--rule", "mu_cont"], "_mu_cont"),
        "T-S5b": ([], ["--rule", "none"], "_none"),
        "T-S6": ([], ["--rule", "grade_quant"], "_grade_quant"),
        "T-S7a": (
            ["--variant", "na_floor"],
            ["--chem-suffix", "_na_floor", "--variant", "na_floor", "--pils-only"],
            "_na_floor",
        ),
        "T-S7b": (
            ["--variant", "att8"],
            ["--chem-suffix", "_att8", "--variant", "att8", "--pils-only"],
            "_att8",
        ),
        "T-S7c": (
            ["--variant", "mean_rank"],
            ["--chem-suffix", "_mean_rank", "--variant", "mean_rank", "--pils-only"],
            "_mean_rank",
        ),
    }
    chem_args, tr_args, sfx = plan[arm]
    if chem_args:
        _run(["code/scripts/solve_q3_transfer_chem.py", "--config", "final", *chem_args, *sm])
    _run(["code/scripts/solve_q3_transfer.py", *tr_args, *w, *sm])
    a = load_transfer(sfx)
    files[f"q3_transfer{sfx}.npz"] = sha12(PROCESSED_DIR / f"q3_transfer{sfx}.npz")
    res = transfer_compare(base, a, chem)
    if arm in ("T-S7a", "T-S1"):
        p = pd.read_csv(PROCESSED_DIR / f"q3_transfer_pils_tripel{sfx}.csv")
        b = pd.read_csv(PROCESSED_DIR / "q3_transfer_pils_tripel.csv")
        res["pils_rows"] = p.merge(b, on="beer", suffixes=("_arm", "_base"))[
            [
                "beer",
                "alpha_star_arm",
                "alpha_star_base",
                "category_arm",
                "category_base",
                "active_cols_arm",
                "active_cols_base",
                "src_missing_diacetyl_base",
            ]
        ].to_dict("records")
    if arm == "T-S1":
        pa = pd.read_csv(PROCESSED_DIR / f"q3_transfer_portraits{sfx}.csv")
        pb = pd.read_csv(PROCESSED_DIR / "q3_transfer_portraits.csv")
        dims = [c for c in pb.columns if c not in ("beer", "profile")]
        da = pa[pa.profile == "after"].set_index("beer")[dims]
        db = pb[pb.profile == "after"].set_index("beer")[dims]
        res["pils_sens_pred_change_after"] = float(np.abs(da - db.loc[da.index]).to_numpy().mean())
    return {arm: res}, files


def run_portrait_arm(arm, args, chem):
    dic = load_dictionary()
    cfg = load_final_config(PROCESSED_DIR / "q3_select.json")
    gen, comp = fit_full(chem, cfg)
    n = 50 if args.smoke else N_SYNTH
    base_z = np.load(PROCESSED_DIR / "q3_portraits.npz", allow_pickle=False)
    base_q50 = base_z["mu_q50"]
    files = {"q3_portraits.npz": sha12(PROCESSED_DIR / "q3_portraits.npz")}
    del dic
    if arm == "G-S9":
        bundle = load_q1_bundle(chem_cols=chem.chem_cols)
        q = portrait_q50(chem, gen, comp, bundle, n, "draw")
        np.savez_compressed(SENS_DIR / "G-S9_portrait.npz", mu_q50=q, seed=SEED, meta=run_meta())
        return {"G-S9": {"portrait": compare_portrait(base_q50, q)}}, files
    md = Q1_DIR / "q1_models_drop_sum"
    if not (md / "encoder.joblib").exists():
        return {"G-S8": {"note": "Q1 W7 的 q1_models_drop_sum 不存在，未运行"}}, files
    with open(Q1_DIR / "q1_drop_sum_final.json", encoding="utf-8") as f:
        ds = json.load(f)
    bundle = load_q1_bundle(md, chem.chem_cols)
    assert list(bundle.feature_names) == list(ds["input_columns"]), "drop_sum 编码器列与清单不一致"
    q = portrait_q50(chem, gen, comp, bundle, n)
    np.savez_compressed(SENS_DIR / "G-S8_portrait.npz", mu_q50=q, seed=SEED, meta=run_meta())
    _run(
        [
            "code/scripts/solve_q3_transfer.py",
            "--q1-models",
            "drop_sum",
            "--workers",
            str(args.workers),
        ]
        + (["--smoke"] if args.smoke else [])
    )
    a = load_transfer("_drop_sum")
    files["q3_transfer_drop_sum.npz"] = sha12(PROCESSED_DIR / "q3_transfer_drop_sum.npz")
    files["q1_drop_sum_final.json"] = sha12(Q1_DIR / "q1_drop_sum_final.json")
    res = {
        "portrait": compare_portrait(base_q50, q),
        "transfer": transfer_compare(load_transfer(""), a, chem),
        "note": (
            "残差池与切点沿用 Q1 主模型 final_resid/final_qcuts；"
            "q1_drop_sum_final.json 未提供对照模型残差"
        ),
        "n_input_columns": len(bundle.feature_names),
    }
    return {"G-S8": res}, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    if args.list:
        for k, v in ARMS.items():
            print(k, v["kind"], v["change"])
        return
    if args.summary:
        write_summary()
        return
    arms = list(ARMS) if args.arm in (None, "all") else args.arm.split(",")
    chem, _ = variant_data("base")
    for arm in arms:
        out_dir = SMOKE_DIR if args.smoke else SENS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        js = out_dir / f"{arm}.json"
        if js.exists() and not args.force and not args.smoke:
            print(f"跳过 {arm}（已存在）")
            continue
        t0 = time.time()
        kind = ARMS[arm]["kind"]
        if kind == "generator":
            base_path = (SMOKE_DIR if args.smoke else PROCESSED_DIR) / "q3_genval.npz"
            if not base_path.exists():
                if arm == "G-S1":
                    continue
                run_generator_arm(arm, args, None)
                print(f"{arm}：基线 q3_genval.npz 尚不存在，只完成臂的运行，比较待基线完成后重跑")
                continue
            base = load_genval(base_path)
            res, files = run_generator_arm(arm, args, base)
            files.setdefault("q3_genval.npz", sha12(base_path))
        elif kind == "transfer":
            res, files = run_transfer_arm(arm, args, chem)
        else:
            res, files = run_portrait_arm(arm, args, chem)
        wall = time.time() - t0
        dump_json(
            {"arm": arm, **ARMS[arm], "results": res, "inputs_sha12": files, "wall_time_s": wall},
            js,
        )
        np.savez_compressed(
            out_dir / f"{arm}.npz",
            result=json.dumps(res, default=float),
            seed=SEED,
            meta=run_meta(),
        )
        print(arm, f"{wall:.0f}s")
    if not args.smoke:
        write_summary()


GEN_METRICS = (
    "main.auc_marg",
    "main.auc_cond",
    "main.dcr_ratio",
    "main.near_copy",
    "main.tstr",
    "main.trts",
    "main.floor_err",
    "main.corr_err",
    "main.wcorr_err",
    "main.module_rho",
)


def write_summary():
    rows = []
    for arm, spec in ARMS.items():
        p = SENS_DIR / f"{arm}.json"
        if not p.exists():
            rows.append({"arm": arm, "baseline": "", "change": spec["change"], "note": "未运行"})
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        sha = ";".join(f"{k}:{v}" for k, v in d["inputs_sha12"].items())
        for sub, r in d["results"].items():
            base_row = {
                "arm": sub,
                "baseline": "W2 final_config / W3 / W5b 主设定",
                "change": spec["change"],
                "inputs_sha12": sha,
            }
            if spec["kind"] == "generator":
                for m in GEN_METRICS:
                    if m in r:
                        base_row[f"{m}.base"] = r[m]["base"]
                        base_row[f"{m}.arm"] = r[m]["arm"]
                        base_row[f"{m}.diff"] = r[m]["diff"]
                        base_row[f"{m}.diff_lo"] = r[m]["diff_ci95"][0]
                        base_row[f"{m}.diff_hi"] = r[m]["diff_ci95"][1]
                ps = r.get("per_style.main")
                if ps:
                    base_row["per_style_auc_macro.base"] = ps["macro_base"]
                    base_row["per_style_auc_macro.arm"] = ps["macro_arm"]
                    base_row["per_style_auc_macro.diff_lo"] = ps["macro_diff_ci95"][0]
                    base_row["per_style_auc_macro.diff_hi"] = ps["macro_diff_ci95"][1]
                base_row["note"] = ""
            elif spec["kind"] == "transfer":
                for scope in ("pils_tripel", "all_pairs"):
                    t = r.get(scope)
                    if t is None:
                        continue
                    for k in ("alpha_median", "active_frac", "p_median"):
                        base_row[f"{scope}.{k}.base"] = t[f"{k}_base"]
                        base_row[f"{scope}.{k}.arm"] = t[f"{k}_arm"]
                        base_row[f"{scope}.{k}.diff"] = t[f"{k}_diff"]
                    base_row[f"{scope}.n_alpha_changed"] = t["n_alpha_changed"]
                base_row["note"] = (
                    ""
                    if "all_pairs" in r
                    else "只对 Pils/Lager → Tripel 11 款运行；全部风格对的指标不适用"
                )
            else:
                if "portrait" in r:
                    for k, v in r["portrait"].items():
                        base_row[f"portrait.{k}"] = v
                if "transfer" in r:
                    for scope in ("pils_tripel", "all_pairs"):
                        t = r["transfer"].get(scope)
                        if t:
                            for k in ("alpha_median", "active_frac", "p_median"):
                                base_row[f"{scope}.{k}.base"] = t[f"{k}_base"]
                                base_row[f"{scope}.{k}.arm"] = t[f"{k}_arm"]
                                base_row[f"{scope}.{k}.diff"] = t[f"{k}_diff"]
                base_row["note"] = r.get("note", "生成器 §6.2 指标不适用（只改画像输入）")
            rows.append(base_row)
    pd.DataFrame(rows).to_csv(
        PROCESSED_DIR / "q3_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    print("写出 q3_sensitivity.csv", len(rows))


if __name__ == "__main__":
    main()
