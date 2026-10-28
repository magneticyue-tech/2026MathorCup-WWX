# code/src/q2/sensitivity_runs.py
"""Q2 §7.3 敏感性：逐方案完整重跑（选择、置换、bootstrap、等级）并与正式基线比较。

可续跑：方案的 meta.json 记录完成状态与次数，已通过的方案在相同设置下跳过。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd

import config as C
import core
from check_attachment6 import match_attachment6, pairwise_marginal
from data import anchored_reference_target, attachment8_conflict_chem, load_q2_dataset

NA_TEXT = "不适用：逐对删除各系数样本集合不同，不能形成统一相关矩阵，不定义完整选择流程"


def arm_specs(ds, n_perm, n_boot):
    base = list(ds.candidate_cols)
    comp = [c for c in ds.chem_cols if c in base or c in C.COMPOSITE_COLS]
    istd = [c for c in ds.chem_cols if c in base or c == C.INTERNAL_STANDARD_COL]
    chem8, log8 = attachment8_conflict_chem(ds)
    target, anchored = anchored_reference_target(ds)
    kw = dict(n_perm=n_perm, n_boot=n_boot)
    return [
        (core.ArmSpec("tau_0.5", tau=0.5, **kw), "模块阈值 τ=0.5", {}),
        (core.ArmSpec("tau_0.7", tau=0.7, **kw), "模块阈值 τ=0.7", {}),
        (core.ArmSpec("cand_plus_composites", candidates=comp, **kw), "候选加回 5 个综合指标（230）", {}),
        (core.ArmSpec("cand_plus_internal_std", candidates=istd, **kw), "候选加回内标（226）", {}),
        (core.ArmSpec("na_lowest", na_scheme="lowest", **kw), "单化合物 NA 最低位并列", {}),
        (core.ArmSpec("attachment8_conflict", chem=chem8, graph_source="recompute", **kw),
         "24 款啤酒 80 个冲突单元格改用附件 8；图由替换后的附件 1 逐对重算", {"replacements": log8}),
        (core.ArmSpec("boot_fixed_graph", rebuild_graph=False, **kw), "bootstrap 中固定附件 2 图（与主分析同一组品评团索引）",
         {"reuse_q": True}),
        (core.ArmSpec("boot_beer_unit", boot_unit="beer", **kw), "bootstrap 单位改为单款啤酒（250 有放回）", {"reuse_q": True}),
        (core.ArmSpec("sensory_anchored", target=target, **kw),
         f"{len(anchored)} 个有锚维度改用 z 减去本团零点（团内无零点用该维度零点中位数）",
         {"anchored": anchored}),
    ]


def summarize(name, desc, table, cmp, n_perm, n_boot):
    r1 = table[table["rank"] == 1]
    pos = table.groupby("rank")["pi_c"].mean()
    abc = r1["grade"].value_counts().reindex(["A", "B", "C"]).fillna(0).astype(int)
    return {
        "arm": name, "description": desc, "n_dims": int(table["dimension"].nunique()),
        "n_permutation": n_perm, "n_bootstrap": n_boot,
        "mean_jaccard_compound": float(cmp["jaccard_compound"].mean()),
        "mean_jaccard_module": float(cmp["jaccard_module"].mean()),
        "n_top1_changed": int(cmp["top1_changed"].sum()),
        "n_grade_changed": int(cmp["grade_changed"].sum()),
        "grade_changed_dims": "|".join(
            f"{r.dimension}:{r.base_grade}->{r.arm_grade}" for r in cmp[cmp["grade_changed"]].itertuples()),
        "pi_c_pos1": float(pos[1]), "pi_c_pos2": float(pos[2]), "pi_c_pos3": float(pos[3]),
        "n_stable": int((table["stability"] == "稳定").sum()),
        "n_medium": int((table["stability"] == "中等").sum()),
        "n_unstable": int((table["stability"] == "不稳定").sum()),
        "n_A": int(abc["A"]), "n_B": int(abc["B"]), "n_C": int(abc["C"]),
        "not_applicable": "",
    }


def run(n_perm, n_boot, out_dir, force=False, only=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_q2_dataset()
    base_file = C.Q2_DIR / "final_top3_with_stability.csv"
    a0 = core.prepare_arm(ds, core.ArmSpec("main"))
    base_mod = core.module_label_map(a0.cand, a0.mod)

    spec0 = core.ArmSpec("baseline", n_perm=n_perm, n_boot=n_boot)
    base, ex0 = core.run_arm(ds, spec0)
    if n_perm == C.N_PERMUTATION and n_boot == C.N_BOOTSTRAP:
        formal = pd.read_csv(base_file)
        m = formal.merge(base, on=["dimension", "rank"], suffixes=("", "_re"))
        if not ((m["compound"] == m["compound_re"]).all() and (m["grade"] == m["grade_re"]).all()
                and np.allclose(m["pi_c"], m["pi_c_re"]) and np.allclose(m["q_d"], m["q_d_re"])):
            raise AssertionError("敏感性入口复算的基线与正式 final_top3_with_stability.csv 不一致")
        print("基线复算与正式结果逐项一致")
    base.to_csv(out_dir / "baseline_table.csv", index=False)

    summary = [summarize("baseline", "正式基线（τ=0.6，225 候选，列均值，附件 1，重建图，品评团）",
                         base, core.compare_tables(base, base, base_mod), n_perm, n_boot)]
    for spec, desc, extra in arm_specs(ds, n_perm, n_boot):
        if only and spec.name not in only:
            continue
        meta_path = out_dir / f"{spec.name}_meta.json"
        if meta_path.exists() and not force:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") == "ok" and meta["n_permutation"] == n_perm and meta["n_bootstrap"] == n_boot:
                table = pd.read_csv(out_dir / f"{spec.name}_table.csv")
                cmp = pd.read_csv(out_dir / f"{spec.name}_compare.csv")
                summary.append(summarize(spec.name, desc, table, cmp, n_perm, n_boot))
                print(f"[跳过] {spec.name} 已完成")
                continue
        t0 = time.time()
        print(f"[运行] {spec.name}：{desc}", flush=True)
        try:
            table, ex = core.run_arm(ds, spec, q_override=ex0["q"] if extra.get("reuse_q") else None)
            core.check_147(table, spec.name)
            cmp = core.compare_tables(base, table, base_mod)
            table.to_csv(out_dir / f"{spec.name}_table.csv", index=False)
            cmp.to_csv(out_dir / f"{spec.name}_compare.csv", index=False)
            if "replacements" in extra:
                extra["replacements"].to_csv(out_dir / "attachment8_replacements.csv", index=False)
            if "anchored" in extra:
                pd.Series(extra["anchored"], name="dimension").to_csv(out_dir / "anchored_dims.csv", index=False)
            s = summarize(spec.name, desc, table, cmp, n_perm, n_boot)
            summary.append(s)
            meta = {"status": "ok", "arm": spec.name, "description": desc, "n_permutation": n_perm,
                    "n_bootstrap": n_boot, "seed": spec.seed, "tau": spec.tau, "na_scheme": spec.na_scheme,
                    "n_candidates": len(ex["arm"].cand), "graph_source": spec.graph_source,
                    "rebuild_graph": spec.rebuild_graph, "boot_unit": spec.boot_unit,
                    "q_reused_from_baseline": bool(extra.get("reuse_q")),
                    "seconds": round(time.time() - t0, 1)}
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    Jaccard {s['mean_jaccard_compound']:.3f}/{s['mean_jaccard_module']:.3f}；Top-1 改变 "
                  f"{s['n_top1_changed']}；等级改变 {s['n_grade_changed']}；{meta['seconds']}s", flush=True)
        except Exception as e:
            meta_path.write_text(json.dumps({"status": "failed", "error": repr(e)}, ensure_ascii=False),
                                 encoding="utf-8")
            raise

    if not only or "pairwise_deletion" in only:
        a6, chem6, sens6 = match_attachment6(ds)
        R6, _ = pairwise_marginal(ds.chem_raw[chem6], ds.sensory_raw[sens6])
        err6 = float(np.nanmax(np.abs(R6.to_numpy() - a6.to_numpy(float))))
        if err6 > 1e-12:
            raise AssertionError(f"逐对删除复现附件 6 误差 {err6:.2e}")
        R, N = pairwise_marginal(ds.chem_raw[ds.candidate_cols], ds.sensory_raw[ds.delivery_sensory_cols])
        _, Cxy = core.corr_mats(a0.Zx, a0.okx, a0.Zy, a0.oky)
        long = R.stack().rename("rho_pairwise").reset_index()
        long.columns = ["compound", "dimension", "rho_pairwise"]
        long["n_effective"] = N.stack().values
        long["rho_mean_fill"] = Cxy.ravel()
        long.to_csv(out_dir / "pairwise_deletion_marginal.csv", index=False)
        summary.append({
            "arm": "pairwise_deletion", "description": "逐对删除：只作边际 Spearman 诊断",
            "n_dims": int(long["dimension"].nunique()), "n_permutation": np.nan, "n_bootstrap": np.nan,
            "mean_jaccard_compound": np.nan, "mean_jaccard_module": np.nan, "n_top1_changed": np.nan,
            "n_grade_changed": np.nan, "grade_changed_dims": "",
            "not_applicable": f"{NA_TEXT}；附件 6 44 行复算最大误差 {err6:.2e}；有效样本数范围 "
                              f"{int(long['n_effective'].min())}–{int(long['n_effective'].max())}；"
                              f"与列均值边际系数最大差 {float((long['rho_pairwise'] - long['rho_mean_fill']).abs().max()):.3f}",
        })
        print(f"[逐对删除] 附件 6 误差 {err6:.2e}；Top-3/Jaccard/Top-1/等级：不适用")

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="10 次置换/bootstrap，写入 _smoke，重复两次核对一致")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    if args.smoke:
        d1 = run(10, 10, C.SMOKE_DIR / "sensitivity_a", force=True, only=args.only)
        d2 = run(10, 10, C.SMOKE_DIR / "sensitivity_b", force=True, only=args.only)
        pd.testing.assert_frame_equal(d1, d2)
        for arm in d1["arm"]:
            if arm == "pairwise_deletion":
                continue
            f = "baseline_table.csv" if arm == "baseline" else f"{arm}_table.csv"
            pd.testing.assert_frame_equal(pd.read_csv(C.SMOKE_DIR / "sensitivity_a" / f),
                                          pd.read_csv(C.SMOKE_DIR / "sensitivity_b" / f))
        assert (d1["n_dims"] == 49).all()
        print("敏感性小样本：两次固定种子运行逐项一致；各方案 49 维")
        print(d1.drop(columns=["description", "not_applicable", "grade_changed_dims"]).round(3).to_string(index=False))
    else:
        df = run(C.N_PERMUTATION, C.N_BOOTSTRAP, C.SENS_DIR, force=args.force, only=args.only)
        print(df.drop(columns=["description", "not_applicable", "grade_changed_dims"]).round(3).to_string(index=False))
