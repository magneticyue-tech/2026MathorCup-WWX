# code/src/q2/stability.py
"""Q2.3 日期组簇 bootstrap：每次重建图与模块，计算 π^c、π^m、CI、Jaccard、竞争者与等级。"""
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd

import config as C
import core
from data import load_q2_dataset


def main(n_boot=C.N_BOOTSTRAP, seed=C.SEED):
    ds = load_q2_dataset()
    spec = core.ArmSpec("main", n_boot=n_boot, seed=seed)
    a = core.prepare_arm(ds, spec)
    sig = pd.read_csv(C.Q2_DIR / "top3_results_with_sig.csv")
    sel = core.sel_from_table(sig, a.cand, a.dims)
    score = sig.sort_values(["dimension", "rank"]).set_index(["dimension", "rank"])["score"]
    score = np.array([[score[(dim, k)] for k in (1, 2, 3)] for dim in a.dims])
    q = np.load(C.Q2_DIR / "q2_permutation.npz")["q"]

    n_groups = len(np.unique(a.groups))
    print(f"bootstrap {n_boot} 次，单位 {spec.boot_unit}，每次抽取 {n_groups} 个日期组，seed {seed}")
    boot_sel, boot_fix, n_distinct = core.run_bootstrap(a, spec, sel, progress=True)
    print(f"各次重复抽到的不同日期组数：最小 {n_distinct.min()}，中位 {np.median(n_distinct):g}，最大 {n_distinct.max()}")
    summ = core.summarize_bootstrap(a, sel, boot_sel, boot_fix)
    table = core.build_table(a, sel, score, q, summ)

    keep = ["dimension", "rank", "compound", "compound_type", "same_module_compounds"]
    out = sig[keep].merge(table, on=["dimension", "rank", "compound"], how="right", validate="1:1")
    out["n_bootstrap"] = n_boot
    core.check_147(out, "final_top3_with_stability")
    out.to_csv(C.Q2_DIR / "final_top3_with_stability.csv", index=False)
    np.savez_compressed(C.Q2_DIR / "q2_bootstrap_main.npz", boot_sel=boot_sel, boot_fix=boot_fix,
                        sel=sel, mod=a.mod, cand=np.array(a.cand), dims=np.array(a.dims), seed=seed,
                        n_groups_drawn=n_groups, n_distinct_groups=n_distinct, boot_unit=spec.boot_unit)

    r1 = out[out["rank"] == 1]
    pos = out.groupby("rank")["pi_c"].mean()
    print(f"π^c 按位置 {pos.round(2).tolist()}；J_boot 均值 {r1['jaccard_boot'].mean():.3f}；"
          f"稳定/中等/不稳定 {[(out['stability'] == s).sum() for s in ('稳定', '中等', '不稳定')]}；"
          f"等级 {r1['grade'].value_counts().reindex(['A', 'B', 'C']).fillna(0).astype(int).to_dict()}")


if __name__ == "__main__":
    main()
