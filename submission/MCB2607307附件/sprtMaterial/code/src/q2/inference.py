# code/src/q2/inference.py
"""Q2 显著性门槛：第 1 位逐行置换 max|r|，第 2、3 位 Freedman–Lane（含图约束）。"""
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


def main(n_perm=C.N_PERMUTATION, seed=C.SEED, out_dir=C.Q2_DIR):
    ds = load_q2_dataset()
    a = core.prepare_arm(ds, core.ArmSpec("main"))
    df = pd.read_csv(C.Q2_DIR / "top3_results.csv")
    sel = core.sel_from_table(df, a.cand, a.dims)
    q = core.permutation_thresholds(a.Zx, a.Zy, sel, a.mod, a.okx, n_perm, seed)
    df["q_d"] = q[np.repeat(np.arange(len(a.dims)), 3), np.tile([0, 1, 2], len(a.dims))]
    df["pass_threshold"] = df["score"].abs() > df["q_d"]
    df["n_permutation"] = n_perm
    df.to_csv(Path(out_dir) / "top3_results_with_sig.csv", index=False)
    np.savez_compressed(Path(out_dir) / "q2_permutation.npz", q=q, n_perm=n_perm, seed=seed)
    r1 = df[df["rank"] == 1]
    print(f"置换 {n_perm} 次（seed {seed}）：q_d 中位 {r1['q_d'].median():.3f}；"
          f"超过门槛 第1位 {int(r1['pass_threshold'].sum())}/49，"
          f"第2位 {int(df[df['rank'] == 2]['pass_threshold'].sum())}，"
          f"第3位 {int(df[df['rank'] == 3]['pass_threshold'].sum())}；"
          f"第1位未过门槛：{r1.loc[~r1['pass_threshold'], 'dimension'].tolist()}")


if __name__ == "__main__":
    main()
