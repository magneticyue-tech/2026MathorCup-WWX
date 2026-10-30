# code/src/q2/controls_sensitivity.py
"""Q2 对照 M0（无图约束边际前 3）与 M1（图约束边际选择）。τ 等敏感性见 sensitivity_runs.py。"""
from __future__ import annotations

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


def main():
    ds = load_q2_dataset()
    a = core.prepare_arm(ds, core.ArmSpec("main"))
    base = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    base = base[base["method"] == "main"]
    core.check_147(base, "主方法")
    sel = core.sel_from_table(base, a.cand, a.dims)
    _, Cxy = core.corr_mats(a.Zx, a.okx, a.Zy, a.oky)
    m0 = core.ranked_select(Cxy, None, a.okx)
    m1 = core.ranked_select(Cxy, a.mod, a.okx)

    rows = []
    for d, dim in enumerate(a.dims):
        b = sel[d]
        row = {"dimension": dim, "main_top3": "|".join(a.cand[j] for j in b)}
        for name, s in (("M0", m0[d]), ("M1", m1[d])):
            row[f"{name}_top3"] = "|".join(a.cand[j] for j in s)
            row[f"jaccard_{name}"] = core.jaccard(b, s)
            row[f"jaccard_mod_{name}"] = core.jaccard(a.mod[b], a.mod[s])
            row[f"{name}_top1_match"] = bool(b[0] == s[0])
        row["M0_n_same_module_pairs"] = int(len(m0[d]) - len(set(a.mod[m0[d]])))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(C.Q2_DIR / "controls_sensitivity_results.csv", index=False)
    for m in ("M0", "M1"):
        print(f"主方法 vs {m}：化合物层 Jaccard {df[f'jaccard_{m}'].mean():.3f}，模块层 "
              f"{df[f'jaccard_mod_{m}'].mean():.3f}，Top-1 相同比例 {df[f'{m}_top1_match'].mean():.2f}")
    print(f"M0 前 3 含同模块化合物的维度 {int((df['M0_n_same_module_pairs'] > 0).sum())}/49")


if __name__ == "__main__":
    main()
