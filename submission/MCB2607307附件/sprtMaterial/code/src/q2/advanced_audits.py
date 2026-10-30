# code/src/q2/advanced_audits.py
"""Q2 审计：附件 2/6 复算验收（调用 check_attachment6）与 §6.6 C 级维度零膨胀检查。"""
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
from check_attachment6 import check_att6
from data import load_q2_dataset


def test_attachment6_replication():
    return check_att6(verbose=True)


def zero_inflation(ds, dims):
    """§6.6：C 级维度中 x̄=0（全部品评员均评 0）的啤酒占比 ≥30% 者，在 x̄>0 的啤酒上重算 Top-1 关联。"""
    cand = ds.candidate_cols
    Zx_full = ds.X_rank.to_numpy()
    rows = []
    for dim in dims:
        x = ds.xbar[dim].to_numpy()
        zero_share = float((x == 0).mean())
        y = ds.Y_rank[dim].to_numpy()
        r_full = Zx_full.T @ y / len(y)
        j_full = int(np.argmax(np.abs(r_full)))
        row = {"dimension": dim, "n_all": len(x), "zero_share": zero_share,
               "eligible": zero_share >= C.ZERO_SHARE_MIN, "top1_all": cand[j_full], "r_all": float(r_full[j_full])}
        if row["eligible"]:
            mask = x > 0
            Zx, okx = core.zrank(core.fill_array(ds.chem_raw.loc[mask, cand].to_numpy(float), "mean"))
            Zy, _ = core.zrank(x[mask][:, None])
            r_sub = Zx.T @ Zy[:, 0] / mask.sum()
            r_sub[~okx] = 0.0
            j_sub = int(np.argmax(np.abs(r_sub)))
            row.update({
                "n_above_zero": int(mask.sum()), "top1_above_zero": cand[j_sub], "r_above_zero": float(r_sub[j_sub]),
                "r_above_zero_of_top1_all": float(r_sub[j_full]),
                "abs_r_increased": bool(abs(r_sub[j_sub]) > abs(r_full[j_full])),
                "note": "",
            })
        else:
            row["note"] = f"不适用：x̄=0 的啤酒占比 {zero_share:.3f} < {C.ZERO_SHARE_MIN}，不满足方法 §6.6 的检查条件"
        rows.append(row)
    cols = ["dimension", "n_all", "zero_share", "eligible", "top1_all", "r_all", "n_above_zero", "top1_above_zero",
            "r_above_zero", "r_above_zero_of_top1_all", "abs_r_increased", "note"]
    return pd.DataFrame(rows, columns=cols)


def main():
    acc = test_attachment6_replication()
    ds = load_q2_dataset()
    fin = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    r1 = fin[(fin["method"] == "main") & (fin["rank"] == 1)]
    c_dims = r1.loc[r1["grade"] == "C", "dimension"].tolist()
    df = zero_inflation(ds, c_dims)
    df.to_csv(C.Q2_DIR / "zero_inflation_c_grade.csv", index=False)
    print(f"C 级维度 {c_dims}")
    print(df.round(3).to_string(index=False))
    return acc


if __name__ == "__main__":
    main()
