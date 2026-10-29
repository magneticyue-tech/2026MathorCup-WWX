# code/src/q2/association.py
"""Q2.2 图约束前向选择：49 维 × Top-3（barnyard 不单独建模）。"""
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
from labels import candidate_type


def main():
    ds = load_q2_dataset()
    nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    a = core.prepare_arm(ds, core.ArmSpec("main"))
    file_mod = nodes.set_index("node").loc[a.cand, "module_id"].astype(int).values
    if not (file_mod == a.mod).all():
        raise AssertionError("graph_nodes.csv 的模块与当前计算不一致，请先运行 graph.py")
    sel, score, _, _ = core.select_full(a)

    rows = []
    for d, dim in enumerate(a.dims):
        for k in range(3):
            j = sel[d, k]
            mates = [c for c, m in zip(a.cand, a.mod) if m == a.mod[j] and c != a.cand[j]]
            rows.append({
                "dimension": dim,
                "rank": k + 1,
                "compound": a.cand[j],
                "compound_type": candidate_type(a.cand[j]),
                "module_id": int(a.mod[j]),
                "same_module_compounds": "|".join(mates),
                "score_type": "spearman" if k == 0 else "partial_spearman",
                "score": float(score[d, k]),
            })
    df = pd.DataFrame(rows)
    core.check_147(df, "top3_results")
    df.to_csv(C.Q2_DIR / "top3_results.csv", index=False)
    np.savez_compressed(C.Q2_DIR / "q2_selection.npz", sel=sel, score=score, mod=a.mod,
                        cand=np.array(a.cand), dims=np.array(a.dims))
    top1 = np.abs(score[:, 0])
    print(f"Top-3 完成：{df['dimension'].nunique()} 维 {len(df)} 行；Top-1 |r| 中位 {np.median(top1):.2f}"
          f"（四分位 {np.quantile(top1, .25):.2f}–{np.quantile(top1, .75):.2f}）")


if __name__ == "__main__":
    main()
