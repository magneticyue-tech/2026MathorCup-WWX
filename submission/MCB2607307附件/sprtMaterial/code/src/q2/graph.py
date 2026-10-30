# code/src/q2/graph.py
"""Q2.1 共表达图：231 节点、候选模块（τ=0.6 平均连接）、节点/模块/边表。"""
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
from labels import attachment8_labels, candidate_type


def build_graph(ds, tau=C.TAU_MAIN):
    cand, ann = ds.candidate_cols, ds.annotation_cols
    nodes = cand + ann
    R = ds.corr.loc[nodes, nodes]
    A = R.abs()
    mod = core.modules(A.loc[cand, cand].values, tau)
    cmod = dict(zip(cand, mod.astype(int)))
    lab = attachment8_labels(nodes)

    rows = []
    for c in nodes:
        if c in cmod:
            mates = [k for k in cand if cmod[k] == cmod[c] and k != c]
            k_in = float(A.loc[c, mates].sum()) if mates else 0.0
            near, near_r = "", np.nan
        else:
            mates, k_in = [], np.nan
            s = A.loc[c, cand]
            near, near_r = s.idxmax(), float(R.loc[c, s.idxmax()])
        rows.append({
            "node": c,
            "is_candidate": c in cmod,
            "is_annotation": c not in cmod,
            "node_type": candidate_type(c),
            "module_id": cmod.get(c, np.nan),
            "module_size": len(mates) + 1 if c in cmod else np.nan,
            "k_in": k_in,
            "origin_paper": lab.at[c, "origin_paper"],
            "chemical_name": lab.at[c, "chemical_name"],
            "nearest_candidate": near,
            "nearest_candidate_rho": near_r,
        })
    df_nodes = pd.DataFrame(rows)
    df_nodes["is_hub"] = False
    for m, g in df_nodes[df_nodes.is_candidate].groupby("module_id"):
        df_nodes.loc[g["k_in"].idxmax(), "is_hub"] = True

    mrows = []
    for m, g in df_nodes[df_nodes.is_candidate].groupby("module_id"):
        members = g["node"].tolist()
        sub = A.loc[members, members].values
        iu = np.triu_indices(len(members), 1)
        comp = g["origin_paper"].fillna("NA").value_counts()
        mrows.append({
            "module_id": int(m),
            "size": len(members),
            "hub": g.loc[g["k_in"].idxmax(), "node"],
            "mean_abs_rho": float(sub[iu].mean()) if len(members) > 1 else np.nan,
            "origin_composition": ";".join(f"{k}:{v}" for k, v in comp.items()),
            "majority_origin_share": float(comp.iloc[0] / comp.sum()),
            "members": "|".join(members),
        })
    df_mod = pd.DataFrame(mrows).sort_values(["size", "module_id"], ascending=[False, True])

    erows = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            r = R.iat[i, j]
            if abs(r) >= tau:
                a, b = nodes[i], nodes[j]
                both = a in cmod and b in cmod
                erows.append({
                    "source": a, "target": b, "rho": float(r), "weight_abs": float(abs(r)),
                    "sign": "+" if r >= 0 else "-",
                    "edge_type": "candidate-candidate" if both else "annotation",
                    "same_module": bool(both and cmod[a] == cmod[b]),
                })
    df_edges = pd.DataFrame(erows)
    return df_nodes, df_mod, df_edges


def main():
    ds = load_q2_dataset()
    nodes, mods, edges = build_graph(ds)
    n_cc = int((edges.edge_type == "candidate-candidate").sum())
    n_ann = int((edges.edge_type == "annotation").sum())
    if len(nodes) != 231 or n_cc != 245:
        raise AssertionError(f"节点 {len(nodes)}（应 231），候选间边 {n_cc}（应 245）")
    nodes.to_csv(C.Q2_DIR / "graph_nodes.csv", index=False)
    mods.to_csv(C.Q2_DIR / "graph_modules.csv", index=False)
    edges.to_csv(C.Q2_DIR / "graph_edges.csv", index=False)
    multi = mods[mods["size"] > 1]
    print(f"节点 {len(nodes)}；候选间边 {n_cc}；注释节点相关边 {n_ann}；模块 {len(mods)}"
          f"（多成员 {len(multi)}，覆盖 {int(multi['size'].sum())}，最大 {int(mods['size'].max())}）；"
          f"多成员模块来源纯度 {multi['majority_origin_share'].mean():.2f}")


if __name__ == "__main__":
    main()
