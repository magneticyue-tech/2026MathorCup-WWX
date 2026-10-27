# code/src/q2/association.py
"""Q2 关联层：图约束前向选择 (严格 49 维，剔除 barnyard)。"""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd

import config as C
from data import load_q2_dataset

def compute_top3():
    print("="*50)
    print("计算 Top-3 关键化合物（前向选择）")
    print("="*50)

    ds = load_q2_dataset()
    df_nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    node_to_module = dict(zip(df_nodes["node"], df_nodes["module_id"]))

    X_mat = ds.X_rank.values
    Y_mat = ds.Y_rank.values
    cand_cols = ds.candidate_cols
    
    # 【严格修复】：剔除 barnyard 维度，保留 49 维
    dim_cols = [d for d in ds.Y_rank.columns if d != "barnyard"]
    n = len(ds.sample_ids)

    results = []
    for d_idx, d_name in enumerate(dim_cols):
        y = Y_mat[:, ds.Y_rank.columns.get_loc(d_name)]
        
        # 第 1 位
        r_1_all = (X_mat.T @ y) / n
        best_1_idx = np.argmax(np.abs(r_1_all))
        best_1_node = cand_cols[best_1_idx]
        
        selected = [best_1_node]
        results.append({
            "dimension": d_name, "rank": 1, "compound": best_1_node,
            "module_id": node_to_module.get(best_1_node, -1),
            "score_type": "spearman", "score": r_1_all[best_1_idx]
        })

        # 第 2、3 位
        for rank in [2, 3]:
            S_indices = [cand_cols.index(s) for s in selected]
            X_S = np.column_stack([np.ones(n), X_mat[:, S_indices]])
            
            beta_y, _, _, _ = np.linalg.lstsq(X_S, y, rcond=None)
            e_y = y - X_S @ beta_y
            
            best_score, best_idx, best_r = -1.0, -1, 0.0
            selected_modules = {node_to_module[s] for s in selected}
            
            for j_idx, j_node in enumerate(cand_cols):
                if node_to_module.get(j_node) in selected_modules:
                    continue
                x_j = X_mat[:, j_idx]
                beta_j, _, _, _ = np.linalg.lstsq(X_S, x_j, rcond=None)
                e_j = x_j - X_S @ beta_j
                
                denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
                r_j = np.dot(e_j, e_y) / denom if denom > 0 else 0.0
                
                if abs(r_j) > best_score:
                    best_score, best_idx, best_r = abs(r_j), j_idx, r_j
            
            if best_idx != -1:
                best_node = cand_cols[best_idx]
                selected.append(best_node)
                results.append({
                    "dimension": d_name, "rank": rank, "compound": best_node,
                    "module_id": node_to_module.get(best_node, -1),
                    "score_type": "partial_spearman", "score": best_r
                })

    df_results = pd.DataFrame(results)
    # 【严格修复】：按工单要求在 stale_hops 注明同 barnyard
    df_results["display_dimension"] = df_results["dimension"].apply(
        lambda x: "stale_hops (同 barnyard)" if x == "stale_hops" else x
    )
    df_results.to_csv(C.Q2_DIR / "top3_results.csv", index=False)
    print(f"✅ Top-3 提取完成！共 {len(dim_cols)} 维，已保存 top3_results.csv (无独立 barnyard)")

if __name__ == "__main__":
    compute_top3()