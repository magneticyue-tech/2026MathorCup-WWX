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
    print("=" * 50)
    print("计算 Top-3 关键化合物（前向选择）")
    print("=" * 50)

    ds = load_q2_dataset()
    
    all_nodes = ds.candidate_cols + ds.annotation_cols
    if len(all_nodes) != 231:
        raise ValueError(f"图节点应为 231，实际为 {len(all_nodes)}")
        
    print(f"✅ 图节点验证通过：共 {len(all_nodes)} 个节点。")

    graph_nodes_path = C.Q2_DIR / "graph_nodes.csv"
    if not graph_nodes_path.exists():
        raise FileNotFoundError(f"找不到图节点文件：{graph_nodes_path}")
        
    df_nodes = pd.read_csv(graph_nodes_path)
    
    # 校验模块映射是否覆盖所有候选列
    node_to_module = dict(zip(df_nodes["node"], df_nodes["module_id"]))
    missing_nodes = [c for c in ds.candidate_cols if c not in node_to_module]
    if missing_nodes:
        raise ValueError(f"graph_nodes.csv 缺失以下候选节点的模块映射：{missing_nodes[:10]}...")

    X_mat = ds.X_rank.values
    Y_mat = ds.Y_rank.values
    cand_cols = ds.candidate_cols
    
    # 严格 49 维（Y_rank 已在数据层排除了 barnyard）
    dim_cols = ds.Y_rank.columns.tolist()
    n = len(ds.sample_ids)

    results = []
    for d_idx, d_name in enumerate(dim_cols):
        y = Y_mat[:, ds.Y_rank.columns.get_loc(d_name)]
        
        # 第 1 位：边际相关（Spearman）
        r_1_all = (X_mat.T @ y) / n
        best_1_idx = np.argmax(np.abs(r_1_all))
        best_1_node = cand_cols[best_1_idx]
        
        selected = [best_1_node]
        results.append({
            "dimension": d_name, 
            "rank": 1, 
            "compound": best_1_node,
            "module_id": node_to_module.get(best_1_node, -1),
            "score_type": "spearman", 
            "score": r_1_all[best_1_idx]
        })

        # 第 2、3 位：固定 S 集合的偏相关（Partial Spearman）
        for rank in [2, 3]:
            S_indices = [cand_cols.index(s) for s in selected]
            X_S = np.column_stack([np.ones(n), X_mat[:, S_indices]])
            
            # 对 y 和候选 x_j 分别针对 X_S 做回归取残差
            beta_y, _, _, _ = np.linalg.lstsq(X_S, y, rcond=None)
            e_y = y - X_S @ beta_y
            
            best_score, best_idx, best_r = -1.0, -1, 0.0
            selected_modules = {node_to_module[s] for s in selected}
            
            for j_idx, j_node in enumerate(cand_cols):
                # 模块约束：确保 Top-3 来自不同模块
                if node_to_module.get(j_node) in selected_modules:
                    continue
                    
                x_j = X_mat[:, j_idx]
                beta_j, _, _, _ = np.linalg.lstsq(X_S, x_j, rcond=None)
                e_j = x_j - X_S @ beta_j
                
                # 计算残差的 Pearson 相关即偏相关
                denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
                r_j = np.dot(e_j, e_y) / denom if denom > 1e-12 else 0.0
                
                if abs(r_j) > best_score:
                    best_score, best_idx, best_r = abs(r_j), j_idx, r_j
            
            if best_idx != -1:
                best_node = cand_cols[best_idx]
                selected.append(best_node)
                results.append({
                    "dimension": d_name, 
                    "rank": rank, 
                    "compound": best_node,
                    "module_id": node_to_module.get(best_node, -1),
                    "score_type": "partial_spearman", 
                    "score": best_r
                })
            else:
                raise RuntimeError(f"维度 {d_name} 的前向选择在 rank {rank} 失败，可能是模块约束导致无候选可选。")

    df_results = pd.DataFrame(results)
    
    # 检查是否每个维度严格 3 行
    assert len(df_results) == 49 * 3, f"期望 147 行结果，实际生成了 {len(df_results)} 行"
    
    df_results.to_csv(C.Q2_DIR / "top3_results.csv", index=False)
    print(f"✅ Top-3 提取完成！共 {len(dim_cols)} 维，已保存 top3_results.csv (无独立 barnyard)")


if __name__ == "__main__":
    compute_top3()