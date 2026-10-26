# code/src/q2/graph.py
"""Q2 图构建：平均连接层次聚类、模块划分、节点/模块/边表。"""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

import config as C
from data import load_q2_dataset

def build_coexpression_graph():
    print("="*50)
    print("开始构建 Q2 化合物共表达图")
    print("="*50)
    
    ds = load_q2_dataset()
    all_nodes = ds.candidate_cols + ds.annotation_cols
    corr_matrix = ds.corr.loc[all_nodes, all_nodes]
    
    abs_corr = corr_matrix.abs()
    dist_matrix = 1.0 - abs_corr.values
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='average')
    
    cutoff_height = 1.0 - C.TAU_MAIN
    cluster_labels = fcluster(Z, t=cutoff_height, criterion='distance')
    node_to_module = dict(zip(all_nodes, cluster_labels))
    
    node_rows = []
    for node in all_nodes:
        m_id = node_to_module[node]
        module_nodes = [n for n in all_nodes if node_to_module[n] == m_id and n != node]
        k_in = abs_corr.loc[node, module_nodes].sum() if module_nodes else 0.0
        
        node_rows.append({
            "node": node,
            "module_id": m_id,
            "k_in": k_in,
            "is_candidate": node in ds.candidate_cols,
            "is_annotation": node in ds.annotation_cols
        })
    df_nodes = pd.DataFrame(node_rows)
    
    module_rows = []
    for m_id, group in df_nodes.groupby("module_id"):
        members = group["node"].tolist()
        hub = group.loc[group["k_in"].idxmax(), "node"]
        if len(members) > 1:
            sub_corr = abs_corr.loc[members, members].values
            upper_tri = sub_corr[np.triu_indices(len(members), k=1)]
            mean_abs_corr = np.mean(upper_tri)
        else:
            mean_abs_corr = 1.0
            
        module_rows.append({
            "module_id": m_id,
            "size": len(members),
            "hub": hub,
            "mean_abs_corr": mean_abs_corr,
            "members": ",".join(members)
        })
    df_modules = pd.DataFrame(module_rows).sort_values(by="size", ascending=False)
    
    edge_rows = []
    for i in range(len(all_nodes)):
        for j in range(i+1, len(all_nodes)):
            node_i, node_j = all_nodes[i], all_nodes[j]
            rho = corr_matrix.loc[node_i, node_j]
            if abs(rho) >= C.TAU_MAIN:
                edge_rows.append({"source": node_i, "target": node_j, "weight_abs": abs(rho), "rho": rho})
    df_edges = pd.DataFrame(edge_rows)
    
    df_nodes.to_csv(C.Q2_DIR / "graph_nodes.csv", index=False)
    df_modules.to_csv(C.Q2_DIR / "graph_modules.csv", index=False)
    df_edges.to_csv(C.Q2_DIR / "graph_edges.csv", index=False)
    
    print(f"✅ 图构建完成！节点 {len(df_nodes)} 个，边 {len(df_edges)} 条，已保存。")

if __name__ == "__main__":
    build_coexpression_graph()