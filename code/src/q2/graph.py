# code/src/q2/graph.py
"""Q2 图构建：平均连接层次聚类、模块划分、节点/模块/边表。"""
from __future__ import annotations

import sys
from pathlib import Path

# 路径补丁
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
    
    # 1. 读取数据
    ds = load_q2_dataset()
    
    # 我们使用所有 231 个节点（225候选 + 6注释）构图
    all_nodes = ds.candidate_cols + ds.annotation_cols
    corr_matrix = ds.corr.loc[all_nodes, all_nodes]
    
    # 边权用绝对值，符号作为属性保留。距离矩阵 = 1 - |rho|
    abs_corr = corr_matrix.abs()
    dist_matrix = 1.0 - abs_corr.values
    np.fill_diagonal(dist_matrix, 0.0)
    
    # 确保对称
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    # 2. 层次聚类 (平均连接, average linkage)
    # squareform 要求传入一维压缩向量
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='average')
    
    # 3. 切树划分模块：主值 tau = 0.6，高度切在 1 - tau = 0.4
    cutoff_height = 1.0 - C.TAU_MAIN
    cluster_labels = fcluster(Z, t=cutoff_height, criterion='distance')
    
    # 映射回节点名
    node_to_module = dict(zip(all_nodes, cluster_labels))
    
    # 4. 构建节点表
    print("正在生成节点表...")
    node_rows = []
    for node in all_nodes:
        m_id = node_to_module[node]
        # 找出同模块的其它节点
        module_nodes = [n for n in all_nodes if node_to_module[n] == m_id and n != node]
        
        # 模块内连接度 = sum(|rho|)
        k_in = abs_corr.loc[node, module_nodes].sum() if module_nodes else 0.0
        
        node_rows.append({
            "node": node,
            "module_id": m_id,
            "k_in": k_in,
            "is_candidate": node in ds.candidate_cols,
            "is_annotation": node in ds.annotation_cols
        })
    df_nodes = pd.DataFrame(node_rows)
    
    # 5. 构建模块表
    print("正在生成模块表...")
    module_rows = []
    for m_id, group in df_nodes.groupby("module_id"):
        members = group["node"].tolist()
        # 找出模块内的 hub（k_in 最大的节点）
        hub_row = group.loc[group["k_in"].idxmax()]
        hub = hub_row["node"]
        
        # 计算模块内的平均绝对相关（不含对角线）
        if len(members) > 1:
            sub_corr = abs_corr.loc[members, members].values
            # 取上三角，排除对角线
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
    df_modules = pd.DataFrame(module_rows)
    # 按 size 降序排列
    df_modules = df_modules.sort_values(by="size", ascending=False)
    
    # 6. 构建边表 (|rho| >= tau)
    print("正在生成边表...")
    edge_rows = []
    # 只遍历上三角，避免重复
    for i in range(len(all_nodes)):
        for j in range(i+1, len(all_nodes)):
            node_i = all_nodes[i]
            node_j = all_nodes[j]
            rho = corr_matrix.loc[node_i, node_j]
            if abs(rho) >= C.TAU_MAIN:
                edge_rows.append({
                    "source": node_i,
                    "target": node_j,
                    "weight_abs": abs(rho),
                    "rho": rho
                })
    df_edges = pd.DataFrame(edge_rows)
    
    # 7. 保存输出
    out_dir = C.Q2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_nodes.to_csv(out_dir / "graph_nodes.csv", index=False)
    df_modules.to_csv(out_dir / "graph_modules.csv", index=False)
    df_edges.to_csv(out_dir / "graph_edges.csv", index=False)
    
    print("\n✅ 图构建完成！")
    print(f"节点总数: {len(df_nodes)}")
    print(f"模块总数: {len(df_modules)}")
    
    # 按照文档，tau=0.6 时预期多成员模块有 33 个
    multi_member_modules = df_modules[df_modules["size"] > 1]
    print(f"多成员模块数: {len(multi_member_modules)} (期望: 33)")
    print(f"总边数: {len(df_edges)} (期望: 245)")
    
    print(f"结果已保存至: {out_dir}")

if __name__ == "__main__":
    build_coexpression_graph()