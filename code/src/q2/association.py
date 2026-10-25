# code/src/q2/association.py
"""Q2 关联层：图约束前向选择，计算每个感官维度的 Top-3 关键化合物。"""
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
    print("开始计算感官维度的 Top-3 关键化合物（图约束前向选择）")
    print("="*50)

    # 1. 加载数据
    ds = load_q2_dataset()
    
    # 2. 读取图结构中的模块信息
    node_file = C.Q2_DIR / "graph_nodes.csv"
    if not node_file.exists():
        raise FileNotFoundError(f"找不到节点表文件：{node_file}，请先运行 graph.py！")
    df_nodes = pd.read_csv(node_file)
    node_to_module = dict(zip(df_nodes["node"], df_nodes["module_id"]))

    # 提取矩阵数据
    X_mat = ds.X_rank.values  # 250 x 225 (已经标准化，直接内积即相关系数)
    Y_mat = ds.Y_rank.values  # 250 x 50
    cand_cols = ds.candidate_cols
    dim_cols = ds.Y_rank.columns
    n = len(ds.sample_ids)

    results = []

    # 3. 遍历每一个感官维度
    for d_idx, d_name in enumerate(dim_cols):
        # 如果维度叫 barnyard，我们按文档要求映射到 stale_hops 的模块和计算
        # 但按照附件 4，barnyard 列是独立存在的，所以这里还是按照它本身的 y 值算。
        y = Y_mat[:, d_idx]
        
        # --- 第 1 位：边际相关最强 ---
        # (1/n) * x^T * y 就是 Pearson，因为输入已经是秩，所以也是 Spearman
        r_1_all = (X_mat.T @ y) / n
        abs_r_1_all = np.abs(r_1_all)
        
        best_1_idx = np.argmax(abs_r_1_all)
        best_1_node = cand_cols[best_1_idx]
        best_1_r = r_1_all[best_1_idx]
        
        selected = [best_1_node]
        results.append({
            "dimension": d_name,
            "rank": 1,
            "compound": best_1_node,
            "module_id": node_to_module.get(best_1_node, -1),
            "score_type": "spearman",
            "score": best_1_r
        })

        # --- 第 2、3 位：偏相关最强（图约束） ---
        for rank in [2, 3]:
            # 构造已选化合物的设计矩阵 S，包含截距项（虽然数据已中心化，但为了严谨保留）
            S_indices = [cand_cols.index(s) for s in selected]
            X_S = np.column_stack([np.ones(n), X_mat[:, S_indices]])
            
            # 计算 e_y: y 对 S 回归后的残差
            beta_y, _, _, _ = np.linalg.lstsq(X_S, y, rcond=None)
            e_y = y - X_S @ beta_y
            
            best_score = -1.0
            best_idx = -1
            best_r = 0.0
            
            # 当前已选化合物的模块集合
            selected_modules = {node_to_module[s] for s in selected}
            
            # 遍历所有候选化合物（同模块的跳过）
            for j_idx, j_node in enumerate(cand_cols):
                # 图约束：如果该化合物所在模块已被选过，直接跳过
                if node_to_module.get(j_node) in selected_modules:
                    continue
                    
                x_j = X_mat[:, j_idx]
                
                # 计算 e_j: x_j 对 S 回归后的残差
                beta_j, _, _, _ = np.linalg.lstsq(X_S, x_j, rcond=None)
                e_j = x_j - X_S @ beta_j
                
                # 计算 e_j 与 e_y 的相关系数（即偏 Spearman）
                denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
                if denom == 0:
                    r_j = 0.0
                else:
                    r_j = np.dot(e_j, e_y) / denom
                
                if abs(r_j) > best_score:
                    best_score = abs(r_j)
                    best_idx = j_idx
                    best_r = r_j
            
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
                # 理论上不会发生，除非 225 个候选全在一个模块里
                print(f"警告：维度 {d_name} 找不到第 {rank} 个化合物。")

    # 4. 保存输出
    df_results = pd.DataFrame(results)
    
    # 处理 barnyard 的展示逻辑（按方法论要求）
    df_results["display_dimension"] = df_results["dimension"].apply(
        lambda x: "stale_hops (同 barnyard)" if x == "barnyard" else x
    )
    
    out_file = C.Q2_DIR / "top3_results.csv"
    df_results.to_csv(out_file, index=False)
    
    print("\n✅ 关联分析完成！")
    print(f"共计算了 {len(dim_cols)} 个维度，保存至: {out_file}")
    
    # 打印几个维度的结果供预览
    print("\n【预览】部分维度的 Top-3：")
    for dim in ["A_malt_all", "A_hops_citrus", "lactic", "acetic"]:
        if dim in dim_cols:
            print(f"\n--- {dim} ---")
            sub = df_results[df_results["dimension"] == dim]
            for _, row in sub.iterrows():
                print(f"Top {row['rank']}: {row['compound']:<30} (M{row['module_id']}) r={row['score']:.3f}")

if __name__ == "__main__":
    compute_top3()