# code/src/q2/simulation.py
"""Q2 模拟还原 (Simulation Recovery)：在已知标准答案下测试算法的找回率。"""
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

def forward_select(X_mat, y, cand_cols, node_to_module):
    """主方法：图约束前向选择"""
    n = len(y)
    r_1_all = (X_mat.T @ y) / n
    best_1_idx = np.argmax(np.abs(r_1_all))
    selected = [cand_cols[best_1_idx]]
    
    for rank in [2, 3]:
        S_indices = [cand_cols.index(s) for s in selected]
        X_S = np.column_stack([np.ones(n), X_mat[:, S_indices]])
        beta_y, _, _, _ = np.linalg.lstsq(X_S, y, rcond=None)
        e_y = y - X_S @ beta_y
        
        best_score = -1.0
        best_idx = -1
        selected_modules = {node_to_module[s] for s in selected}
        
        for j_idx, j_node in enumerate(cand_cols):
            if node_to_module[j_node] in selected_modules:
                continue
            x_j = X_mat[:, j_idx]
            beta_j, _, _, _ = np.linalg.lstsq(X_S, x_j, rcond=None)
            e_j = x_j - X_S @ beta_j
            denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
            r_j = np.dot(e_j, e_y) / denom if denom > 0 else 0.0
            if abs(r_j) > best_score:
                best_score = abs(r_j)
                best_idx = j_idx
        if best_idx != -1:
            selected.append(cand_cols[best_idx])
    return selected

def marginal_select(X_mat, y, cand_cols, node_to_module):
    """M1 方法：图约束边际选择"""
    n = len(y)
    r_all = (X_mat.T @ y) / n
    sorted_indices = np.argsort(-np.abs(r_all))
    
    selected = []
    selected_modules = set()
    for idx in sorted_indices:
        node = cand_cols[idx]
        m_id = node_to_module[node]
        if m_id in selected_modules:
            continue
        selected_modules.add(m_id)
        selected.append(node)
        if len(selected) == 3:
            break
    return selected

def run_simulation():
    print("="*60)
    print("开始模拟还原实验 (Simulation Recovery)")
    print("="*60)

    ds = load_q2_dataset()
    X_mat = ds.X_rank.values
    cand_cols = ds.candidate_cols
    n = len(ds.sample_ids)

    # 获取模块划分
    abs_corr = ds.corr.loc[cand_cols, cand_cols].abs()
    dist_matrix = 1.0 - abs_corr.values
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    labels = fcluster(linkage(squareform(dist_matrix, checks=False), method='average'), t=0.4, criterion='distance')
    node_to_module = dict(zip(cand_cols, labels))
    module_to_nodes = {}
    for node, m_id in node_to_module.items():
        module_to_nodes.setdefault(m_id, []).append(node)

    # 实验参数 (按文档 §7.2 设定)
    weight_sets = [(1.0, 0.75, 0.5), (1.0, 1.0, 1.0)]
    target_r_values = [0.2, 0.3, 0.45]
    N_SIM = 100  # 文档设置 200 次，这里为了速度用 100 次，你可改成 200

    np.random.seed(C.SEED)
    results = []

    for weights in weight_sets:
        for target_r in target_r_values:
            print(f"正在测试: 权重={weights}, 目标 r={target_r} ...")
            
            main_hits = {1: 0, 2: 0, 3: 0}
            m1_hits = {1: 0, 2: 0, 3: 0}
            main_mod_hits = {1: 0, 2: 0, 3: 0}
            m1_mod_hits = {1: 0, 2: 0, 3: 0}

            for _ in range(N_SIM):
                # 1. 随机从3个不同模块中选3个化合物
                valid_modules = [m for m, nodes in module_to_nodes.items() if len(nodes) > 0]
                sampled_modules = np.random.choice(valid_modules, 3, replace=False)
                
                idx1 = np.random.choice(module_to_nodes[sampled_modules[0]])
                idx2 = np.random.choice(module_to_nodes[sampled_modules[1]])
                idx3 = np.random.choice(module_to_nodes[sampled_modules[2]])
                
                c1 = X_mat[:, cand_cols.index(idx1)]
                c2 = X_mat[:, cand_cols.index(idx2)]
                c3 = X_mat[:, cand_cols.index(idx3)]
                
                # 2. 构造无噪声的 y_base，并计算与 c1 的相关性 r_base
                y_base = weights[0]*c1 + weights[1]*c2 + weights[2]*c3
                y_base = (y_base - y_base.mean()) / y_base.std()
                r_base = np.corrcoef(c1, y_base)[0, 1]
                
                # 3. 注入噪声，使得 c1 与最终的 y 相关系数精确达到 target_r
                k = target_r / r_base
                noise = np.random.randn(n)
                noise = (noise - noise.mean()) / noise.std()
                beta = np.sqrt((1 - k**2) / k**2)
                y = y_base + beta * noise
                
                # 验证相关系数是否准确
                # print(f"实际 r: {np.corrcoef(c1, y)[0,1]:.3f}")
                
                # 4. 运行主方法与 M1
                main_top3 = forward_select(X_mat, y, cand_cols, node_to_module)
                m1_top3 = marginal_select(X_mat, y, cand_cols, node_to_module)
                
                # 统计找回率 (化合物层)
                true_compounds = [idx1, idx2, idx3]
                for rank_idx, node in enumerate(main_top3):
                    if node in true_compounds:
                        main_hits[rank_idx+1] += 1
                for rank_idx, node in enumerate(m1_top3):
                    if node in true_compounds:
                        m1_hits[rank_idx+1] += 1

                # 统计找回率 (模块层)
                true_modules = set(sampled_modules)
                for rank_idx, node in enumerate(main_top3):
                    if node_to_module[node] in true_modules:
                        main_mod_hits[rank_idx+1] += 1
                for rank_idx, node in enumerate(m1_top3):
                    if node_to_module[node] in true_modules:
                        m1_mod_hits[rank_idx+1] += 1

            # 计算平均找回率
            results.append({
                "weights": str(weights),
                "target_r": target_r,
                "main_comp_1": main_hits[1]/N_SIM, "main_comp_2": main_hits[2]/N_SIM, "main_comp_3": main_hits[3]/N_SIM,
                "m1_comp_1": m1_hits[1]/N_SIM, "m1_comp_2": m1_hits[2]/N_SIM, "m1_comp_3": m1_hits[3]/N_SIM,
                "main_mod_1": main_mod_hits[1]/N_SIM, "main_mod_2": main_mod_hits[2]/N_SIM, "main_mod_3": main_mod_hits[3]/N_SIM,
                "m1_mod_1": m1_mod_hits[1]/N_SIM, "m1_mod_2": m1_mod_hits[2]/N_SIM, "m1_mod_3": m1_mod_hits[3]/N_SIM,
            })

    # 5. 打印结果
    print("\n" + "="*60)
    print("✅ 模拟还原实验完成！(每种设置运行 100 次)")
    print("="*60)
    print("【化合物层找回率】")
    print(f"{'权重':<15} | {'r':<4} | {'主方法(第1,2,3位)':<20} | {'M1边际(第1,2,3位)'}")
    print("-" * 65)
    for res in results:
        w = res['weights']
        r = res['target_r']
        m_str = f"{res['main_comp_1']:.2f}, {res['main_comp_2']:.2f}, {res['main_comp_3']:.2f}"
        m1_str = f"{res['m1_comp_1']:.2f}, {res['m1_comp_2']:.2f}, {res['m1_comp_3']:.2f}"
        print(f"{w:<15} | {r:<4} | {m_str:<20} | {m1_str}")

    # 保存到文件
    df_res = pd.DataFrame(results)
    df_res.to_csv(C.Q2_DIR / "simulation_recovery_results.csv", index=False)
    print(f"\n结果已保存至: {C.Q2_DIR / 'simulation_recovery_results.csv'}")

if __name__ == "__main__":
    run_simulation()