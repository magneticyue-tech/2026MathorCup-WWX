# code/src/q2/inference.py
"""Q2 推断层：严格 Freedman-Lane 置换（包含图约束排除）。"""
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: 
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
import config as C
from data import load_q2_dataset


def run_permutation_test():
    print("=" * 50)
    print("计算置换零分布门槛（严格 Freedman-Lane + 图约束）")
    print("=" * 50)

    ds = load_q2_dataset()
    X_mat, Y_mat = ds.X_rank.values, ds.Y_rank.values
    cand_cols = ds.candidate_cols
    n = len(ds.sample_ids)
    
    # 建立快速索引
    col_to_idx = {c: i for i, c in enumerate(cand_cols)}

    # 读取图模块信息用于约束
    df_nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    node_to_module = dict(zip(df_nodes["node"], df_nodes["module_id"]))

    df_top3 = pd.read_csv(C.Q2_DIR / "top3_results.csv")
    np.random.seed(C.SEED)
    N_PERM = C.N_PERMUTATION
    
    # Y_rank 已经是 49 维
    dim_cols = ds.Y_rank.columns.tolist()
    q_d_dict = {}

    for d_name in dim_cols:
        print(f"  处理维度: {d_name}...")
        y = Y_mat[:, ds.Y_rank.columns.get_loc(d_name)]
        
        # 1. 第 1 位 (边际)
        max_abs_r_perms_1 = np.zeros(N_PERM)
        for p in range(N_PERM):
            y_perm = np.random.permutation(y)
            r_perms = (X_mat.T @ y_perm) / n
            max_abs_r_perms_1[p] = np.max(np.abs(r_perms))
        q1 = np.percentile(max_abs_r_perms_1, 95)

        # 获取全样本选出的 S
        top3_rows = df_top3[df_top3["dimension"] == d_name].sort_values("rank")
        selected = top3_rows["compound"].tolist()
        
        q2, q3 = np.nan, np.nan
        
        # 2. 第 2 位 (Freedman-Lane + 图约束)
        if len(selected) >= 1:
            S_idx = [col_to_idx[selected[0]]]
            X_S = np.column_stack([np.ones(n), X_mat[:, S_idx]])
            
            # 预计算 e_y
            e_y = y - X_S @ np.linalg.lstsq(X_S, y, rcond=None)[0]
            
            selected_mods = {node_to_module.get(selected[0], -1)}
            valid_j = [j for j, c in enumerate(cand_cols) if node_to_module.get(c) not in selected_mods]
            
            # 【性能核心修复】：一次性计算所有合法候选者的残差 e_j 并归一化
            X_valid = X_mat[:, valid_j]
            e_X_valid = X_valid - X_S @ np.linalg.lstsq(X_S, X_valid, rcond=None)[0]
            norm_X_valid = np.linalg.norm(e_X_valid, axis=0)
            
            max_abs_r_perms_2 = np.zeros(N_PERM)
            for p in range(N_PERM):
                e_y_perm = np.random.permutation(e_y)
                e_y_perm_norm = e_y_perm / np.linalg.norm(e_y_perm)
                # 矩阵乘法替代内层循环
                r_all = (e_X_valid.T @ e_y_perm_norm) / norm_X_valid
                max_abs_r_perms_2[p] = np.max(np.abs(r_all)) if len(r_all) > 0 else 0.0
            q2 = np.percentile(max_abs_r_perms_2, 95)

        # 3. 第 3 位 (Freedman-Lane + 图约束)
        if len(selected) >= 2:
            S_idx = [col_to_idx[selected[0]], col_to_idx[selected[1]]]
            X_S = np.column_stack([np.ones(n), X_mat[:, S_idx]])
            
            e_y = y - X_S @ np.linalg.lstsq(X_S, y, rcond=None)[0]
            
            mod1 = node_to_module.get(selected[0], -1)
            mod2 = node_to_module.get(selected[1], -1)
            valid_j = [j for j, c in enumerate(cand_cols) if node_to_module.get(c) not in (mod1, mod2)]
            
            X_valid = X_mat[:, valid_j]
            e_X_valid = X_valid - X_S @ np.linalg.lstsq(X_S, X_valid, rcond=None)[0]
            norm_X_valid = np.linalg.norm(e_X_valid, axis=0)
            
            max_abs_r_perms_3 = np.zeros(N_PERM)
            for p in range(N_PERM):
                e_y_perm = np.random.permutation(e_y)
                e_y_perm_norm = e_y_perm / np.linalg.norm(e_y_perm)
                r_all = (e_X_valid.T @ e_y_perm_norm) / norm_X_valid
                max_abs_r_perms_3[p] = np.max(np.abs(r_all)) if len(r_all) > 0 else 0.0
            q3 = np.percentile(max_abs_r_perms_3, 95)
            
        q_d_dict[d_name] = {1: q1, 2: q2, 3: q3}

    # 写回结果
    df_top3["q_d"] = 0.0
    df_top3["is_significant"] = False
    for d_name in dim_cols:
        for r_idx in [1, 2, 3]:
            mask = (df_top3["dimension"] == d_name) & (df_top3["rank"] == r_idx)
            q_val = q_d_dict[d_name][r_idx]
            df_top3.loc[mask, "q_d"] = q_val
            if pd.notna(q_val):
                df_top3.loc[mask, "is_significant"] = df_top3.loc[mask, "score"].abs() > q_val

    df_top3.to_csv(C.Q2_DIR / "top3_results_with_sig.csv", index=False)
    print("✅ Freedman-Lane 显著性检验完成！已加入图约束。")

if __name__ == "__main__":
    run_permutation_test()