# code/src/q2/inference.py
"""Q2 推断层：严格实现 Freedman-Lane 置换。"""
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

def run_permutation_test():
    print("="*50)
    print("计算置换零分布门槛（Freedman-Lane）")
    print("="*50)

    ds = load_q2_dataset()
    X_mat = ds.X_rank.values
    Y_mat = ds.Y_rank.values
    cand_cols = ds.candidate_cols
    dim_cols = ds.Y_rank.columns
    n = len(ds.sample_ids)

    df_top3 = pd.read_csv(C.Q2_DIR / "top3_results.csv")
    np.random.seed(C.SEED)
    N_PERM = 1000 

    q_d_dict = {}

    for d_idx, d_name in enumerate(dim_cols):
        print(f"  处理维度: {d_name}...")
        y = Y_mat[:, d_idx]
        
        # 获取全样本的 S 集合
        top3_rows = df_top3[df_top3["dimension"] == d_name].sort_values("rank")
        selected = top3_rows["compound"].tolist()
        
        # 1. 计算 Top-1 门槛
        r_1_all = (X_mat.T @ y) / n
        max_abs_r_perms = np.zeros(N_PERM)
        
        for p in range(N_PERM):
            y_perm = np.random.permutation(y)
            r_perms = (X_mat.T @ y_perm) / n
            max_abs_r_perms[p] = np.max(np.abs(r_perms))
        q1 = np.percentile(max_abs_r_perms, 95)

        # 2. 计算 Top-2 的 Freedman-Lane 门槛
        S_indices = [cand_cols.index(selected[0])]
        X_S = np.column_stack([np.ones(n), X_mat[:, S_indices]])
        beta_y, _, _, _ = np.linalg.lstsq(X_S, y, rcond=None)
        e_y = y - X_S @ beta_y
        
        # 剩余可选的索引
        module_1 = top3_rows.iloc[0]["module_id"]
        df_nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
        node_to_module = dict(zip(df_nodes["node"], df_nodes["module_id"]))
        valid_j = [j for j, c in enumerate(cand_cols) if node_to_module[c] != module_1]
        
        max_abs_r_perms_2 = np.zeros(N_PERM)
        for p in range(N_PERM):
            e_y_perm = np.random.permutation(e_y)
            # 在可选化合物上算偏相关
            r_j = []
            for j in valid_j:
                x_j = X_mat[:, j]
                beta_j, _, _, _ = np.linalg.lstsq(X_S, x_j, rcond=None)
                e_j = x_j - X_S @ beta_j
                denom = np.linalg.norm(e_j) * np.linalg.norm(e_y_perm)
                r_j.append(np.dot(e_j, e_y_perm) / denom if denom > 0 else 0.0)
            max_abs_r_perms_2[p] = np.max(np.abs(r_j))
        q2 = np.percentile(max_abs_r_perms_2, 95)
        
        # 3. 计算 Top-3 门槛 (以此类推)
        S_indices_3 = [cand_cols.index(selected[0]), cand_cols.index(selected[1])]
        X_S3 = np.column_stack([np.ones(n), X_mat[:, S_indices_3]])
        beta_y3, _, _, _ = np.linalg.lstsq(X_S3, y, rcond=None)
        e_y3 = y - X_S3 @ beta_y3
        
        module_2 = top3_rows.iloc[1]["module_id"]
        valid_j_3 = [j for j, c in enumerate(cand_cols) if node_to_module[c] not in (module_1, module_2)]
        
        max_abs_r_perms_3 = np.zeros(N_PERM)
        for p in range(N_PERM):
            e_y_perm3 = np.random.permutation(e_y3)
            r_j_3 = []
            for j in valid_j_3:
                x_j = X_mat[:, j]
                beta_j, _, _, _ = np.linalg.lstsq(X_S3, x_j, rcond=None)
                e_j = x_j - X_S3 @ beta_j
                denom = np.linalg.norm(e_j) * np.linalg.norm(e_y_perm3)
                r_j_3.append(np.dot(e_j, e_y_perm3) / denom if denom > 0 else 0.0)
            max_abs_r_perms_3[p] = np.max(np.abs(r_j_3))
        q3 = np.percentile(max_abs_r_perms_3, 95)
        
        q_d_dict[d_name] = [q1, q2, q3]

    # 写回结果
    df_top3["q_d"] = 0.0
    df_top3["is_significant"] = False
    for d_name in dim_cols:
        q_vals = q_d_dict[d_name]
        for r_idx in [1, 2, 3]:
            mask = (df_top3["dimension"] == d_name) & (df_top3["rank"] == r_idx)
            df_top3.loc[mask, "q_d"] = q_vals[r_idx-1]
            df_top3.loc[mask, "is_significant"] = df_top3.loc[mask, "score"].abs() > q_vals[r_idx-1]

    df_top3.to_csv(C.Q2_DIR / "top3_results_with_sig.csv", index=False)
    print("✅ Freedman-Lane 显著性检验完成！已保存。")

if __name__ == "__main__":
    run_permutation_test()