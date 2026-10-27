# code/src/q2/inference.py
"""Q2 推断层：严格 Freedman-Lane 置换。"""
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
import config as C
from data import load_q2_dataset

def run_permutation_test():
    print("="*50)
    print("计算置换零分布门槛（严格 Freedman-Lane）")
    print("="*50)

    ds = load_q2_dataset()
    X_mat, Y_mat = ds.X_rank.values, ds.Y_rank.values
    cand_cols, dim_cols = ds.candidate_cols, ds.Y_rank.columns
    n = len(ds.sample_ids)

    df_top3 = pd.read_csv(C.Q2_DIR / "top3_results.csv")
    np.random.seed(C.SEED)
    N_PERM = C.N_PERMUTATION
    
    # 剔除 barnyard
    dim_cols = [d for d in dim_cols if d != "barnyard"]
    q_d_dict = {}

    for d_idx, d_name in enumerate(dim_cols):
        print(f"  处理维度: {d_name}...")
        y = Y_mat[:, ds.Y_rank.columns.get_loc(d_name)]
        
        # Top-1 置换 (边际)
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
        
        # Top-2 置换 (Freedman-Lane)
        if len(selected) >= 1:
            S_idx = [cand_cols.index(selected[0])]
            X_S = np.column_stack([np.ones(n), X_mat[:, S_idx]])
            e_y = y - X_S @ np.linalg.lstsq(X_S, y, rcond=None)[0]
            valid_j = [j for j, c in enumerate(cand_cols) if c != selected[0]]
            
            max_abs_r_perms_2 = np.zeros(N_PERM)
            for p in range(N_PERM):
                e_y_perm = np.random.permutation(e_y)
                r_j = []
                for j in valid_j:
                    x_j = X_mat[:, j]
                    e_j = x_j - X_S @ np.linalg.lstsq(X_S, x_j, rcond=None)[0]
                    denom = np.linalg.norm(e_j) * np.linalg.norm(e_y_perm)
                    r_j.append(np.dot(e_j, e_y_perm) / denom if denom > 0 else 0.0)
                max_abs_r_perms_2[p] = np.max(np.abs(r_j))
            q2 = np.percentile(max_abs_r_perms_2, 95)

        # Top-3 置换 (Freedman-Lane)
        if len(selected) >= 2:
            S_idx = [cand_cols.index(selected[0]), cand_cols.index(selected[1])]
            X_S = np.column_stack([np.ones(n), X_mat[:, S_idx]])
            e_y = y - X_S @ np.linalg.lstsq(X_S, y, rcond=None)[0]
            valid_j = [j for j, c in enumerate(cand_cols) if c not in selected[:2]]
            
            max_abs_r_perms_3 = np.zeros(N_PERM)
            for p in range(N_PERM):
                e_y_perm = np.random.permutation(e_y)
                r_j = []
                for j in valid_j:
                    x_j = X_mat[:, j]
                    e_j = x_j - X_S @ np.linalg.lstsq(X_S, x_j, rcond=None)[0]
                    denom = np.linalg.norm(e_j) * np.linalg.norm(e_y_perm)
                    r_j.append(np.dot(e_j, e_y_perm) / denom if denom > 0 else 0.0)
                max_abs_r_perms_3[p] = np.max(np.abs(r_j))
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
    print("✅ Freedman-Lane 显著性检验完成！已剔除 barnyard。")

if __name__ == "__main__":
    run_permutation_test()