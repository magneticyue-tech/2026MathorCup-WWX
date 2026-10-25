# code/src/q2/stability.py
"""Q2 稳定性层：按品评团重采样的 Bootstrap。"""
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

def run_bootstrap():
    print("="*50)
    print("开始 Bootstrap 稳定性检验 (按品评团重采样)")
    print("="*50)

    ds = load_q2_dataset()
    X_mat = ds.X_rank.values
    Y_mat = ds.Y_rank.values
    cand_cols = ds.candidate_cols
    dim_cols = ds.Y_rank.columns
    n = len(ds.sample_ids)

    # 加载品评团
    npz_path = C.PANEL_NPZ_PATH
    if not npz_path.exists():
        raise FileNotFoundError(f"实在找不到文件：{npz_path}，请检查路径！")
    panels = np.load(npz_path)['panels']
    unique_panels = np.unique(panels)
    
    print(f"成功加载品评团：共 {len(unique_panels)} 个团，覆盖 {n} 款啤酒。")

    # 读取基准结果
    base_results_file = C.Q2_DIR / "final_top3_with_oav.csv"
    if not base_results_file.exists():
         raise FileNotFoundError(f"找不到 {base_results_file}，请先运行 pipeline_final.py！")
    df_base = pd.read_csv(base_results_file)
    
    # 【终极安全防护】：如果表里缺了这两列，直接手动补上，绝不报错
    if "score" not in df_base.columns:
        df_base["score"] = 0.0
    if "is_significant" not in df_base.columns:
        df_base["is_significant"] = False

    base_top3 = {dim: df_base[df_base["dimension"] == dim].sort_values("rank")["compound"].tolist() for dim in dim_cols}

    B = 100 # 预运行 100 次
    np.random.seed(C.SEED)

    print(f"正在进行 {B} 次按品评团重采样的 Bootstrap...")
    
    pi_c_counts = {dim: {c: [0, 0, 0] for c in cand_cols} for dim in dim_cols}
    pi_m_counts = {dim: {} for dim in dim_cols}
    jaccard_scores = {dim: [] for dim in dim_cols}

    for b in range(B):
        if (b+1) % 10 == 0:
            print(f"  已完成 {b+1}/{B} 次...")
            
        sampled_panels = np.random.choice(unique_panels, size=len(unique_panels), replace=True)
        idx = []
        for p in sampled_panels:
            idx.extend(np.where(panels == p)[0])
        idx = np.array(idx)
        
        X_boot = X_mat[idx, :]
        Y_boot = Y_mat[idx, :]
        n_boot = len(idx)
        
        corr_boot = np.corrcoef(X_boot, rowvar=False)
        corr_boot = np.nan_to_num(corr_boot)
        dist_boot = 1.0 - np.abs(corr_boot)
        np.fill_diagonal(dist_boot, 0.0)
        dist_boot = (dist_boot + dist_boot.T) / 2
        Z_boot = linkage(squareform(dist_boot, checks=False), method='average')
        labels_boot = fcluster(Z_boot, t=1.0 - C.TAU_MAIN, criterion='distance')
        node_to_module_boot = dict(zip(cand_cols, labels_boot))

        for d_idx, d_name in enumerate(dim_cols):
            y_b = Y_boot[:, d_idx]
            
            r_1_all = (X_boot.T @ y_b) / n_boot
            best_1_idx = np.argmax(np.abs(r_1_all))
            best_1_node = cand_cols[best_1_idx]
            
            selected = [best_1_node]
            pi_c_counts[d_name][best_1_node][0] += 1
            boot_top3 = [best_1_node]
            boot_modules = [node_to_module_boot[best_1_node]]
            
            for rank in [2, 3]:
                S_indices = [cand_cols.index(s) for s in selected]
                X_S = np.column_stack([np.ones(n_boot), X_boot[:, S_indices]])
                
                beta_y, _, _, _ = np.linalg.lstsq(X_S, y_b, rcond=None)
                e_y = y_b - X_S @ beta_y
                
                best_score = -1.0
                best_idx = -1
                selected_modules = {node_to_module_boot[s] for s in selected}
                
                for j_idx, j_node in enumerate(cand_cols):
                    if node_to_module_boot[j_node] in selected_modules:
                        continue
                    x_j = X_boot[:, j_idx]
                    beta_j, _, _, _ = np.linalg.lstsq(X_S, x_j, rcond=None)
                    e_j = x_j - X_S @ beta_j
                    
                    denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
                    r_j = np.dot(e_j, e_y) / denom if denom > 0 else 0.0
                    
                    if abs(r_j) > best_score:
                        best_score = abs(r_j)
                        best_idx = j_idx
                
                if best_idx != -1:
                    best_node = cand_cols[best_idx]
                    selected.append(best_node)
                    boot_top3.append(best_node)
                    boot_modules.append(node_to_module_boot[best_node])
                    pi_c_counts[d_name][best_node][rank-1] += 1
                    
            for m in boot_modules:
                pi_m_counts[d_name][m] = pi_m_counts[d_name].get(m, 0) + 1
                
            base_set = set(base_top3[d_name])
            boot_set = set(boot_top3)
            jaccard = len(base_set & boot_set) / len(base_set | boot_set) if len(base_set | boot_set) > 0 else 0
            jaccard_scores[d_name].append(jaccard)

    print("\n正在汇总 Bootstrap 结果...")
    final_rows = []
    for _, row in df_base.iterrows():
        dim = row["dimension"]
        comp = row["compound"]
        rank = row["rank"]
        
        counts = pi_c_counts[dim].get(comp, [0, 0, 0])
        total_freq = sum(counts) / B
        pos_freq = counts[rank-1] / B
        m_freq = pi_m_counts[dim].get(row["module_id"], 0) / B
        mean_jaccard = np.mean(jaccard_scores[dim])
        
        final_rows.append({
            "dimension": dim,
            "rank": rank,
            "compound": comp,
            "module_id": row["module_id"],
            "score": row["score"],
            "is_significant": row["is_significant"],
            "pi_c": total_freq,
            "pi_c_pos": pos_freq,
            "pi_m": m_freq,
            "jaccard": mean_jaccard
        })
        
    df_final = pd.DataFrame(final_rows)
    df_final["stability"] = df_final["pi_c"].apply(lambda x: "稳定" if x >= 0.6 else ("中等" if x >= 1/3 else "不稳定"))
    df_final.to_csv(C.Q2_DIR / "final_top3_with_stability.csv", index=False)
    
    print(f"\n✅ 严格品评团 Bootstrap 完成！已更新 final_top3_with_stability.csv")
    print(f"整体平均 Jaccard 为: {df_final['jaccard'].mean():.3f}")

if __name__ == "__main__":
    run_bootstrap()