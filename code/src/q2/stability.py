# code/src/q2/stability.py
"""Q2 稳定性层：按品评团重采样 Bootstrap，并输出 95% CI 与双重显著性。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import config as C
from data import load_q2_dataset

def run_bootstrap():
    print("="*50)
    print("Bootstrap 稳定性检验 (含 95% CI 与双重显著性判定)")
    print("="*50)

    ds = load_q2_dataset()
    X_mat, Y_mat = ds.X_rank.values, ds.Y_rank.values
    cand_cols, dim_cols = ds.candidate_cols, ds.Y_rank.columns
    n = len(ds.sample_ids)

    npz_path = C.PANEL_NPZ_PATH
    if not npz_path.exists():
        found_paths = list(C.PROJECT_ROOT.rglob("q1_panels.npz"))
        if found_paths: npz_path = found_paths[0]
        else: raise FileNotFoundError("找不到 q1_panels.npz！")
            
    panels = np.load(npz_path)['panels']
    unique_panels = np.unique(panels)
    
    base_results_file = C.Q2_DIR / "top3_results_with_sig.csv"
    df_base = pd.read_csv(base_results_file)
    base_top3 = {dim: df_base[df_base["dimension"] == dim].sort_values("rank")["compound"].tolist() for dim in dim_cols}

    df_nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    orig_node_to_module = dict(zip(df_nodes["node"], df_nodes["module_id"]))

    B = 1000 # 预运行，正式改1000
    np.random.seed(C.SEED)
    pi_c_counts = {dim: {c: [0, 0, 0] for c in cand_cols} for dim in dim_cols}
    pi_m_counts = {dim: {} for dim in dim_cols}
    jaccard_scores = {dim: [] for dim in dim_cols}
    
    # 【修复】：用于记录每个化合物在 Bootstrap 中的 r 值
    r_boot_values = {dim: {c: [] for c in cand_cols} for dim in dim_cols}

    print(f"正在进行 {B} 次按品评团重采样的 Bootstrap...")
    for b in range(B):
        if (b+1) % 20 == 0: print(f"  已完成 {b+1}/{B} 次...")
        sampled_panels = np.random.choice(unique_panels, size=len(unique_panels), replace=True)
        idx = []
        for p in sampled_panels: idx.extend(np.where(panels == p)[0])
        idx = np.array(idx)
        
        # 【修复】：子集重新算秩和标准化
        X_boot = pd.DataFrame(X_mat[idx, :]).rank(method="average").values
        Y_boot = pd.DataFrame(Y_mat[idx, :]).rank(method="average").values
        X_boot = (X_boot - X_boot.mean(axis=0)) / X_boot.std(axis=0, ddof=0)
        Y_boot = (Y_boot - Y_boot.mean(axis=0)) / Y_boot.std(axis=0, ddof=0)
        n_boot = len(idx)
        
        corr_boot = np.nan_to_num(np.corrcoef(X_boot, rowvar=False))
        dist_boot = 1.0 - np.abs(corr_boot)
        np.fill_diagonal(dist_boot, 0.0)
        dist_boot = (dist_boot + dist_boot.T) / 2
        Z_boot = linkage(squareform(dist_boot, checks=False), method='average')
        labels_boot = fcluster(Z_boot, t=1.0 - C.TAU_MAIN, criterion='distance')
        node_to_module_boot = dict(zip(cand_cols, labels_boot))

        for d_idx, d_name in enumerate(dim_cols):
            y_b = Y_boot[:, d_idx]
            best_1_idx = np.argmax(np.abs((X_boot.T @ y_b) / n_boot))
            best_1_node = cand_cols[best_1_idx]
            
            selected = [best_1_node]
            pi_c_counts[d_name][best_1_node][0] += 1
            boot_top3 = [best_1_node]
            boot_modules = [orig_node_to_module[best_1_node]]
            
            # 记录 Top-1 的 Bootstrap r 值
            r_best1 = (X_boot[:, best_1_idx] @ y_b) / n_boot
            r_boot_values[d_name][best_1_node].append(r_best1)
            
            for rank in [2, 3]:
                S_indices = [cand_cols.index(s) for s in selected]
                X_S = np.column_stack([np.ones(n_boot), X_boot[:, S_indices]])
                e_y = y_b - X_S @ np.linalg.lstsq(X_S, y_b, rcond=None)[0]
                
                best_score, best_idx, best_r = -1.0, -1, 0.0
                selected_modules = {node_to_module_boot[s] for s in selected}
                
                for j_idx, j_node in enumerate(cand_cols):
                    if node_to_module_boot[j_node] in selected_modules: continue
                    x_j = X_boot[:, j_idx]
                    e_j = x_j - X_S @ np.linalg.lstsq(X_S, x_j, rcond=None)[0]
                    denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
                    r_j = np.dot(e_j, e_y) / denom if denom > 0 else 0.0
                    if abs(r_j) > best_score: best_score, best_idx, best_r = abs(r_j), j_idx, r_j
                
                if best_idx != -1:
                    best_node = cand_cols[best_idx]
                    selected.append(best_node)
                    boot_top3.append(best_node)
                    boot_modules.append(orig_node_to_module[best_node])
                    pi_c_counts[d_name][best_node][rank-1] += 1
                    # 记录 2/3 位的 Bootstrap 偏相关 r 值
                    r_boot_values[d_name][best_node].append(best_r)
                    
            for m in boot_modules: pi_m_counts[d_name][m] = pi_m_counts[d_name].get(m, 0) + 1
                
            base_set, boot_set = set(base_top3[d_name]), set(boot_top3)
            jaccard_scores[d_name].append(len(base_set & boot_set) / len(base_set | boot_set) if len(base_set | boot_set) > 0 else 0)

    final_rows = []
    for _, row in df_base.iterrows():
        dim, comp, rank = row["dimension"], row["compound"], row["rank"]
        counts = pi_c_counts[dim].get(comp, [0, 0, 0])
        
        # 【修复】：计算 Bootstrap CI 和双重显著性
        r_vals = r_boot_values[dim].get(comp, [])
        if len(r_vals) > 10:
            ci_lower = np.percentile(r_vals, 2.5)
            ci_upper = np.percentile(r_vals, 97.5)
            # CI 不含 0 意味着上下界同号（都大于0，或都小于0）
            ci_not_contain_zero = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
        else:
            ci_lower, ci_upper = np.nan, np.nan
            ci_not_contain_zero = False
            
        # 双重条件判定：超过置换门槛 AND CI不含0
        is_sig = row["is_significant"] and ci_not_contain_zero

        final_rows.append({
            "dimension": dim, "rank": rank, "compound": comp, "module_id": row["module_id"],
            "score": row["score"], "is_significant": is_sig,
            "r_ci_lower": ci_lower, "r_ci_upper": ci_upper,
            "pi_c": sum(counts) / B, "pi_c_pos": counts[rank-1] / B,
            "pi_m": pi_m_counts[dim].get(row["module_id"], 0) / B, "jaccard": np.mean(jaccard_scores[dim])
        })
        
    df_final = pd.DataFrame(final_rows)
    df_final["stability"] = df_final["pi_c"].apply(lambda x: "稳定" if x >= 0.6 else ("中等" if x >= 1/3 else "不稳定"))
    df_final.to_csv(C.Q2_DIR / "final_top3_with_stability.csv", index=False)
    print(f"✅ Bootstrap 完成！已加入双条件显著性判定。")

if __name__ == "__main__":
    run_bootstrap()