# code/src/q2/stability.py
"""Q2 稳定性层：修复 231 节点图重建的文本列报错隐患。"""
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import config as C
from data import load_q2_dataset

def run_bootstrap():
    print("="*50)
    print("Bootstrap 稳定性检验 (按品评团)")
    print("="*50)

    ds = load_q2_dataset()
    all_nodes = ds.candidate_cols + ds.annotation_cols
    X_mat_all, Y_mat_all = ds.X_rank.values, ds.Y_rank.values
    cand_cols = ds.candidate_cols
    
    # 剔除 barnyard
    valid_dim_indices = [i for i, d in enumerate(ds.Y_rank.columns) if d != "barnyard"]
    dim_cols = [ds.Y_rank.columns[i] for i in valid_dim_indices]
    Y_mat = Y_mat_all[:, valid_dim_indices]
    
    n = len(ds.sample_ids)
    npz_path = C.PANEL_NPZ_PATH if C.PANEL_NPZ_PATH.exists() else list(C.PROJECT_ROOT.rglob("q1_panels.npz"))[0]
    panels = np.load(npz_path)['panels']
    unique_panels = np.unique(panels)
    
    df_base = pd.read_csv(C.Q2_DIR / "top3_results_with_sig.csv")
    df_base = df_base[df_base["dimension"].isin(dim_cols)].copy()
    
    df_nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    orig_node_to_module = dict(zip(df_nodes["node"], df_nodes["module_id"]))

    B = C.N_BOOTSTRAP
    np.random.seed(C.SEED)
    pi_c_counts = {dim: {c: [0, 0, 0] for c in cand_cols} for dim in dim_cols}
    pi_m_counts = {dim: {} for dim in dim_cols}
    jaccard_scores = {dim: [] for dim in dim_cols}
    boot_stats = {dim: {r: [] for r in [1, 2, 3]} for dim in dim_cols}

    print(f"正在进行 {B} 次 Bootstrap...")
    for b in range(B):
        if (b+1) % max(1, B//10) == 0: print(f"  已完成 {b+1}/{B} 次...")
        sampled_panels = np.random.choice(unique_panels, size=len(unique_panels), replace=True)
        idx = []
        for p in sampled_panels: idx.extend(np.where(panels == p)[0])
        idx = np.array(idx)
        
        X_boot = pd.DataFrame(X_mat_all[idx, :]).rank(method="average").values
        X_boot = (X_boot - X_boot.mean(axis=0)) / X_boot.std(axis=0, ddof=0)
        Y_boot = pd.DataFrame(Y_mat[idx, :]).rank(method="average").values
        Y_boot = (Y_boot - Y_boot.mean(axis=0)) / Y_boot.std(axis=0, ddof=0)
        n_boot = len(idx)
        
        # 【关键修复】：强制取数值列并转换 NA，防止 corrcoef 爆炸
        chem_boot = ds.chem_raw.iloc[idx][all_nodes].apply(pd.to_numeric, errors='coerce')
        chem_boot = chem_boot.fillna(chem_boot.mean(axis=0))
        # 截断下限
        chem_boot = chem_boot.clip(lower=C.SIGNAL_FLOOR)
        
        corr_boot = np.nan_to_num(np.corrcoef(pd.DataFrame(chem_boot).rank(method="average").values, rowvar=False))
        dist_boot = 1.0 - np.abs(corr_boot)
        np.fill_diagonal(dist_boot, 0.0)
        Z_boot = linkage(squareform((dist_boot+dist_boot.T)/2, checks=False), method='average')
        labels_boot = fcluster(Z_boot, t=1.0 - C.TAU_MAIN, criterion='distance')
        node_to_module_boot = dict(zip(all_nodes, labels_boot))

        for d_idx, d_name in enumerate(dim_cols):
            y_b = Y_boot[:, d_idx]
            best_1_idx = np.argmax(np.abs((X_boot.T @ y_b) / n_boot))
            best_1_node = cand_cols[best_1_idx]
            
            selected = [best_1_node]
            pi_c_counts[d_name][best_1_node][0] += 1
            boot_top3 = [best_1_node]
            boot_modules = [orig_node_to_module[best_1_node]]
            
            for rank in [2, 3]:
                S_indices = [cand_cols.index(s) for s in selected]
                X_S = np.column_stack([np.ones(n_boot), X_boot[:, S_indices]])
                e_y = y_b - X_S @ np.linalg.lstsq(X_S, y_b, rcond=None)[0]
                
                best_score, best_idx = -1.0, -1
                selected_modules = {node_to_module_boot[s] for s in selected}
                
                for j_idx, j_node in enumerate(cand_cols):
                    if node_to_module_boot[j_node] in selected_modules: continue
                    x_j = X_boot[:, j_idx]
                    e_j = x_j - X_S @ np.linalg.lstsq(X_S, x_j, rcond=None)[0]
                    denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
                    r_j = np.dot(e_j, e_y) / denom if denom > 0 else 0.0
                    if abs(r_j) > best_score: best_score, best_idx = abs(r_j), j_idx
                
                if best_idx != -1:
                    best_node = cand_cols[best_idx]
                    selected.append(best_node)
                    boot_top3.append(best_node)
                    boot_modules.append(orig_node_to_module[best_node])
                    pi_c_counts[d_name][best_node][rank-1] += 1
                    
            for m in boot_modules: pi_m_counts[d_name][m] = pi_m_counts[d_name].get(m, 0) + 1
                
            base_set, boot_set = set(df_base[df_base["dimension"] == d_name]["compound"]), set(boot_top3)
            jaccard_scores[d_name].append(len(base_set & boot_set) / len(base_set | boot_set) if len(base_set | boot_set) > 0 else 0)
            
            full_selected = df_base[df_base["dimension"] == d_name].sort_values("rank")["compound"].tolist()
            for r_idx, comp in enumerate(full_selected):
                if comp not in cand_cols: continue
                c_idx = cand_cols.index(comp)
                if r_idx == 0:
                    boot_stats[d_name][1].append((X_boot[:, c_idx] @ y_b) / n_boot)
                elif r_idx in [1, 2]:
                    S = full_selected[:r_idx]
                    S_idx = [cand_cols.index(s) for s in S]
                    X_S = np.column_stack([np.ones(n_boot), X_boot[:, S_idx]])
                    e_y = y_b - X_S @ np.linalg.lstsq(X_S, y_b, rcond=None)[0]
                    x_c = X_boot[:, c_idx]
                    e_c = x_c - X_S @ np.linalg.lstsq(X_S, x_c, rcond=None)[0]
                    denom = np.linalg.norm(e_c) * np.linalg.norm(e_y)
                    boot_stats[d_name][r_idx+1].append(np.dot(e_c, e_y) / denom if denom > 0 else 0.0)

    final_rows = []
    for _, row in df_base.iterrows():
        dim, comp, rank = row["dimension"], row["compound"], row["rank"]
        counts = pi_c_counts[dim].get(comp, [0, 0, 0])
        stats = boot_stats[dim][rank]
        if len(stats) > 10:
            ci_l, ci_u = np.percentile(stats, 2.5), np.percentile(stats, 97.5)
            ci_not_zero = (ci_l > 0 and ci_u > 0) or (ci_l < 0 and ci_u < 0)
        else:
            ci_l, ci_u, ci_not_zero = np.nan, np.nan, False
            
        is_sig = row["is_significant"] and ci_not_zero
        
        final_rows.append({
            "dimension": dim, "rank": rank, "compound": comp, "module_id": row["module_id"],
            "score": row["score"], "is_significant": is_sig, "r_ci_lower": ci_l, "r_ci_upper": ci_u,
            "pi_c": sum(counts) / B, "pi_c_pos": counts[rank-1] / B,
            "pi_m": pi_m_counts[dim].get(row["module_id"], 0) / B, "jaccard": np.mean(jaccard_scores[dim])
        })
        
    df_final = pd.DataFrame(final_rows)
    df_final["stability"] = df_final["pi_c"].apply(lambda x: "稳定" if x >= 0.6 else ("中等" if x >= 1/3 else "不稳定"))
    df_final.to_csv(C.Q2_DIR / "final_top3_with_stability.csv", index=False)
    print(f"✅ Bootstrap 完成！")

if __name__ == "__main__":
    run_bootstrap()