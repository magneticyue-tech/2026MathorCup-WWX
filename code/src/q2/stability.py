# code/src/q2/stability.py
"""Q2 稳定性层：修复重抽样秩计算、图重建红线问题。"""
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
from data import load_q2_dataset, rank_standardize


def run_bootstrap():
    print("=" * 50)
    print("Bootstrap 稳定性检验 (按品评团)")
    print("=" * 50)

    ds = load_q2_dataset()
    all_nodes = ds.candidate_cols + ds.annotation_cols
    cand_cols = ds.candidate_cols
    
    # 强制剔除 barnyard 维度，Y_rank 已是 49 维
    dim_cols = ds.Y_rank.columns.tolist()
    
    npz_path = C.PANEL_NPZ_PATH if C.PANEL_NPZ_PATH.exists() else list(C.PROJECT_ROOT.rglob("q1_panels.npz"))[0]
    panels = np.load(npz_path)['panels']
    unique_panels = np.unique(panels)
    if len(unique_panels) != 63:
        print(f"⚠️ 警告：当前品评团数量为 {len(unique_panels)}，工单要求 63 团。")
    
    df_base = pd.read_csv(C.Q2_DIR / "top3_results_with_sig.csv")
    
    df_nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    orig_node_to_module = dict(zip(df_nodes["node"], df_nodes["module_id"]))
    col_to_idx = {c: i for i, c in enumerate(cand_cols)}

    B = C.N_BOOTSTRAP
    np.random.seed(C.SEED)
    pi_c_counts = {dim: {c: [0, 0, 0] for c in cand_cols} for dim in dim_cols}
    pi_m_counts = {dim: {} for dim in dim_cols}
    jaccard_scores = {dim: [] for dim in dim_cols}
    boot_stats = {dim: {r: [] for r in [1, 2, 3]} for dim in dim_cols}

    print(f"正在进行 {B} 次 Bootstrap...")
    for b in range(B):
        if (b + 1) % max(1, B // 10) == 0: 
            print(f"  已完成 {b + 1}/{B} 次...")
            
        sampled_panels = np.random.choice(unique_panels, size=len(unique_panels), replace=True)
        idx = []
        for p in sampled_panels: 
            idx.extend(np.where(panels == p)[0])
        idx = np.array(idx)
        
        # 【红线修复1】：必须从原始数据重抽，并重新处理缺失值和秩
        chem_boot_raw = ds.chem_raw.iloc[idx][cand_cols]
        X_boot = rank_standardize(chem_boot_raw, fill_na_mean=True, clip_floor=True).values
        
        sens_boot_raw = ds.sensory_raw.iloc[idx][dim_cols]
        Y_boot = rank_standardize(sens_boot_raw, fill_na_mean=False, clip_floor=False).values
        
        n_boot = len(idx)
        
        # 【红线修复2】：逐对删除 Spearman 建图
        chem_boot_all = ds.chem_raw.iloc[idx][all_nodes]
        corr_boot = chem_boot_all.corr(method='spearman').values
        corr_boot = np.nan_to_num(corr_boot, nan=0.0)  # 处理部分组合样本不足导致的 NaN
        
        dist_boot = 1.0 - np.abs(corr_boot)
        np.fill_diagonal(dist_boot, 0.0)
        Z_boot = linkage(squareform((dist_boot + dist_boot.T) / 2, checks=False), method='average')
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
                S_indices = [col_to_idx[s] for s in selected]
                X_S = np.column_stack([np.ones(n_boot), X_boot[:, S_indices]])
                e_y = y_b - X_S @ np.linalg.lstsq(X_S, y_b, rcond=None)[0]
                
                best_score, best_idx = -1.0, -1
                selected_modules = {node_to_module_boot[s] for s in selected}
                
                for j_idx, j_node in enumerate(cand_cols):
                    if node_to_module_boot[j_node] in selected_modules: 
                        continue
                    x_j = X_boot[:, j_idx]
                    e_j = x_j - X_S @ np.linalg.lstsq(X_S, x_j, rcond=None)[0]
                    denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
                    r_j = np.dot(e_j, e_y) / denom if denom > 0 else 0.0
                    if abs(r_j) > best_score: 
                        best_score, best_idx = abs(r_j), j_idx
                
                if best_idx != -1:
                    best_node = cand_cols[best_idx]
                    selected.append(best_node)
                    boot_top3.append(best_node)
                    boot_modules.append(orig_node_to_module[best_node])
                    pi_c_counts[d_name][best_node][rank - 1] += 1
                    
            # 【红线修复3】：模块去重后统计，防止频次超过 100%
            for m in set(boot_modules): 
                pi_m_counts[d_name][m] = pi_m_counts[d_name].get(m, 0) + 1
                
            base_set = set(df_base[df_base["dimension"] == d_name]["compound"])
            boot_set = set(boot_top3)
            jaccard_scores[d_name].append(
                len(base_set & boot_set) / len(base_set | boot_set) if len(base_set | boot_set) > 0 else 0
            )
            
            # 计算固定的全样本入选项的分布区间
            full_selected = df_base[df_base["dimension"] == d_name].sort_values("rank")["compound"].tolist()
            for r_idx, comp in enumerate(full_selected):
                if comp not in col_to_idx: 
                    continue
                c_idx = col_to_idx[comp]
                if r_idx == 0:
                    boot_stats[d_name][1].append((X_boot[:, c_idx] @ y_b) / n_boot)
                elif r_idx in [1, 2]:
                    S = full_selected[:r_idx]
                    S_idx = [col_to_idx[s] for s in S]
                    X_S = np.column_stack([np.ones(n_boot), X_boot[:, S_idx]])
                    e_y = y_b - X_S @ np.linalg.lstsq(X_S, y_b, rcond=None)[0]
                    x_c = X_boot[:, c_idx]
                    e_c = x_c - X_S @ np.linalg.lstsq(X_S, x_c, rcond=None)[0]
                    denom = np.linalg.norm(e_c) * np.linalg.norm(e_y)
                    boot_stats[d_name][r_idx + 1].append(np.dot(e_c, e_y) / denom if denom > 0 else 0.0)

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
            
        # 【红线】：双重显著性必须同时满足“置换过门槛”和“Bootstrap CI 不含 0”
        is_sig = row["is_significant"] and ci_not_zero
        
        final_rows.append({
            "dimension": dim, 
            "rank": rank, 
            "compound": comp, 
            "module_id": row["module_id"],
            "score": row["score"], 
            "is_significant": is_sig, 
            "r_ci_lower": ci_l, 
            "r_ci_upper": ci_u,
            "pi_c": sum(counts) / B, 
            "pi_c_pos": counts[rank - 1] / B,
            "pi_m": pi_m_counts[dim].get(row["module_id"], 0) / B, 
            "jaccard": np.mean(jaccard_scores[dim])
        })
        
    df_final = pd.DataFrame(final_rows)
    df_final["stability"] = df_final["pi_c"].apply(
        lambda x: "稳定" if x >= C.STABLE_THRESHOLD else ("中等" if x >= C.MEDIUM_THRESHOLD else "不稳定")
    )
    
    df_final.to_csv(C.Q2_DIR / "final_top3_with_stability.csv", index=False)
    print(f"✅ Bootstrap 完成！")


if __name__ == "__main__":
    run_bootstrap()