# code/src/q2/stability.py
"""Q2 稳定性层：Bootstrap 重采样，计算入选频率 pi_c, pi_m 与 Jaccard 系数。"""
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
    print("开始 Bootstrap 稳定性检验")
    print("="*50)

    # 1. 加载数据
    ds = load_q2_dataset()
    X_mat = ds.X_rank.values  # 250 x 225
    Y_mat = ds.Y_rank.values  # 250 x 50
    cand_cols = ds.candidate_cols
    dim_cols = ds.Y_rank.columns
    n = len(ds.sample_ids)

    # 读取全样本的基准结果
    base_results_file = C.Q2_DIR / "top3_results_with_sig.csv"
    if not base_results_file.exists():
        raise FileNotFoundError("找不到基准结果文件，请先运行 inference.py")
    df_base = pd.read_csv(base_results_file)
    
    # 构造基准 Top-3 字典 {维度: [化合物1, 化合物2, 化合物3]}
    base_top3 = {}
    base_modules = {}
    for dim in dim_cols:
        sub = df_base[df_base["dimension"] == dim].sort_values("rank")
        base_top3[dim] = sub["compound"].tolist()
        base_modules[dim] = sub["module_id"].tolist()

    # 设置 Bootstrap 次数（预运行 100 次，正式可改 1000）
    B = 100 
    np.random.seed(C.SEED)

    print(f"正在进行 {B} 次 Bootstrap 重采样...")
    
    # 用于记录频率的字典
    # 结构: {维度: {化合物: [位置1次数, 位置2次数, 位置3次数]}}
    pi_c_counts = {dim: {c: [0, 0, 0] for c in cand_cols} for dim in dim_cols}
    # 用于记录模块频率的字典
    pi_m_counts = {dim: {} for dim in dim_cols}
    
    jaccard_scores = {dim: [] for dim in dim_cols}

    for b in range(B):
        if (b+1) % 10 == 0:
            print(f"  已完成 {b+1}/{B} 次...")
            
        # 重采样样本索引
        idx = np.random.choice(n, n, replace=True)
        X_boot = X_mat[idx, :]
        Y_boot = Y_mat[idx, :]
        
        # --- 重建图结构 ---
        # 重算 Spearman (秩上直接 Pearson)
        corr_boot = np.corrcoef(X_boot, rowvar=False)
        corr_boot = np.nan_to_num(corr_boot)
        
        # 层次聚类切树
        dist_boot = 1.0 - np.abs(corr_boot)
        np.fill_diagonal(dist_boot, 0.0)
        dist_boot = (dist_boot + dist_boot.T) / 2
        Z_boot = linkage(squareform(dist_boot, checks=False), method='average')
        labels_boot = fcluster(Z_boot, t=1.0 - C.TAU_MAIN, criterion='distance')
        node_to_module_boot = dict(zip(cand_cols, labels_boot))

        # --- 对每个维度跑前向选择 ---
        for d_idx, d_name in enumerate(dim_cols):
            y_b = Y_boot[:, d_idx]
            
            # 第1位
            r_1_all = (X_boot.T @ y_b) / n
            best_1_idx = np.argmax(np.abs(r_1_all))
            best_1_node = cand_cols[best_1_idx]
            
            selected = [best_1_node]
            pi_c_counts[d_name][best_1_node][0] += 1
            
            # 记录当前 bootstrap 的 Top-3
            boot_top3 = [best_1_node]
            boot_modules = [node_to_module_boot[best_1_node]]
            
            # 第2、3位
            for rank in [2, 3]:
                S_indices = [cand_cols.index(s) for s in selected]
                X_S = np.column_stack([np.ones(n), X_boot[:, S_indices]])
                
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
                    
            # 统计模块频率
            for m in boot_modules:
                pi_m_counts[d_name][m] = pi_m_counts[d_name].get(m, 0) + 1
                
            # 计算与基准 Top-3 的 Jaccard
            base_set = set(base_top3[d_name])
            boot_set = set(boot_top3)
            jaccard = len(base_set & boot_set) / len(base_set | boot_set) if len(base_set | boot_set) > 0 else 0
            jaccard_scores[d_name].append(jaccard)

    # 2. 汇总结果并合并
    print("\n正在汇总 Bootstrap 结果...")
    
    # 计算每个化合物进入 Top-3 的总频率（无论排第几）
    # 以及该化合物在所有 bootstrap 中的平均名次位置
    final_rows = []
    
    for _, row in df_base.iterrows():
        dim = row["dimension"]
        comp = row["compound"]
        rank = row["rank"]
        
        counts = pi_c_counts[dim].get(comp, [0, 0, 0])
        total_freq = sum(counts) / B
        pos_freq = counts[rank-1] / B
        
        # 计算模块频率
        m_id = row["module_id"]
        m_freq = pi_m_counts[dim].get(m_id, 0) / B
        
        # 计算该维度的平均 Jaccard
        mean_jaccard = np.mean(jaccard_scores[dim])
        
        final_rows.append({
            "dimension": dim,
            "rank": rank,
            "compound": comp,
            "module_id": m_id,
            "score": row["score"],
            "is_significant": row["is_significant"],
            "pi_c": total_freq,
            "pi_c_pos": pos_freq,
            "pi_m": m_freq,
            "jaccard": mean_jaccard
        })
        
    df_final = pd.DataFrame(final_rows)
    
    # 打上稳定性标签
    def get_stability_tag(pi_c):
        if pi_c >= 0.6: return "稳定"
        elif pi_c >= 1/3: return "中等"
        else: return "不稳定"
        
    df_final["stability"] = df_final["pi_c"].apply(get_stability_tag)
    
    # 保存最终文件
    out_file = C.Q2_DIR / "final_top3_with_stability.csv"
    df_final.to_csv(out_file, index=False)
    
    print(f"\n✅ Bootstrap 完成！结果已保存至: {out_file}")
    
    # 打印预览
    print("\n【预览】加入稳定性标签的结果 (部分维度):")
    for dim in ["A_malt_all", "lactic", "acetic", "t2n"]:
        print(f"\n--- {dim} (平均 Jaccard = {df_final[df_final['dimension']==dim]['jaccard'].iloc[0]:.3f}) ---")
        sub = df_final[df_final["dimension"] == dim].sort_values("rank")
        for _, row in sub.iterrows():
            print(f"Top {row['rank']}: {row['compound']:<25} pi_c={row['pi_c']:.2f}  [{row['stability']}]")

if __name__ == "__main__":
    run_bootstrap()