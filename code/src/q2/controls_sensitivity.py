# code/src/q2/controls_sensitivity.py
"""Q2 对照方法与敏感性分析：M0, M1 以及 tau 的 Jaccard 相似度 (严格 49 维)。"""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config as C
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from data import load_q2_dataset

def jaccard(set1, set2):
    s1, s2 = set(set1), set(set2)
    if len(s1 | s2) == 0: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def get_modules(dist_matrix, tau):
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")
    labels = fcluster(Z, t=1.0 - tau, criterion="distance")
    return labels

def forward_select(X_mat, y, cand_cols, node_to_module):
    n = len(y)
    r_1_all = (X_mat.T @ y) / n
    best_1_idx = np.argmax(np.abs(r_1_all))
    best_1_node = cand_cols[best_1_idx]
    selected = [best_1_node]

    for rank in [2, 3]:
        S_indices = [cand_cols.index(s) for s in selected]
        X_S = np.column_stack([np.ones(n), X_mat[:, S_indices]])
        beta_y, _, _, _ = np.linalg.lstsq(X_S, y, rcond=None)
        e_y = y - X_S @ beta_y

        best_score, best_idx = -1.0, -1
        selected_modules = {node_to_module[s] for s in selected}

        for j_idx, j_node in enumerate(cand_cols):
            if node_to_module[j_node] in selected_modules: continue
            x_j = X_mat[:, j_idx]
            beta_j, _, _, _ = np.linalg.lstsq(X_S, x_j, rcond=None)
            e_j = x_j - X_S @ beta_j
            denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
            r_j = np.dot(e_j, e_y) / denom if denom > 0 else 0.0
            if abs(r_j) > best_score: best_score, best_idx = abs(r_j), j_idx

        if best_idx != -1: selected.append(cand_cols[best_idx])
    return selected

def marginal_select(X_mat, y, cand_cols, node_to_module=None):
    n = len(y)
    r_all = (X_mat.T @ y) / n
    sorted_indices = np.argsort(-np.abs(r_all))
    selected, selected_modules = [], set()

    for idx in sorted_indices:
        node = cand_cols[idx]
        if node_to_module is not None:
            m_id = node_to_module[node]
            if m_id in selected_modules: continue
            selected_modules.add(m_id)
        selected.append(node)
        if len(selected) == 3: break
    return selected

def run_controls_and_sensitivity():
    print("=" * 50)
    print("开始对照方法与敏感性分析 (严格 49 维)")
    print("=" * 50)

    ds = load_q2_dataset()
    X_mat, Y_mat = ds.X_rank.values, ds.Y_rank.values
    cand_cols = ds.candidate_cols
    
    # 【严格修复】：剔除 barnyard
    dim_cols = [d for d in ds.Y_rank.columns if d != "barnyard"]

    # 读取主方法结果 (需确保读取的是剔除 barnyard 后的表)
    base_file = C.Q2_DIR / "final_top3_with_stability.csv"
    if not base_file.exists():
        raise FileNotFoundError(f"缺少主方法基线表 {base_file}，请先运行 stability.py")
    df_base = pd.read_csv(base_file)
    
    base_top3 = {
        dim: df_base[df_base["dimension"] == dim].sort_values("rank")["compound"].tolist()
        for dim in dim_cols
    }

    abs_corr = ds.corr.loc[cand_cols, cand_cols].abs()
    dist_matrix = 1.0 - abs_corr.values
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    node_to_module_06 = dict(zip(cand_cols, get_modules(dist_matrix, 0.6)))
    node_to_module_05 = dict(zip(cand_cols, get_modules(dist_matrix, 0.5)))
    node_to_module_07 = dict(zip(cand_cols, get_modules(dist_matrix, 0.7)))

    results = []

    print("正在遍历维度进行对比计算...")
    for d_name in dim_cols:
        y = Y_mat[:, ds.Y_rank.columns.get_loc(d_name)]

        m0_top3 = marginal_select(X_mat, y, cand_cols, node_to_module=None)
        m1_top3 = marginal_select(X_mat, y, cand_cols, node_to_module=node_to_module_06)
        tau05_top3 = forward_select(X_mat, y, cand_cols, node_to_module_05)
        tau07_top3 = forward_select(X_mat, y, cand_cols, node_to_module_07)

        base_t3 = base_top3[d_name]
        results.append({
            "dimension": d_name,
            "jaccard_M0": jaccard(base_t3, m0_top3),
            "jaccard_M1": jaccard(base_t3, m1_top3),
            "jaccard_tau05": jaccard(base_t3, tau05_top3),
            "jaccard_tau07": jaccard(base_t3, tau07_top3),
            "M0_top1_match": base_t3[0] == m0_top3[0],
            "M1_top1_match": base_t3[0] == m1_top3[0],
            "tau05_top1_match": base_t3[0] == tau05_top3[0],
            "tau07_top1_match": base_t3[0] == tau07_top3[0],
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(C.Q2_DIR / "controls_sensitivity_results.csv", index=False)

    print("\n✅ 计算完成！汇总结果如下 (49维平均):")
    print(f"主方法 vs M0 (无图边际)  : 平均 Jaccard = {df_res['jaccard_M0'].mean():.3f}, Top1相同比例 = {df_res['M0_top1_match'].mean():.2f}")
    print(f"主方法 vs M1 (图约束边际): 平均 Jaccard = {df_res['jaccard_M1'].mean():.3f}, Top1相同比例 = {df_res['M1_top1_match'].mean():.2f}")
    print(f"主方法 vs tau=0.5       : 平均 Jaccard = {df_res['jaccard_tau05'].mean():.3f}, Top1相同比例 = {df_res['tau05_top1_match'].mean():.2f}")
    print(f"主方法 vs tau=0.7       : 平均 Jaccard = {df_res['jaccard_tau07'].mean():.3f}, Top1相同比例 = {df_res['tau07_top1_match'].mean():.2f}")
    print(f"\n结果已保存至: {C.Q2_DIR / 'controls_sensitivity_results.csv'}")

if __name__ == "__main__":
    run_controls_and_sensitivity()