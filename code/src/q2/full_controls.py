# code/src/q2/full_controls.py
"""Q2 高级对照方法：M2 (GroupKFold), M3 (5x5嵌套), 风格内分析。"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config as C
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, KFold

from data import load_q2_dataset

warnings.filterwarnings("ignore")

def jaccard(set1, set2):
    s1, s2 = set(set1), set(set2)
    return len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0.0

def forward_select(X_mat, y, cand_cols, node_to_module):
    n = len(y)
    r_1_all = (X_mat.T @ y) / n
    best_1_idx = np.argmax(np.abs(r_1_all))
    selected = [cand_cols[best_1_idx]]

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

def top3_from_importance(importance_scores, cand_cols, node_to_module):
    sorted_idx = np.argsort(-importance_scores)
    selected, selected_modules = [], set()
    for idx in sorted_idx:
        node = cand_cols[idx]
        m_id = node_to_module[node]
        if m_id in selected_modules: continue
        selected.append(node)
        selected_modules.add(m_id)
        if len(selected) == 3: break
    return selected

def run_full_controls():
    print("=" * 50)
    print("开始运行高级对照方法 (严格 49 维, GroupKFold M2, 嵌套 M3)")
    print("=" * 50)

    ds = load_q2_dataset()
    X_mat, Y_mat = ds.X_rank.values, ds.Y_rank.values
    cand_cols = ds.candidate_cols
    n = len(ds.sample_ids)

    # 【核心修复】：剔除 barnyard
    dim_cols = [d for d in ds.Y_rank.columns if d != "barnyard"]

    # 读取品评团数据
    npz_path = C.PANEL_NPZ_PATH if C.PANEL_NPZ_PATH.exists() else list(C.PROJECT_ROOT.rglob("q1_panels.npz"))[0]
    panels = np.load(npz_path)['panels']

    base_file = C.Q2_DIR / "final_top3_with_stability.csv"
    df_base = pd.read_csv(base_file)
    base_top3 = {
        dim: df_base[df_base["dimension"] == dim].sort_values("rank")["compound"].tolist()
        for dim in dim_cols
    }

    abs_corr = ds.corr.loc[cand_cols, cand_cols].abs()
    dist_matrix = 1.0 - abs_corr.values
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    labels = fcluster(linkage(squareform(dist_matrix, checks=False), method="average"), t=0.4, criterion="distance")
    node_to_module = dict(zip(cand_cols, labels))

    chem = ds.chem_raw
    if "tasting_category_fine" in chem.columns:
        style_dummies = np.column_stack([np.ones(n), pd.get_dummies(chem["tasting_category_fine"]).values])
        X_resid = X_mat - style_dummies @ np.linalg.lstsq(style_dummies, X_mat, rcond=None)[0]
        Y_resid = Y_mat - style_dummies @ np.linalg.lstsq(style_dummies, Y_mat, rcond=None)[0]
        X_resid = (X_resid - X_resid.mean(axis=0)) / X_resid.std(axis=0, ddof=0)
        Y_resid = (Y_resid - Y_resid.mean(axis=0)) / Y_resid.std(axis=0, ddof=0)
    else:
        X_resid, Y_resid = X_mat, Y_mat

    results = []
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=C.SEED)
    gkf = GroupKFold(n_splits=5)

    print(f"总共 {len(dim_cols)} 个维度，预计耗时较长...")

    for d_idx, d_name in enumerate(dim_cols):
        print(f"  [{d_idx+1}/{len(dim_cols)}] 正在计算: {d_name}")
        y = Y_mat[:, ds.Y_rank.columns.get_loc(d_name)]

        # M2: ElasticNet (GroupKFold)
        enet = ElasticNetCV(
            l1_ratio=0.5, cv=gkf.split(X_mat, y, groups=panels), random_state=C.SEED, n_jobs=-1, max_iter=2000
        )
        enet.fit(X_mat, y)
        m2_top3 = top3_from_importance(np.abs(enet.coef_), cand_cols, node_to_module)

        # M3: GBM + 5x5 嵌套 CV (外层 5 折计算置换重要度)
        m3_importance_cv = np.zeros(len(cand_cols))
        for train_idx, test_idx in outer_kf.split(X_mat):
            gbm = HistGradientBoostingRegressor(
                max_depth=3, min_samples_leaf=20, learning_rate=0.05, max_iter=200, random_state=C.SEED
            )
            gbm.fit(X_mat[train_idx], y[train_idx])
            perm = permutation_importance(
                gbm, X_mat[test_idx], y[test_idx], n_repeats=5, random_state=C.SEED, n_jobs=-1
            )
            m3_importance_cv += perm.importances_mean
        m3_top3 = top3_from_importance(m3_importance_cv / 5.0, cand_cols, node_to_module)

        # 风格内分析
        within_top3 = forward_select(X_resid, Y_resid[:, ds.Y_rank.columns.get_loc(d_name)], cand_cols, node_to_module)

        b_t3 = base_top3[d_name]
        results.append({
            "dimension": d_name,
            "jaccard_M2": jaccard(b_t3, m2_top3),
            "jaccard_M3": jaccard(b_t3, m3_top3),
            "jaccard_Within": jaccard(b_t3, within_top3),
            "M2_top1_match": b_t3[0] == m2_top3[0],
            "M3_top1_match": b_t3[0] == m3_top3[0],
            "Within_top1_match": b_t3[0] == within_top3[0],
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(C.Q2_DIR / "full_controls_results.csv", index=False)

    print("\n" + "=" * 50)
    print("✅ 高级对照方法计算完成！(严格 49 维)")
    print("=" * 50)
    print(f"主方法 vs M2 (ElasticNet)   : 平均 Jaccard = {df_res['jaccard_M2'].mean():.3f}")
    print(f"主方法 vs M3 (GBM重要度)    : 平均 Jaccard = {df_res['jaccard_M3'].mean():.3f}")
    print(f"主方法 vs 风格内分析          : 平均 Jaccard = {df_res['jaccard_Within'].mean():.3f}")
    print(f"\n结果已保存至: {C.Q2_DIR / 'full_controls_results.csv'}")

if __name__ == "__main__":
    run_full_controls()