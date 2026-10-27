# code/src/q2/full_controls.py
"""Q2 高级对照方法：M2 (GroupKFold), M3 (嵌套 CV), 风格内分析，合并最终表。"""
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
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV

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
        e_y = y - X_S @ np.linalg.lstsq(X_S, y, rcond=None)[0]

        best_score, best_idx = -1.0, -1
        selected_modules = {node_to_module[s] for s in selected}

        for j_idx, j_node in enumerate(cand_cols):
            if node_to_module[j_node] in selected_modules:
                continue
            x_j = X_mat[:, j_idx]
            e_j = x_j - X_S @ np.linalg.lstsq(X_S, x_j, rcond=None)[0]
            denom = np.linalg.norm(e_j) * np.linalg.norm(e_y)
            r_j = np.dot(e_j, e_y) / denom if denom > 0 else 0.0
            if abs(r_j) > best_score:
                best_score, best_idx = abs(r_j), j_idx
        if best_idx != -1:
            selected.append(cand_cols[best_idx])
    return selected


def top3_from_importance(importance_scores, cand_cols, node_to_module):
    sorted_idx = np.argsort(-importance_scores)
    selected, selected_modules = [], set()
    for idx in sorted_idx:
        node = cand_cols[idx]
        m_id = node_to_module[node]
        if m_id in selected_modules:
            continue
        selected.append(node)
        selected_modules.add(m_id)
        if len(selected) == 3:
            break
    return selected


def run_full_controls():
    print("=" * 50)
    print("开始运行高级对照方法 (严格 49 维, 嵌套 CV)")
    print("=" * 50)

    ds = load_q2_dataset()
    X_mat, Y_mat = ds.X_rank.values, ds.Y_rank.values
    cand_cols = ds.candidate_cols
    n = len(ds.sample_ids)

    dim_cols = ds.Y_rank.columns.tolist()

    # 品评团数据
    npz_path = C.PANEL_NPZ_PATH if C.PANEL_NPZ_PATH.exists() else list(C.PROJECT_ROOT.rglob("q1_panels.npz"))[0]
    panels = np.load(npz_path)['panels']

    # 读取基线
    base_file = C.Q2_DIR / "final_top3_with_stability.csv"
    df_base = pd.read_csv(base_file)
    base_top3 = {
        dim: df_base[df_base["dimension"] == dim].sort_values("rank")["compound"].tolist()
        for dim in dim_cols
    }

    # 【核心读取】：直接读取 M0 和 M1
    ctrl_path = C.Q2_DIR / "controls_sensitivity_results.csv"
    if not ctrl_path.exists():
        raise FileNotFoundError(f"缺少 {ctrl_path}，请先运行 controls_sensitivity.py")
        
    df_ctrl = pd.read_csv(ctrl_path)
    if "M0_top3" not in df_ctrl.columns or "M1_top3" not in df_ctrl.columns:
        raise ValueError("旧版 controls_sensitivity_results.csv 缺少 M0_top3 或 M1_top3 列！请重新运行最新版 controls_sensitivity.py。")
        
    m0_dict = dict(zip(df_ctrl["dimension"], df_ctrl["M0_top3"].str.split("|")))
    m1_dict = dict(zip(df_ctrl["dimension"], df_ctrl["M1_top3"].str.split("|")))

    # 构建全局模块字典
    abs_corr = ds.corr.loc[cand_cols, cand_cols].abs()
    dist_matrix = 1.0 - abs_corr.values
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    labels = fcluster(linkage(squareform(dist_matrix, checks=False), method="average"), t=1.0 - C.TAU_MAIN, criterion="distance")
    node_to_module = dict(zip(cand_cols, labels))

    # 风格内残差（加入 drop_first=True 防共线）
    chem = ds.chem_raw
    if "tasting_category_fine" in chem.columns:
        style_dummies = np.column_stack([np.ones(n), pd.get_dummies(chem["tasting_category_fine"], drop_first=True).values])
        X_resid = X_mat - style_dummies @ np.linalg.lstsq(style_dummies, X_mat, rcond=None)[0]
        Y_resid = Y_mat - style_dummies @ np.linalg.lstsq(style_dummies, Y_mat, rcond=None)[0]
        X_resid = (X_resid - X_resid.mean(axis=0)) / X_resid.std(axis=0, ddof=0)
        Y_resid = (Y_resid - Y_resid.mean(axis=0)) / Y_resid.std(axis=0, ddof=0)
    else:
        X_resid, Y_resid = X_mat, Y_mat

    results = []
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=C.SEED)
    gkf = GroupKFold(n_splits=5)

    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.05, 0.1],
        'max_iter': [100, 200]
    }

    print(f"总共 {len(dim_cols)} 个维度，预计耗时较长...")

    for d_idx, d_name in enumerate(dim_cols):
        print(f"  [{d_idx+1}/{len(dim_cols)}] 正在计算: {d_name}")
        y = Y_mat[:, ds.Y_rank.columns.get_loc(d_name)]

        m0_top3 = m0_dict[d_name]
        m1_top3 = m1_dict[d_name]

        # --- M2: ElasticNet (GroupKFold) ---
        enet = ElasticNetCV(
            l1_ratio=0.5, cv=gkf.split(X_mat, y, groups=panels), random_state=C.SEED, n_jobs=-1, max_iter=2000
        )
        enet.fit(X_mat, y)
        m2_top3 = top3_from_importance(np.abs(enet.coef_), cand_cols, node_to_module)

        # --- M3: GBM 嵌套 CV (外层 5 折计算置换重要度) ---
        m3_importance_cv = np.zeros(len(cand_cols))
        for train_idx, test_idx in outer_kf.split(X_mat):
            # 内层选参
            gbm = HistGradientBoostingRegressor(random_state=C.SEED)
            grid = GridSearchCV(gbm, param_grid, cv=3, n_jobs=-1)
            grid.fit(X_mat[train_idx], y[train_idx])
            best_gbm = grid.best_estimator_
            
            # 外层测试集计算置换重要度 (n_repeats=5)
            perm = permutation_importance(
                best_gbm, X_mat[test_idx], y[test_idx], n_repeats=5, random_state=C.SEED, n_jobs=-1
            )
            m3_importance_cv += perm.importances_mean
            
        m3_top3 = top3_from_importance(m3_importance_cv / 5.0, cand_cols, node_to_module)

        # --- 风格内分析 ---
        within_top3 = forward_select(X_resid, Y_resid[:, ds.Y_rank.columns.get_loc(d_name)], cand_cols, node_to_module)

        b_t3 = base_top3[d_name]
        
        results.append({
            "dimension": d_name,
            "main_top3": "|".join(b_t3),
            "M0_top3": "|".join(m0_top3),
            "M1_top3": "|".join(m1_top3),
            "M2_top3": "|".join(m2_top3),
            "M3_top3": "|".join(m3_top3),
            "Within_top3": "|".join(within_top3),
            "jaccard_M0": jaccard(b_t3, m0_top3),
            "jaccard_M1": jaccard(b_t3, m1_top3),
            "jaccard_M2": jaccard(b_t3, m2_top3),
            "jaccard_M3": jaccard(b_t3, m3_top3),
            "jaccard_Within": jaccard(b_t3, within_top3),
            "M3_top1_match": b_t3[0] == m3_top3[0],
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(C.Q2_DIR / "full_controls_results.csv", index=False)

    # ==========================================
    # 合并最终表：添加 in_M0~M3 标识列，追加 M3 Top-1
    # ==========================================
    df_rat = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    
    # 如果之前跑过这步，先剔除旧的 M3 追加行，只保留 main
    df_rat = df_rat[df_rat["method"] == "main"].copy()
    
    df_rat["in_M0"] = False
    df_rat["in_M1"] = False
    df_rat["in_M2"] = False
    df_rat["in_M3"] = False
    
    for _, row in df_res.iterrows():
        dim = row["dimension"]
        m0_set = set(row["M0_top3"].split("|"))
        m1_set = set(row["M1_top3"].split("|"))
        m2_set = set(row["M2_top3"].split("|"))
        m3_set = set(row["M3_top3"].split("|"))
        
        mask = df_rat["dimension"] == dim
        df_rat.loc[mask, "in_M0"] = df_rat.loc[mask, "compound"].isin(m0_set)
        df_rat.loc[mask, "in_M1"] = df_rat.loc[mask, "compound"].isin(m1_set)
        df_rat.loc[mask, "in_M2"] = df_rat.loc[mask, "compound"].isin(m2_set)
        df_rat.loc[mask, "in_M3"] = df_rat.loc[mask, "compound"].isin(m3_set)

    # 追加 M3 Top-1 不同的行
    m3_diff_dims = df_res[df_res["M3_top1_match"] == False]
    new_rows = []
    
    for _, row in m3_diff_dims.iterrows():
        dim = row["dimension"]
        m3_top1 = row["M3_top3"].split("|")[0]
        mod = node_to_module[m3_top1]
        
        new_row = {c: np.nan for c in df_rat.columns}
        new_row["dimension"] = dim
        new_row["rank"] = 1
        new_row["compound"] = m3_top1
        new_row["module_id"] = mod
        new_row["method"] = "M3"
        new_row["in_M0"] = m3_top1 in row["M0_top3"].split("|")
        new_row["in_M1"] = m3_top1 in row["M1_top3"].split("|")
        new_row["in_M2"] = m3_top1 in row["M2_top3"].split("|")
        new_row["in_M3"] = True
        new_rows.append(new_row)
        
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_rat = pd.concat([df_rat, df_new], ignore_index=True)

    df_rat.to_csv(C.Q2_DIR / "final_top3_with_rationality.csv", index=False)

    print("\n" + "=" * 50)
    print("✅ 高级对照方法计算完成！已更新最终表。")
    print("=" * 50)
    print(f"主方法 vs M2 (ElasticNet)   : 平均 Jaccard = {df_res['jaccard_M2'].mean():.3f}")
    print(f"主方法 vs M3 (嵌套 GBM)     : 平均 Jaccard = {df_res['jaccard_M3'].mean():.3f}")
    print(f"结果已保存至: {C.Q2_DIR / 'full_controls_results.csv'}")


if __name__ == "__main__":
    run_full_controls()