# code/src/q2/method_flags.py
"""追加对比方法标识 (GroupKFold M2, 嵌套 M3, 剔除 barnyard)。"""
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
import warnings
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, KFold
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
import config as C
from data import load_q2_dataset

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

def process_m3_nested_cv(d_idx, y, X_mat, ds_cand_cols):
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=C.SEED)
    m3_importance = np.zeros(len(ds_cand_cols))
    for train_idx, test_idx in outer_kf.split(X_mat):
        gbm = HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=20, learning_rate=0.05, max_iter=200, random_state=C.SEED)
        gbm.fit(X_mat[train_idx], y[train_idx])
        perm = permutation_importance(gbm, X_mat[test_idx], y[test_idx], n_repeats=5, random_state=C.SEED, n_jobs=1)
        m3_importance += perm.importances_mean
    m3_importance /= 5.0
    return ds_cand_cols[np.argsort(-m3_importance)[0]]

def add_flags():
    print("追加对比方法标识 (GroupKFold M2)...")
    ds = load_q2_dataset()
    X_mat, Y_mat, cand_cols = ds.X_rank.values, ds.Y_rank.values, ds.candidate_cols
    panels = np.load(C.PANEL_NPZ_PATH if C.PANEL_NPZ_PATH.exists() else list(C.PROJECT_ROOT.rglob("q1_panels.npz"))[0])['panels']
    
    final_path = C.Q2_DIR / "final_top3_with_rationality.csv"
    df_final = pd.read_csv(final_path)
    
    # 剔除 barnyard
    dim_cols = [d for d in ds.Y_rank.columns if d != "barnyard"]

    abs_corr = ds.corr.loc[cand_cols, cand_cols].abs()
    dist_matrix = 1.0 - abs_corr.values
    np.fill_diagonal(dist_matrix, 0.0)
    labels = fcluster(linkage(squareform(dist_matrix, checks=False), method="average"), t=0.4, criterion="distance")
    node_to_module = dict(zip(cand_cols, labels))

    m0_dict, m1_dict, m2_dict = {}, {}, {}
    for d_name in dim_cols:
        y = Y_mat[:, ds.Y_rank.columns.get_loc(d_name)]
        m0_dict[d_name] = marginal_select(X_mat, y, cand_cols, None)
        m1_dict[d_name] = marginal_select(X_mat, y, cand_cols, node_to_module)
        
        gkf = GroupKFold(n_splits=5)
        best_enet = ElasticNetCV(l1_ratio=0.5, cv=gkf.split(X_mat, y, groups=panels), random_state=C.SEED, n_jobs=-1, max_iter=2000)
        best_enet.fit(X_mat, y)
        sorted_idx = np.argsort(-np.abs(best_enet.coef_))
        sel, sel_mods = [], set()
        for idx in sorted_idx:
            node = cand_cols[idx]
            m_id = node_to_module[node]
            if m_id in sel_mods: continue
            sel_mods.add(m_id)
            sel.append(node)
            if len(sel) == 3: break
        m2_dict[d_name] = sel

    print("  -> 计算 M3 (GBM) 嵌套 CV...")
    m3_top1_list = Parallel(n_jobs=-1)(
        delayed(process_m3_nested_cv)(i, Y_mat[:, ds.Y_rank.columns.get_loc(d)], X_mat, cand_cols)
        for i, d in enumerate(dim_cols)
    )
    m3_dict = {d: [c] for d, c in zip(dim_cols, m3_top1_list)}

    def check_in(row, md): return row["compound"] in md.get(row["dimension"], [])

    df_final = df_final[df_final["dimension"].isin(dim_cols)].copy()
    df_final["in_M0"] = df_final.apply(lambda r: check_in(r, m0_dict), axis=1)
    df_final["in_M1"] = df_final.apply(lambda r: check_in(r, m1_dict), axis=1)
    df_final["in_M2"] = df_final.apply(lambda r: check_in(r, m2_dict), axis=1)
    df_final["in_M3"] = df_final.apply(lambda r: check_in(r, m3_dict), axis=1)

    # M3 并列行 (仅限 Top-1 不同的)
    m3_rows = []
    for d_name in dim_cols:
        m3_c = m3_top1_list[dim_cols.index(d_name)]
        main_c = df_final[(df_final["dimension"] == d_name) & (df_final["rank"] == 1)]["compound"].iloc[0]
        if m3_c != main_c:
            m3_rows.append({
                "dimension": d_name, "rank": 1, "compound": m3_c, "module_id": node_to_module.get(m3_c, -1),
                "score": np.nan, "is_significant": False, "pi_c": np.nan, "stability": "N/A",
                "sign": np.nan, "interpretation": "", "method": "M3", 
                "grade": df_final[df_final["dimension"] == d_name]["grade"].iloc[0],
                "flag_style_ci0": np.nan, "flag_breadth5": False, "flag_oav_lt1": np.nan, "covariation": False,
                "threshold_mean": np.nan, "oav_ge_1_ratio": np.nan, "n_dims_top3": np.nan, "chemical_name": "",
                "origin_paper": "", "off_flavors": "", "aging": ""
            })
            
    df_final = pd.concat([df_final, pd.DataFrame(m3_rows)], ignore_index=True)
    df_final = df_final.sort_values(by=["dimension", "method", "rank"])
    df_final.to_csv(final_path, index=False)
    print(f"✅ 追加完成！主方法 49x3 行，M3 并列 {len(m3_rows)} 行。")

if __name__ == "__main__":
    add_flags()