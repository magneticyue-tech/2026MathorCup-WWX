# code/src/q2/method_flags.py
"""为最终表追加 M0/M1/M2/M3 标识。"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import ElasticNetCV

warnings.filterwarnings("ignore") # 屏蔽满屏的红色收敛警告

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config as C
from data import load_q2_dataset

def marginal_select(X_mat, y, cand_cols, node_to_module):
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

def add_flags():
    print("追加对比方法标识 (包含完整的 M2)...")
    ds = load_q2_dataset()
    X_mat, Y_mat, cand_cols = ds.X_rank.values, ds.Y_rank.values, ds.candidate_cols
    final_path = C.Q2_DIR / "final_top3_with_rationality.csv"
    df_final = pd.read_csv(final_path)

    abs_corr = ds.corr.loc[cand_cols, cand_cols].abs()
    dist_matrix = 1.0 - abs_corr.values
    np.fill_diagonal(dist_matrix, 0.0)
    labels = fcluster(linkage(squareform(dist_matrix, checks=False), method="average"), t=0.4, criterion="distance")
    node_to_module = dict(zip(cand_cols, labels))

    m0_dict, m1_dict, m2_dict = {}, {}, {}
    for d_idx, d_name in enumerate(ds.Y_rank.columns):
        y = Y_mat[:, d_idx]
        m0_dict[d_name] = marginal_select(X_mat, y, cand_cols, node_to_module=None)
        m1_dict[d_name] = marginal_select(X_mat, y, cand_cols, node_to_module=node_to_module)
        
        enet = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=C.SEED, n_jobs=-1, max_iter=2000)
        enet.fit(X_mat, y)
        sorted_idx = np.argsort(-np.abs(enet.coef_))
        sel, sel_mods = [], set()
        for idx in sorted_idx:
            node = cand_cols[idx]
            m_id = node_to_module[node]
            if m_id in sel_mods: continue
            sel_mods.add(m_id)
            sel.append(node)
            if len(sel) == 3: break
        m2_dict[d_name] = sel

    m3_dict = {dim: group["compound"].tolist() for dim, group in df_final[df_final["method"] == "M3"].groupby("dimension")}

    def check_in_methods(row, method_dict):
        return row["compound"] in method_dict.get(row["dimension"], [])

    df_final["in_M0"] = df_final.apply(lambda r: check_in_methods(r, m0_dict), axis=1)
    df_final["in_M1"] = df_final.apply(lambda r: check_in_methods(r, m1_dict), axis=1)
    df_final["in_M2"] = df_final.apply(lambda r: check_in_methods(r, m2_dict), axis=1)
    df_final["in_M3"] = df_final.apply(lambda r: check_in_methods(r, m3_dict), axis=1)
    df_final.loc[df_final["method"] == "M3", "in_M3"] = True

    cols = list(df_final.columns)
    for c in ["in_M0", "in_M1", "in_M2", "in_M3"]:
        if c in cols: cols.remove(c)
    idx = cols.index("rank") + 1
    for c in reversed(["in_M0", "in_M1", "in_M2", "in_M3"]):
        cols.insert(idx, c)

    df_final[cols].to_csv(final_path, index=False)
    print("✅ 标识列已成功追加 (M0-M3 全部计算完毕，已屏蔽警告)。")

if __name__ == "__main__":
    add_flags()