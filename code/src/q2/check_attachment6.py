# code/src/q2/check_attachment6.py
"""Q2 验收检查：附件 6 的 44 行复现。"""
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: sys.path.insert(0, str(CURRENT_DIR))

import re
import numpy as np
import pandas as pd
import config as C

def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def check_att6():
    print("开始验证附件 6 复现...")
    
    # 读取原始数据
    chem = pd.read_csv(C.CHEM_PATH).set_index("beer")
    sens = pd.read_csv(C.SENSORY_PATH).set_index("beer")
    
    chem_clean = chem.drop(columns=["beer_id", "tasting_category_fine"], errors="ignore")
    sens_clean = sens.drop(columns=["beer_id", "tasting_category_fine"], errors="ignore")

    # 读取附件 6
    corr6_path = C.DATA_RAW / "6.csv"
    if not corr6_path.exists():
        print("⚠️ 找不到附件6，跳过。")
        return
        
    corr6 = pd.read_csv(corr6_path, index_col=0)
    
    # 建立模糊匹配
    chem_map = {_norm_name(c): c for c in chem_clean.columns}
    sens_map = {_norm_name(c): c for c in sens_clean.columns}
    
    matched_chem, matched_sens = [], []
    for row_name in corr6.index:
        n = _norm_name(row_name)
        if n in chem_map: matched_chem.append(chem_map[n])
            
    for col_name in corr6.columns:
        n = _norm_name(col_name)
        if n in sens_map: matched_sens.append(sens_map[n])

    if not matched_chem or not matched_sens:
        print("⚠️ 无法对齐列名。")
        return

    # 逐对删除计算 Spearman (pandas 的 corr 自动处理 NA)
    merged_df = pd.concat([chem_clean[matched_chem], sens_clean[matched_sens]], axis=1)
    calced_matrix = merged_df.corr(method='spearman').loc[matched_chem, matched_sens]
    
    # 提取附件 6 对应的真实值
    true_matrix = corr6.loc[
        [r for r in corr6.index if _norm_name(r) in chem_map],
        [c for c in corr6.columns if _norm_name(c) in sens_map]
    ]
    true_matrix.index = matched_chem
    true_matrix.columns = matched_sens

    # 计算最大误差
    diff = np.abs(calced_matrix - true_matrix)
    max_error = np.nanmax(diff.values)
    
    print(f"✅ 复算完成！共比对 {len(matched_chem)} 个化学特征 × {len(matched_sens)} 个感官特征。")
    print(f"   -> 最大逐元素误差为: {max_error:.2e}")
    if max_error < 1e-12:
        print("   -> 验收通过：误差低于 10^-12！完全符合工单要求。")
    else:
        print("   -> ⚠️ 误差过大，请检查计算逻辑！")

if __name__ == "__main__":
    check_att6()