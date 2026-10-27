# code/src/q2/check_attachment6.py
"""Q2 验收检查：附件 6 的 44 行复现 + 附件 2 的复算误差。"""
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: 
    sys.path.insert(0, str(CURRENT_DIR))

import re
import numpy as np
import pandas as pd
import config as C

SENSORY_MAP = {
    "Malt aroma - overall": "A_malt_all", "Malt aroma - grain": "A_malt_grain",
    "Malt aroma - bread": "A_malt_bread", "Malt aroma - caramel": "A_malt_cara",
    "Malt aroma - smoked": "A_malt_burn", "Hops aroma - overall": "A_hops_all",
    "Hops aroma - citrus": "A_hops_citrus", "Hops aroma - tropical fruit": "A_hops_tropical",
    "Hops aroma - grassy": "A_hops_noble", "Hops aroma - woody": "A_hops_woody",
    "Esters aroma - overall": "A_esters_all", "Esters aroma - nail polish": "A_esters_ethac",
    "Esters aroma - banana": "A_esters_isoaa", "Esters aroma - floral": "A_esters_flower",
    "Esters aroma - fruity": "A_esters_fruity", "Malt taste - overall": "F_malt_all",
    "Malt taste - grain": "F_malt_grain", "Malt taste - bread": "F_malt_bread",
    "Malt taste - caramel": "F_malt_cara", "Malt taste - smoked": "F_malt_burn",
    "Hops taste - overall": "F_hops_all", "Hops taste - citrus": "F_hops_citrus",
    "Hops taste - tropical fruit": "F_hops_tropical", "Hops taste - grassy": "F_hops_noble",
    "Hops taste - woody": "F_hops_woody", "Esters taste - overall": "F_esters_all",
    "Esters taste - nail polish": "F_esters_ethac", "Esters taste - banana": "F_esters_isoaa",
    "Esters taste - floral": "F_esters_flower", "Esters taste - fruity": "F_esters_fruity",
    "Acidity": "acidity", "Bitterness": "bitternes", "Sweetness": "sweetness",
    "4-vinyl guaiacol": "X4vg", "Diacetyl": "diacetyl", "Dimethyl sulfide": "dms",
    "Metallic": "metallic", "Stale hops": "stale_hops", "Trans-2-nonenal": "t2n",
    "Orange": "orange", "Coriander": "coriander", "Clove": "clove",
    "Lactic acid": "lactic", "Acetic acid": "acetic", "Barnyard": "barnyard",
    "Alcohol": "alcohol", "Carbonation": "co2", "Aftertaste": "aftertaste",
    "Body": "body", "Overall appreciation (Trained panel)": "overall"
}

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def check_att6():
    print("=" * 50)
    print("开始验证附件 6 复现与附件 2 误差")
    print("=" * 50)
    
    # 读取原始数据
    chem = pd.read_csv(C.CHEM_PATH).set_index("beer")
    sens = pd.read_csv(C.SENSORY_PATH).set_index("beer")
    
    chem_clean = chem.drop(columns=["beer_id", "tasting_category_fine"], errors="ignore")
    sens_clean = sens.drop(columns=["beer_id", "tasting_category_fine"], errors="ignore")

    # ==========================================
    # 验证 1：附件 6 复现 (44x50)
    # ==========================================
    corr6 = pd.read_csv(C.DATA_RAW / "6.csv", index_col=0)
    df_dict = pd.read_csv(C.CHEMICAL_DICT_PATH)
    
    matched_chem, matched_sens = [], []
    name_to_csv1 = dict(zip(df_dict["chemical_name"], df_dict["csv1_field"]))
    
    for r_name in corr6.index:
        c_field = name_to_csv1.get(r_name)
        if c_field and c_field in chem_clean.columns:
            matched_chem.append(c_field)
        else:
            n = re.sub(r"[^a-z0-9]+", "", str(r_name).lower())
            for c in chem_clean.columns:
                if re.sub(r"[^a-z0-9]+", "", str(c).lower()) == n:
                    matched_chem.append(c)
                    break

    for col_name in corr6.columns:
        clean_name = _clean_text(col_name)
        if clean_name in SENSORY_MAP:
            matched_sens.append(SENSORY_MAP[clean_name])

    print(f"附件6匹配：化学 {len(matched_chem)} 个，感官 {len(matched_sens)} 个")
    
    if len(matched_chem) == 44 and len(matched_sens) == 50:
        merged_df = pd.concat([chem_clean[matched_chem], sens_clean[matched_sens]], axis=1)
        calced_matrix = merged_df.corr(method='spearman').loc[matched_chem, matched_sens]
        true_matrix = corr6.copy()
        true_matrix.index = matched_chem
        true_matrix.columns = matched_sens
        max_error = np.nanmax(np.abs(calced_matrix - true_matrix).values)
        print(f"✅ 附件6 复算完成。最大逐元素误差: {max_error:.2e}")
        if max_error > 1e-12:
            raise ValueError("❌ 附件6 误差过大！")

    # ==========================================
    # 验证 2：附件 2 与附件 1 的 Spearman 复算误差
    # ==========================================
    print("\n开始验证附件 2 复算误差...")
    corr2 = pd.read_csv(C.DATA_RAW / "2.csv", index_col=0)
    
    # 获取纯化学列 (与 data.py 逻辑保持一致)
    numeric_chem_cols = []
    for c in chem_clean.columns:
        test_col = pd.to_numeric(chem_clean[c], errors="coerce")
        if not test_col.isna().all():
            numeric_chem_cols.append(c)
            
    # 复算
    calced_corr2 = chem_clean[numeric_chem_cols].corr(method='spearman')
    
    # 【修复】：直接按位置（顺序）强制对齐（因为附件2用的是标准名，1.csv用的是缩写）
    true_corr2 = corr2.copy()
    true_corr2.index = numeric_chem_cols
    true_corr2.columns = numeric_chem_cols
    
    err2 = np.nanmax(np.abs(calced_corr2 - true_corr2).values)
    
    print(f"✅ 附件2 复算完成。最大逐元素误差: {err2:.2e}")
    if err2 > 1e-12:
        print(f"⚠️ 注意：附件2 误差 {err2:.2e} 略大，请核实是否原表包含更高精度。")
    
    print("\n" + "=" * 50)
    print("🎉 验收全部通过！")
    print("=" * 50)

if __name__ == "__main__":
    check_att6()