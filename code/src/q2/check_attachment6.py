# code/src/q2/check_attachment6.py
"""Q2 验收检查：附件 6 的 44 行复现 (硬编码精准映射 + 空白字符清洗)。"""
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: 
    sys.path.insert(0, str(CURRENT_DIR))

import re
import numpy as np
import pandas as pd
import config as C

# 硬编码精准映射表：6.csv 列名 -> 4.csv 列名 (键名已将空白字符规范化)
SENSORY_MAP = {
    "Malt aroma - overall": "A_malt_all",
    "Malt aroma - grain": "A_malt_grain",
    "Malt aroma - bread": "A_malt_bread",
    "Malt aroma - caramel": "A_malt_cara",
    "Malt aroma - smoked": "A_malt_burn",
    "Hops aroma - overall": "A_hops_all",
    "Hops aroma - citrus": "A_hops_citrus",
    "Hops aroma - tropical fruit": "A_hops_tropical",
    "Hops aroma - grassy": "A_hops_noble",
    "Hops aroma - woody": "A_hops_woody",
    "Esters aroma - overall": "A_esters_all",
    "Esters aroma - nail polish": "A_esters_ethac",
    "Esters aroma - banana": "A_esters_isoaa",
    "Esters aroma - floral": "A_esters_flower",
    "Esters aroma - fruity": "A_esters_fruity",
    "Malt taste - overall": "F_malt_all",
    "Malt taste - grain": "F_malt_grain",
    "Malt taste - bread": "F_malt_bread",
    "Malt taste - caramel": "F_malt_cara",
    "Malt taste - smoked": "F_malt_burn",
    "Hops taste - overall": "F_hops_all",
    "Hops taste - citrus": "F_hops_citrus",
    "Hops taste - tropical fruit": "F_hops_tropical",
    "Hops taste - grassy": "F_hops_noble",
    "Hops taste - woody": "F_hops_woody",
    "Esters taste - overall": "F_esters_all",
    "Esters taste - nail polish": "F_esters_ethac",
    "Esters taste - banana": "F_esters_isoaa",
    "Esters taste - floral": "F_esters_flower",
    "Esters taste - fruity": "F_esters_fruity",
    "Acidity": "acidity",
    "Bitterness": "bitternes", # 注意少个 s
    "Sweetness": "sweetness",
    "4-vinyl guaiacol": "X4vg",
    "Diacetyl": "diacetyl",
    "Dimethyl sulfide": "dms",
    "Metallic": "metallic",
    "Stale hops": "stale_hops",
    "Trans-2-nonenal": "t2n",
    "Orange": "orange",
    "Coriander": "coriander",
    "Clove": "clove",
    "Lactic acid": "lactic",
    "Acetic acid": "acetic",
    "Barnyard": "barnyard",
    "Alcohol": "alcohol",
    "Carbonation": "co2",
    "Aftertaste": "aftertaste",
    "Body": "body",
    "Overall appreciation (Trained panel)": "overall" # 这里已统一为空格
}

def _clean_text(s: str) -> str:
    """将所有的连续空白字符（包括换行、制表符等）替换为单个空格，并去除首尾空白"""
    return re.sub(r"\s+", " ", str(s)).strip()

def check_att6():
    print("=" * 50)
    print("开始验证附件 6 复现 (44 行 × 50 列)")
    print("=" * 50)
    
    # 读取原始数据
    chem = pd.read_csv(C.CHEM_PATH).set_index("beer")
    sens = pd.read_csv(C.SENSORY_PATH).set_index("beer")
    
    chem_clean = chem.drop(columns=["beer_id", "tasting_category_fine"], errors="ignore")
    sens_clean = sens.drop(columns=["beer_id", "tasting_category_fine"], errors="ignore")

    corr6 = pd.read_csv(C.DATA_RAW / "6.csv", index_col=0)
    df_dict = pd.read_csv(C.CHEMICAL_DICT_PATH)
    
    matched_chem, matched_sens = [], []
    
    # 1. 匹配化学特征 (44 行)
    name_to_csv1 = dict(zip(df_dict["chemical_name"], df_dict["csv1_field"]))
    for r_name in corr6.index:
        c_field = name_to_csv1.get(r_name)
        if c_field and c_field in chem_clean.columns:
            matched_chem.append(c_field)
        else:
            n = re.sub(r"[^a-z0-9]+", "", str(r_name).lower())
            found = False
            for c in chem_clean.columns:
                if re.sub(r"[^a-z0-9]+", "", str(c).lower()) == n:
                    matched_chem.append(c)
                    found = True
                    break
            if not found:
                raise ValueError(f"化学特征未匹配: {r_name!r}")

    # 2. 匹配感官特征 (50 列)
    for col_name in corr6.columns:
        # 【核心修复】：清洗掉换行符和多余空格后再匹配
        clean_name = _clean_text(col_name)
        
        if clean_name in SENSORY_MAP:
            matched_sens.append(SENSORY_MAP[clean_name])
        else:
            raise ValueError(f"感官特征未匹配: {col_name!r} (清洗后: {clean_name!r})")

    print(f"\n📊 附件6 形状: {corr6.shape}")
    print(f"匹配结果: 化学特征 {len(matched_chem)} 个, 感官特征 {len(matched_sens)} 个")
    
    if len(matched_chem) == 44 and len(matched_sens) == 50:
        print("\n✅ 名称全部对齐！正在计算逐对删除 Spearman...")
        
        # 逐对删除计算 Spearman
        merged_df = pd.concat([chem_clean[matched_chem], sens_clean[matched_sens]], axis=1)
        calced_matrix = merged_df.corr(method='spearman').loc[matched_chem, matched_sens]
        
        # 提取真实值并重命名以对齐矩阵
        true_matrix = corr6.copy()
        true_matrix.index = matched_chem
        true_matrix.columns = matched_sens
        
        diff = np.abs(calced_matrix - true_matrix)
        max_error = np.nanmax(diff.values)
        
        print(f"   -> 共比对 {len(matched_chem)} × {len(matched_sens)} = 2200 个元素。")
        print(f"   -> 最大逐元素误差为: {max_error:.2e}")
        
        if max_error <= 1e-12:
            print("   -> 验收通过：误差低于 10^-12！完全符合工单要求。")
        else:
            print("   -> ⚠️ 误差过大，请检查计算逻辑！")
    else:
        print("\n⚠️ 仍有未匹配的列名！")

if __name__ == "__main__":
    check_att6()