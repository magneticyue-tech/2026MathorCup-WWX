import numpy as np
import pandas as pd

# 方法论 3.3 节：6 个化合物 -> 感官配对
MONOTONE_PAIRS = {
    "alcohol": ("ethanol..v.v.", 1),       # 乙醇 -> 酒精感 (+)
    "bitternes": ("bitterness", 1),        # 苦味值 -> 苦味 (+)
    "acidity": ("pH", -1),                 # pH -> 酸感 (-)
    "lactic": ("lactic_acid.mg.L.", 1),    # 乳酸 -> 乳酸味 (+)
    "acetic": ("acetic_acid.g.L.", 1),     # 乙酸 -> 醋酸味 (+)
    "X4vg": ("X4VG", 1)                    # 4VG -> 4VG 气味 (+)
}

def monotone_cst(feature_names: list, dim: str, enabled: bool) -> list[int]:
    """
    生成 HistGradientBoostingRegressor 要求的 monotonic_cst 列表。
    0 表示无约束，1 表示正相关，-1 表示负相关。
    """
    if not enabled or dim not in MONOTONE_PAIRS:
        return [0] * len(feature_names)
    
    feature_name, direction = MONOTONE_PAIRS[dim]
    cst = [0] * len(feature_names)
    
    if feature_name in feature_names:
        idx = feature_names.index(feature_name)
        cst[idx] = direction
        
    return cst