# code/src/q2/labels.py
"""附件 8 Sensory thresholds 的原始标签，按附件 1 列名（field_name_R）取值。"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd

import config as C

LABEL_FIELDS = ("chemical_name", "origin_paper", "off_flavors", "aging", "unit", "threshold_mean")


def attachment8_labels(chem_cols: list[str]) -> pd.DataFrame:
    t = pd.read_excel(C.ATT8_PATH, sheet_name=C.THRESHOLD_SHEET)
    if t["field_name_R"].duplicated().any():
        raise ValueError("附件 8 field_name_R 重复")
    t = t.set_index("field_name_R")
    missing = [c for c in chem_cols if c not in t.index]
    if missing:
        raise KeyError(f"附件 8 缺少化合物键：{missing[:10]}")
    out = t.loc[chem_cols, list(LABEL_FIELDS)].copy()
    out.index.name = "compound"
    return out


def check_against_dictionary(lab: pd.DataFrame) -> int:
    """附件 8 标签与审计 chemical_dictionary.csv 按化合物键逐项一致；返回核对单元数。"""
    d = pd.read_csv(C.CHEMICAL_DICT_PATH).set_index("csv1_field").reindex(lab.index)
    n = 0
    for f in LABEL_FIELDS:
        a, b = lab[f], d[f]
        if f == "threshold_mean":
            same = (a.isna() & b.isna()) | np.isclose(a.astype(float), b.astype(float), equal_nan=False)
        else:
            same = (a.isna() & b.isna()) | (a.astype(str).str.strip() == b.astype(str).str.strip())
        if not same.all():
            raise ValueError(f"附件 8 与 chemical_dictionary 不一致：{f} {lab.index[~same].tolist()[:5]}")
        n += len(same)
    return n


def transform_kind(chem_cols: list[str]) -> pd.Series:
    d = pd.read_csv(C.CHEMICAL_DICT_PATH).set_index("csv1_field").reindex(chem_cols)
    def kind(s: str) -> str:
        if s.startswith("log10"):
            return "log10"
        if s.startswith("线性"):
            return "linear"
        return "uncalibrated"

    return d["transform_interpretation"].fillna("").map(kind)


def candidate_type(col: str) -> str:
    if col in C.PHYSCHEM_COLS:
        return "理化指标"
    if col in C.CLASS_COLS:
        return "化合物类"
    if col in C.COMPOSITE_COLS:
        return "综合指标"
    if col == C.INTERNAL_STANDARD_COL:
        return "内标"
    return "单一化合物"
