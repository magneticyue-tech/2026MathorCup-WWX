# code/src/q2/data.py
"""Q2 数据层：候选集合、样本对齐、NA/下限/秩。"""
from __future__ import annotations

import sys
from pathlib import Path

# 路径补丁
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import re
from dataclasses import dataclass
import numpy as np
import pandas as pd

import config as C


@dataclass(frozen=True)
class Q2Dataset:
    """Q2 主流程所需的最小数据包。"""
    chem_raw: pd.DataFrame
    corr: pd.DataFrame
    sensory_raw: pd.DataFrame
    sample_ids: pd.Index
    candidate_cols: list[str]
    annotation_cols: list[str]
    X_rank: pd.DataFrame
    Y_rank: pd.DataFrame


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"不支持的文件类型：{path.suffix}")


def _read_with_sample_id(path: Path, id_col: str | None = None) -> pd.DataFrame:
    df = _read_table(path)
    if id_col is None:
        id_col = str(df.columns[0])
    if id_col not in df.columns:
        raise KeyError(f"{path} 中找不到样本 ID 列 {id_col!r}")
    df = df.set_index(id_col)
    df.index = df.index.astype(str)
    df.index.name = "sample_id"
    df.columns = df.columns.astype(str)
    return df


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _resolve_columns(available: list[str], wanted: tuple[str, ...]) -> dict[str, str]:
    norm_map: dict[str, list[str]] = {}
    for col in available:
        norm_map.setdefault(_norm_name(col), []).append(col)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: list[str] = []

    for w in wanted:
        hits = norm_map.get(_norm_name(w), [])
        if len(hits) == 1:
            resolved[w] = hits[0]
        elif len(hits) == 0:
            missing.append(w)
        else:
            ambiguous.append(f"{w} -> {hits}")

    if missing or ambiguous:
        msg = []
        if missing: msg.append(f"缺失列：{missing}")
        if ambiguous: msg.append(f"列名歧义：{ambiguous}")
        raise KeyError("候选集合排除列解析失败；" + "；".join(msg))

    return resolved


def _rank_standardize(
    df: pd.DataFrame,
    *,
    fill_na_mean: bool,
    clip_floor: bool,
) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")

    if fill_na_mean:
        out = out.fillna(out.mean(axis=0))

    if clip_floor:
        out = out.clip(lower=C.SIGNAL_FLOOR)

    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"秩变换前仍有 NA，列：{bad[:10]}...")

    ranks = out.rank(axis=0, method="average")
    mu = ranks.mean(axis=0)
    sd = ranks.std(axis=0, ddof=0)
    
    if (sd == 0).any():
        bad = sd[sd == 0].index.tolist()
        raise ValueError(f"存在常数列，无法标准化：{bad[:10]}")

    return (ranks - mu) / sd


def load_q2_dataset() -> Q2Dataset:
    print("正在读取数据文件...")
    chem = _read_with_sample_id(C.CHEM_PATH)
    corr = _read_with_sample_id(C.CORR_PATH)
    sensory = _read_with_sample_id(C.SENSORY_PATH)

    print("正在处理化合物候选集...")
    ignore_keywords = ["beer_id", "tasting"]
    
    # 1. 过滤 1.csv (化合物数据) 中的非数值列
    numeric_chem_cols = []
    for c in chem.columns:
        if any(k in str(c).lower() for k in ignore_keywords):
            continue
        test_col = pd.to_numeric(chem[c], errors='coerce')
        if not test_col.isna().all(): 
            numeric_chem_cols.append(c)
            
    if len(numeric_chem_cols) != 231:
        raise ValueError(f"过滤后，纯化学列数应为 231，实际为 {len(numeric_chem_cols)}。")

    corr.index = numeric_chem_cols
    corr.columns = numeric_chem_cols

    resolved = _resolve_columns(numeric_chem_cols, C.ANNOTATION_ONLY_COLS)
    annotation_cols = list(resolved.values())
    excluded = set(annotation_cols)
    candidate_cols = [c for c in numeric_chem_cols if c not in excluded]

    # 2. 过滤 4.csv (感官数据) 中的非数值列
    print("正在处理感官目标...")
    numeric_sensory_cols = []
    for c in sensory.columns:
        if any(k in str(c).lower() for k in ignore_keywords):
            continue
        test_col = pd.to_numeric(sensory[c], errors='coerce')
        if not test_col.isna().all():
            numeric_sensory_cols.append(c)
            
    if len(numeric_sensory_cols) != 50:
        raise ValueError(f"过滤后，纯感官列数应为 50，实际为 {len(numeric_sensory_cols)}。")
    
    sensory = sensory[numeric_sensory_cols]

    common = chem.index.intersection(sensory.index)
    if len(common) != len(chem.index) or len(common) != len(sensory.index):
        raise ValueError(f"样本不一致：chem={len(chem.index)}, sensory={len(sensory.index)}, common={len(common)}")

    chem = chem.loc[common]
    sensory = sensory.loc[common]

    print("正在计算标准化秩...")
    X_rank = _rank_standardize(chem[candidate_cols], fill_na_mean=True, clip_floor=True)
    Y_rank = _rank_standardize(sensory, fill_na_mean=False, clip_floor=False)

    return Q2Dataset(
        chem_raw=chem,
        corr=corr,
        sensory_raw=sensory,
        sample_ids=common,
        candidate_cols=candidate_cols,
        annotation_cols=annotation_cols,
        X_rank=X_rank,
        Y_rank=Y_rank,
    )


if __name__ == "__main__":
    print("="*50)
    print("开始 Q2 数据层验收测试")
    print("="*50)
    
    try:
        ds = load_q2_dataset()
        print("\n✅ 数据读取成功！")
        print(f"样本数: {len(ds.sample_ids)} (期望: 250)")
        print(f"候选列数: {len(ds.candidate_cols)} (期望: 225)")
        print(f"注释列数: {len(ds.annotation_cols)} (期望: 6)")
        print(f"附件2形状: {ds.corr.shape} (期望: (231, 231))")
        print(f"X_rank形状: {ds.X_rank.shape} (期望: (250, 225))")
        print(f"Y_rank形状: {ds.Y_rank.shape} (期望: (250, 50))")
        
        max_mean = float(ds.X_rank.mean(axis=0).abs().max())
        print(f"X_rank列均值最大绝对值: {max_mean:.6f} (应接近 0)")
        
        std_min = float(ds.X_rank.std(axis=0, ddof=0).min())
        std_max = float(ds.X_rank.std(axis=0, ddof=0).max())
        print(f"X_rank列标准差范围: [{std_min:.6f}, {std_max:.6f}] (应接近 1)")
        print("="*50)
        
    except Exception as e:
        print("\n❌ 运行出错：")
        print(e)