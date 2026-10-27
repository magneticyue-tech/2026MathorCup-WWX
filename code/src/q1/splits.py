import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

def outer_splits(groups: np.ndarray, style: pd.Series, n_splits=5, n_repeats=5, seed=0) -> list:
    """
    T2: 外层交叉验证划分。
    按 date_group 分组，按风格(style)分层。
    """
    splits = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        fold_splits = list(sgkf.split(np.zeros(len(groups)), style, groups=groups))
        splits.append(fold_splits)
    return splits

def inner_splits(train_idx: np.ndarray, groups: np.ndarray, style: pd.Series, seed: int) -> list:
    """
    T2: 内层交叉验证划分。随机种子按工单要求使用 SEED + 100 + r。
    """
    inner_groups = groups[train_idx]
    inner_style = style.iloc[train_idx]
    
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(train_idx)), inner_style, groups=inner_groups))

def leave_one_style_splits(style: pd.Series) -> list:
    splits = []
    unique_styles = style.unique()
    for s in unique_styles:
        test_idx = np.where(style == s)[0]
        train_idx = np.where(style != s)[0]
        splits.append((train_idx, test_idx))
    return splits

def paper_split(style: pd.Series, seed=0):
    idx = np.arange(len(style))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=seed, stratify=style)
    return train_idx, test_idx

def split_diagnostics(splits: list, groups: np.ndarray, style: pd.Series) -> dict:
    """T2: 计算各折大小与分组隔离情况"""
    diag = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # 验证是否同组跨越了测试集和训练集
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        leaked_groups = train_groups.intersection(test_groups)
        
        diag.append({
            "fold": fold_idx,
            "test_size": len(test_idx),
            "leaked_groups_count": len(leaked_groups)
        })
    return diag