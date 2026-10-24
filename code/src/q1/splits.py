import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

def outer_splits(panels: np.ndarray, style: pd.Series, n_splits=5, n_repeats=5, seed=0) -> list:
    """
    T2: 外层交叉验证划分。
    按场次(panels)分组，按风格(style)分层。
    返回一个列表，包含 n_repeats 个元素，每个元素是一个外层 CV 的 split 生成器列表。
    """
    splits = []
    for r in range(n_repeats):
        # 注意 random_state 加上了 r，保证每次重复的划分不一样
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        # 因为我们需要把同一个外层 CV 的划分同时用于内层，所以转成 list 存下来
        fold_splits = list(sgkf.split(np.zeros(len(panels)), style, groups=panels))
        splits.append(fold_splits)
    return splits

def inner_splits(train_idx: np.ndarray, panels: np.ndarray, style: pd.Series, seed: int) -> list:
    """
    T2: 内层交叉验证划分（用于调参）。
    只在训练集(train_idx)内部切分，同样要分组和分层。
    """
    # 提取训练集内部的数据
    inner_panels = panels[train_idx]
    inner_style = style.iloc[train_idx]
    
    # 使用 4 折
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(train_idx)), inner_style, groups=inner_panels))

def leave_one_style_splits(style: pd.Series) -> list:
    """
    T2: 压力测试 - 留一个风格（Leave-One-Style-Out）。
    每次把某一个风格的啤酒全当做测试集。
    """
    splits = []
    unique_styles = style.unique()
    for s in unique_styles:
        test_idx = np.where(style == s)[0]
        train_idx = np.where(style != s)[0]
        splits.append((train_idx, test_idx))
    return splits

def paper_split(style: pd.Series, seed=0):
    """
    T2: 论文的 70/30 分层随机划分（只作为对照）。
    """
    idx = np.arange(len(style))
    # 按风格分层
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=seed, stratify=style)
    return train_idx, test_idx