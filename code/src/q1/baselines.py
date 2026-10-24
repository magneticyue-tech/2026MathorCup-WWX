import numpy as np
import pandas as pd

def b0_mean(train_idx: np.ndarray, test_idx: np.ndarray, Z: pd.DataFrame) -> np.ndarray:
    """B0: 直接用训练折的全局均值预测"""
    train_mean = Z.iloc[train_idx].mean().values
    return np.tile(train_mean, (len(test_idx), 1))

def b1_style_shrunk(train_idx: np.ndarray, test_idx: np.ndarray, Z: pd.DataFrame, style: pd.Series, k=5) -> np.ndarray:
    """B1: 风格均值基线（向全局均值收缩 k=5）"""
    Z_train = Z.iloc[train_idx]
    style_train = style.iloc[train_idx]
    
    global_mean = Z_train.mean().values
    style_means = Z_train.groupby(style_train).mean()
    style_counts = Z_train.groupby(style_train).count()
    
    preds = []
    # 遍历测试集里的每一款酒
    for idx in test_idx:
        s = style.iloc[idx]
        if s in style_means.index:
            n_s = style_counts.loc[s].values
            mean_s = style_means.loc[s].values
            # 收缩公式
            pred = (n_s * mean_s + k * global_mean) / (n_s + k)
        else:
            # 如果遇到训练集没见过的风格，退回全局均值
            pred = global_mean
        preds.append(pred)
    
    return np.array(preds)