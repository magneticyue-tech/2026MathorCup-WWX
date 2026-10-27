import numpy as np
import pandas as pd

def b0_mean(train_idx: np.ndarray, test_idx: np.ndarray, xbar: np.ndarray) -> np.ndarray:
    train_mean = np.mean(xbar[train_idx], axis=0)
    return np.tile(train_mean, (len(test_idx), 1))

def b1_style_shrunk(train_idx: np.ndarray, test_idx: np.ndarray, xbar: np.ndarray, style: pd.Series, k=5) -> np.ndarray:
    style_arr = style.values
    train_styles = style_arr[train_idx]
    global_mean = np.mean(xbar[train_idx], axis=0)
    preds = np.zeros((len(test_idx), xbar.shape[1]))
    for i, idx in enumerate(test_idx):
        s = style_arr[idx]
        mask = (train_styles == s)
        n_s = mask.sum()
        if n_s > 0:
            mean_s = np.mean(xbar[train_idx][mask], axis=0)
            preds[i] = (n_s * mean_s + k * global_mean) / (n_s + k)
        else:
            preds[i] = global_mean
    return preds