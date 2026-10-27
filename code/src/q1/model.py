import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from .features import monotone_cst

# T2 工单：补全的 24 种参数网格
PARAM_GRID = []
for depth in [2, 3]:
    for leaf in [10, 20]:
        for max_iter in [100, 200, 400]:
            for l2 in [0, 1]:
                PARAM_GRID.append({
                    "max_depth": depth, 
                    "min_samples_leaf": leaf, 
                    "learning_rate": 0.05, 
                    "max_iter": max_iter, 
                    "l2_regularization": l2
                })

def nested_cv_predict(X_proc: pd.DataFrame, Y: np.ndarray, groups: np.ndarray, 
                      outer_folds: list, inner_splits_func, dim: str, group_dims: list):
    """
    T2: 嵌套 CV。目标变为真实的 xbar (连续值)。
    选参标准变为内层 RMSE/SD (RMSE标准化后)。
    """
    y_dim = Y[:, group_dims.index(dim)] # 确保切片正确
    feature_names = X_proc.columns.tolist()
    
    # 1. 内层调参阶段
    best_score = np.inf
    best_params = PARAM_GRID[0]
    best_use_mono = False
    
    train_idx_0, _ = outer_folds[0]
    inner_folds = inner_splits_func(train_idx_0)
    
    for params in PARAM_GRID:
        for use_mono in [True, False]:
            inner_scores = []
            for in_train_idx, in_val_idx in inner_folds:
                real_in_train = train_idx_0[in_train_idx]
                real_in_val = train_idx_0[in_val_idx]
                
                X_in_train = X_proc.iloc[real_in_train]
                X_in_val = X_proc.iloc[real_in_val]
                y_in_train = y_dim[real_in_train]
                y_in_val = y_dim[real_in_val]
                
                mono_cst = monotone_cst(feature_names, dim, use_mono)
                
                model = HistGradientBoostingRegressor(
                    **params, random_state=0, monotonic_cst=mono_cst
                )
                model.fit(X_in_train, y_in_train)
                pred = model.predict(X_in_val)
                
                # 计算 RMSE / SD
                rmse = np.sqrt(mean_squared_error(y_in_val, pred))
                sd = np.std(y_in_val)
                score = rmse / (sd + 1e-9)
                inner_scores.append(score)
            
            # 取内层平均，越小的 RMSE/SD 越好
            mean_score = np.mean(inner_scores)
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                best_use_mono = use_mono

    # 2. 外层预测阶段
    oof_mu = np.zeros(len(y_dim))
    chosen_params_list = [best_params] * len(outer_folds) 
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        X_train, X_test = X_proc.iloc[train_idx], X_proc.iloc[test_idx]
        y_train = y_dim[train_idx]
        
        mono_cst = monotone_cst(feature_names, dim, best_use_mono)
        model = HistGradientBoostingRegressor(
            **best_params, random_state=0, monotonic_cst=mono_cst
        )
        model.fit(X_train, y_train)
        oof_mu[test_idx] = model.predict(X_test)
        
    return oof_mu, chosen_params_list