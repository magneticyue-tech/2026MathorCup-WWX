import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import spearmanr
from .features import monotone_cst
from .panels import panel_zeros

# 24 种参数网格保持不变
PARAM_GRID = []
for depth in [2, 3]:
    for leaf in [10, 20]:
        for max_iter in [100, 200, 400]:
            for l2 in [0, 1]:
                PARAM_GRID.append({
                    "max_depth": depth, "min_samples_leaf": leaf, 
                    "learning_rate": 0.05, "max_iter": max_iter, "l2_regularization": l2
                })

HURDLE_DIMS = ["acetic", "coriander", "lactic", "clove", "orange", "barnyard", 
               "A_malt_burn", "stale_hops", "F_esters_flower", "F_hops_tropical"]

class HurdleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_params=None):
        self.base_params = base_params or {}
        
    def fit(self, X, z, at_zero, z0_tilde):
        self.z0_tilde = z0_tilde
        self.clf = HistGradientBoostingClassifier(**self.base_params)
        self.clf.fit(X, ~at_zero.astype(bool)) 
        mask_pos = ~at_zero.astype(bool)
        if mask_pos.sum() > 10: 
            self.reg = HistGradientBoostingRegressor(**self.base_params)
            self.reg.fit(X[mask_pos], z[mask_pos])
        else:
            self.reg = None
        return self
        
    def predict(self, X):
        pi = self.clf.predict_proba(X)[:, 1]
        mu_pos = self.reg.predict(X) if self.reg is not None else np.zeros(len(X))
        return pi * mu_pos + (1 - pi) * self.z0_tilde

# 🚨 新增：专门用于在组内搜索一次最优参数
def find_best_params_for_group(X_proc, Z, panels, outer_folds, inner_splits_func, group_dims):
    # 从组里挑第一个维度作为代表来搜参
    rep_dim = group_dims[0]
    y = Z[rep_dim].values
    feature_names = X_proc.columns.tolist()
    
    best_score = -np.inf
    best_params = PARAM_GRID[0]
    best_use_mono = False
    
    train_idx_0, _ = outer_folds[0]
    inner_folds = inner_splits_func(train_idx_0)
    
    for params in PARAM_GRID:
        for use_mono in [True, False]:
            inner_scores = []
            for in_train_idx, in_val_idx in inner_folds:
                real_in_train_idx = train_idx_0[in_train_idx]
                real_in_val_idx = train_idx_0[in_val_idx]
                
                X_in_train = X_proc.iloc[real_in_train_idx]
                X_in_val = X_proc.iloc[real_in_val_idx]
                y_in_train, y_in_val = y[real_in_train_idx], y[real_in_val_idx]
                
                Z_in_train = Z.iloc[real_in_train_idx]
                panels_in_train = panels[real_in_train_idx]
                zeros_in_train = panel_zeros(Z_in_train, panels_in_train)
                dim_idx = Z.columns.get_loc(rep_dim)
                col_zeros = zeros_in_train.iloc[:, dim_idx].dropna().values if zeros_in_train.shape[1] > dim_idx else []
                z0_tilde_inner = np.median(col_zeros) if len(col_zeros) > 0 else np.nan
                
                mono_cst = monotone_cst(feature_names, rep_dim, use_mono)
                
                if rep_dim in HURDLE_DIMS and not np.isnan(z0_tilde_inner):
                    model = HurdleRegressor(base_params=params)
                    at_zero_inner = np.isclose(y_in_train, z0_tilde_inner, atol=1e-5)
                    model.fit(X_in_train, y_in_train, at_zero_inner, z0_tilde_inner)
                    pred = model.predict(X_in_val)
                else:
                    model = HistGradientBoostingRegressor(**params, random_state=0, monotonic_cst=mono_cst)
                    model.fit(X_in_train, y_in_train)
                    pred = model.predict(X_in_val)
                    
                rho, _ = spearmanr(pred, y_in_val)
                inner_scores.append(rho if not np.isnan(rho) else 0)
            
            mean_rho = np.mean(inner_scores)
            if mean_rho > best_score:
                best_score = mean_rho
                best_params = params
                best_use_mono = use_mono
                
    return best_params, best_use_mono

# 🚨 修改：直接用传进来的最优参数跑外层，不再自己搜参
def run_outer_cv(X_proc, Z, panels, outer_folds, dim, best_params, best_use_mono):
    y = Z[dim].values
    feature_names = X_proc.columns.tolist()
    use_hurdle = dim in HURDLE_DIMS
    oof_pred = np.zeros(len(y))
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        X_train, X_test = X_proc.iloc[train_idx], X_proc.iloc[test_idx]
        y_train = y[train_idx]
        
        Z_train = Z.iloc[train_idx]
        panels_train = panels[train_idx]
        zeros_train_df = panel_zeros(Z_train, panels_train)
        dim_idx = Z.columns.get_loc(dim)
        
        if zeros_train_df.shape[1] > dim_idx:
            col_zeros = zeros_train_df.iloc[:, dim_idx].dropna().values
            z0_tilde_train = np.median(col_zeros) if len(col_zeros) > 0 else np.nan
        else:
            z0_tilde_train = np.nan

        mono_cst = monotone_cst(feature_names, dim, best_use_mono)
        
        if use_hurdle and not np.isnan(z0_tilde_train):
            model = HurdleRegressor(base_params=best_params)
            at_zero_train = np.isclose(y_train, z0_tilde_train, atol=1e-5)
            model.fit(X_train, y_train, at_zero_train, z0_tilde_train)
            oof_pred[test_idx] = model.predict(X_test)
        else:
            model = HistGradientBoostingRegressor(**best_params, random_state=0, monotonic_cst=mono_cst)
            model.fit(X_train, y_train)
            oof_pred[test_idx] = model.predict(X_test)
            
    return oof_pred