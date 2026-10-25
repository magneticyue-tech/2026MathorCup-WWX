import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import spearmanr
from .features import monotone_cst
from .panels import panel_zeros

# 8 种极速网格
# 补全队长要求的 24 种参数网格
PARAM_GRID = []
for depth in [2, 3]:
    for leaf in [10, 20]:
        for max_iter in [100, 200, 400]: # 🚨 核心修改：恢复三档迭代次数
            for l2 in [0, 1]:
                PARAM_GRID.append({
                    "max_depth": depth, 
                    "min_samples_leaf": leaf, 
                    "learning_rate": 0.05, 
                    "max_iter": max_iter, 
                    "l2_regularization": l2
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

def find_best_params_for_group(X_proc, Z, panels, outer_folds, inner_splits_func, group_dims):
    best_score = -np.inf
    best_params = PARAM_GRID[0]
    best_use_mono = False
    
    train_idx_0, _ = outer_folds[0]
    inner_folds = inner_splits_func(train_idx_0)
    feature_names = X_proc.columns.tolist()
    
    X_train_np = X_proc.iloc[train_idx_0].values
    Z_train_np = Z.iloc[train_idx_0].values
    panels_train = panels[train_idx_0]
    dim_indices = [Z.columns.get_loc(d) for d in group_dims]
    
    for params in PARAM_GRID:
        for use_mono in [True, False]:
            group_inner_scores = []
            for in_train_idx, in_val_idx in inner_folds:
                X_in_train = X_train_np[in_train_idx]
                X_in_val = X_train_np[in_val_idx]
                
                Z_in_train_df = Z.iloc[train_idx_0[in_train_idx]]
                panels_inner = panels_train[in_train_idx]
                zeros_inner = panel_zeros(Z_in_train_df, panels_inner)
                
                for dim, z_idx in zip(group_dims, dim_indices):
                    y_in_train = Z_train_np[in_train_idx, z_idx]
                    y_in_val = Z_train_np[in_val_idx, z_idx]
                    
                    col_zeros = zeros_inner.iloc[:, z_idx].dropna().values if zeros_inner.shape[1] > z_idx else []
                    z0_tilde = np.median(col_zeros) if len(col_zeros) > 0 else np.nan
                    
                    mono_cst = monotone_cst(feature_names, dim, use_mono)
                    
                    if dim in HURDLE_DIMS and not np.isnan(z0_tilde):
                        model = HurdleRegressor(base_params=params)
                        at_zero = np.isclose(y_in_train, z0_tilde, atol=1e-5)
                        model.fit(X_in_train, y_in_train, at_zero, z0_tilde)
                        pred = model.predict(X_in_val)
                    else:
                        model = HistGradientBoostingRegressor(**params, random_state=0, monotonic_cst=mono_cst)
                        model.fit(X_in_train, y_in_train)
                        pred = model.predict(X_in_val)
                        
                    rho, _ = spearmanr(pred, y_in_val)
                    group_inner_scores.append(rho if not np.isnan(rho) else 0)
            
            mean_rho = np.mean(group_inner_scores)
            if mean_rho > best_score:
                best_score = mean_rho
                best_params = params
                best_use_mono = use_mono
                
    return best_params, best_use_mono

def run_outer_cv(X_proc, Z, panels, outer_folds, inner_splits_func, dim, best_params, best_use_mono):
    y = Z[dim].values
    feature_names = X_proc.columns.tolist()
    use_hurdle = dim in HURDLE_DIMS
    oof_pred = np.zeros(len(y))
    inner_oof_pred = np.full(len(y), np.nan) # 新增：存内层折外预测
    
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
            
        # === 新增：生成内层折外预测 ===
        inner_folds = inner_splits_func(train_idx)
        for in_train_idx, in_val_idx in inner_folds:
            real_in_train = train_idx[in_train_idx]
            real_in_val = train_idx[in_val_idx]
            
            X_in_train, X_in_val = X_proc.iloc[real_in_train], X_proc.iloc[real_in_val]
            y_in_train = y[real_in_train]
            
            Z_in_train = Z.iloc[real_in_train]
            panels_in = panels[real_in_train]
            zeros_in = panel_zeros(Z_in_train, panels_in)
            z0_in = np.median(zeros_in.iloc[:, dim_idx].dropna().values) if zeros_in.shape[1] > dim_idx else np.nan
            
            if use_hurdle and not np.isnan(z0_in):
                m_in = HurdleRegressor(base_params=best_params)
                at_z_in = np.isclose(y_in_train, z0_in, atol=1e-5)
                m_in.fit(X_in_train, y_in_train, at_z_in, z0_in)
                inner_oof_pred[real_in_val] = m_in.predict(X_in_val)
            else:
                m_in = HistGradientBoostingRegressor(**best_params, random_state=0, monotonic_cst=mono_cst)
                m_in.fit(X_in_train, y_in_train)
                inner_oof_pred[real_in_val] = m_in.predict(X_in_val)
                
    return oof_pred, inner_oof_pred