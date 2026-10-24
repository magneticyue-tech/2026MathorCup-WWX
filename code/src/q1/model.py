import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from .features import monotone_cst

# 工单 5.2 节：调参网格
PARAM_GRID = [
    {"max_depth": 2, "min_samples_leaf": 10, "learning_rate": 0.05, "l2_regularization": 0},
    {"max_depth": 3, "min_samples_leaf": 20, "learning_rate": 0.05, "l2_regularization": 0},
    # 为了演示，这里只列了 2 种组合。完整网格需要 2(深度) * 2(叶子) * 2(正则) = 8 种
]

# 工单 3.4 节：零膨胀（障碍）模型
class HurdleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_params=None):
        self.base_params = base_params or {}
        
    def fit(self, X, z, at_zero, z0_tilde):
        self.z0_tilde = z0_tilde
        # 1. 分类器：预测是否高于零点（0为等于零点，1为高于零点）
        self.clf = HistGradientBoostingClassifier(**self.base_params)
        self.clf.fit(X, ~at_zero.astype(bool)) # ~at_zero 表示不为0的记为1
        
        # 2. 回归器：只在高于零点的样本上训练
        mask_pos = ~at_zero.astype(bool)
        if mask_pos.sum() > 10: # 防止样本太少报错
            self.reg = HistGradientBoostingRegressor(**self.base_params)
            self.reg.fit(X[mask_pos], z[mask_pos])
        else:
            self.reg = None
        return self
        
    def predict(self, X):
        # 预测概率 P(高于零点)
        pi = self.clf.predict_proba(X)[:, 1]
        # 预测高于零点时的具体分数 mu^+
        if self.reg is not None:
            mu_pos = self.reg.predict(X)
        else:
            mu_pos = np.zeros(len(X))
            
        # 公式：z_hat = pi * mu^+ + (1 - pi) * z0
        return pi * mu_pos + (1 - pi) * self.z0_tilde


def nested_cv_predict(X_proc: pd.DataFrame, Z: pd.DataFrame, panels: np.ndarray, 
                      outer_splits: list, inner_splits_func, dim: str, 
                      use_hurdle: bool = False, at_zero=None, z0_tilde=None):
    """
    执行嵌套交叉验证（外层选模型，内层选参数）
    返回该维度在所有外层测试折上的折外预测值 (Out-of-Fold)
    """
    y = Z[dim].values
    oof_pred = np.zeros(len(y))
    feature_names = X_proc.columns.tolist()
    
    # 遍历外层的 5 折
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        X_train, X_test = X_proc.iloc[train_idx], X_proc.iloc[test_idx]
        y_train = y[train_idx]
        
        # 获取内层划分
        inner_splits = inner_splits_func(train_idx)
        
        # 内层调参：遍历参数网格
        best_score = -np.inf
        best_params = PARAM_GRID[0]
        best_use_mono = False
        
        for params in PARAM_GRID:
            for use_mono in [True, False]:
                inner_scores = []
                # 遍历内层 4 折
                for in_train_idx, in_val_idx in inner_splits:
                    # 映射回原始全局索引
                    real_in_train_idx = train_idx[in_train_idx]
                    real_in_val_idx = train_idx[in_val_idx]
                    
                    X_in_train = X_proc.iloc[real_in_train_idx]
                    X_in_val = X_proc.iloc[real_in_val_idx]
                    y_in_train, y_in_val = y[real_in_train_idx], y[real_in_val_idx]
                    
                    mono_cst = monotone_cst(feature_names, dim, use_mono)
                    
                    if use_hurdle:
                        model = HurdleRegressor(base_params=params)
                        model.fit(X_in_train, y_in_train, at_zero[real_in_train_idx], z0_tilde)
                        pred = model.predict(X_in_val)
                    else:
                        model = HistGradientBoostingRegressor(**params, random_state=0, monotonic_cst=mono_cst)
                        model.fit(X_in_train, y_in_train)
                        pred = model.predict(X_in_val)
                        
                    rho, _ = spearmanr(pred, y_in_val)
                    inner_scores.append(rho if not np.isnan(rho) else 0)
                
                # 计算该配置的平均 Spearman
                mean_rho = np.mean(inner_scores)
                if mean_rho > best_score:
                    best_score = mean_rho
                    best_params = params
                    best_use_mono = use_mono
                    
        # 用内层选出的最佳参数，在整个外层训练集上重新训练
        mono_cst = monotone_cst(feature_names, dim, best_use_mono)
        if use_hurdle:
            final_model = HurdleRegressor(base_params=best_params)
            final_model.fit(X_train, y_train, at_zero[train_idx], z0_tilde)
            oof_pred[test_idx] = final_model.predict(X_test)
        else:
            final_model = HistGradientBoostingRegressor(**best_params, random_state=0, monotonic_cst=mono_cst)
            final_model.fit(X_train, y_train)
            oof_pred[test_idx] = final_model.predict(X_test)
            
    return oof_pred