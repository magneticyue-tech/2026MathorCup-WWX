import numpy as np
from sklearn.linear_model import LinearRegression

def fit_predict_c1(X_tr, Z_tr, X_te):
    """C1: 预测 z，再直接映射为等级"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from src.q1.discretize import grade
    model = HistGradientBoostingRegressor(max_iter=100, random_state=0)
    model.fit(X_tr, Z_tr)
    return grade(model.predict(X_te))

def fit_predict_c2(X_tr, Z_tr, X_te, xbar_tr):
    """C2: 预测 z，用 OLS 回归到 xbar，再离散化"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from src.q1.discretize import grade
    model = HistGradientBoostingRegressor(max_iter=100, random_state=0)
    model.fit(X_tr, Z_tr)
    z_pred_tr = model.predict(X_tr)
    
    # OLS 拟合
    ols = LinearRegression()
    ols.fit(z_pred_tr.reshape(-1, 1), xbar_tr)
    
    z_pred_te = model.predict(X_te)
    x_pred_te = ols.predict(z_pred_te.reshape(-1, 1))
    return grade(x_pred_te)