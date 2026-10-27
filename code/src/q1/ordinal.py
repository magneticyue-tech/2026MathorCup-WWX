import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

def fit_predict_k2(X_tr, G_tr, X_te, params, min_pos=2):
    """工单要求的 K2：累积概率方法"""
    n_te = len(X_te)
    p_k2 = np.zeros((n_te, 7))
    
    # 对 t=1..6 拟合
    for t in range(1, 7):
        y_bin = (G_tr >= t).astype(int)
        if y_bin.sum() < min_pos or (1 - y_bin).sum() < min_pos:
            p_k2[:, t] = y_bin.mean()
        else:
            clf = HistGradientBoostingClassifier(**params, random_state=0)
            clf.fit(X_tr, y_bin)
            p_k2[:, t] = clf.predict_proba(X_te)[:, 1]
            
    # 沿 t 取累积最小值
    F = np.minimum.accumulate(p_k2, axis=1)
    # 差分得到概率
    p = np.diff(np.hstack([np.zeros((n_te, 1)), F]), axis=1)
    return np.clip(p, 1e-9, 1.0)