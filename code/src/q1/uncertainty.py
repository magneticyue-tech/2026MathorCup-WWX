import numpy as np
from .convert import to_score

def calc_score_distribution(z_pred: np.ndarray, residuals: np.ndarray, dim: str, m: float, z0_tilde: float, M: float, n_bootstraps=200, rng=None):
    """
    T4: 基于预测残差计算每个啤酒的 0-6 分概率分布 (保留 Bootstrap 作为概率输出)
    """
    if rng is None:
        rng = np.random.default_rng(0)
        
    scores_boot = []
    for _ in range(n_bootstraps):
        z_boot = z_pred + rng.choice(residuals, size=len(z_pred), replace=True)
        boot_score = to_score(z_boot, dim, m, z0_tilde, M)
        scores_boot.append(boot_score)
        
    scores_boot = np.array(scores_boot)
    rounded_scores = np.round(scores_boot).astype(int)
    
    probs = {}
    for k in range(7):
        probs[f'p{k}'] = (rounded_scores == k).mean(axis=0)
        
    final_score = np.round(np.mean(scores_boot, axis=0)).astype(int)
    return final_score, probs

def conformal_halfwidth(resid: np.ndarray, alpha=0.1) -> float:
    """
    T4: 计算 Split-Conformal 的半宽
    """
    n = len(resid)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(np.abs(resid), q_level)

def score_interval(z_hat: float, q: float, dim: str, m: float, z0_tilde: float, M: float) -> tuple[float, float]:
    """
    T4: 将 z 空间区间映射到 0-6 分空间
    """
    z_lower = z_hat - q
    z_upper = z_hat + q
    
    score_lower = to_score(np.array([z_lower]), dim, m, z0_tilde, M)[0]
    score_upper = to_score(np.array([z_upper]), dim, m, z0_tilde, M)[0]
    
    return score_lower, score_upper

def coverage(Z_true: np.ndarray, Z_hat: np.ndarray, q: float) -> float:
    """
    T4: 计算 z 空间的实际覆盖率。如果偏离 90% 超过 5 个百分点（即不在 0.85~0.95 之间），说明需要 Mondrian。
    """
    lower = Z_hat - q
    upper = Z_hat + q
    in_interval = (Z_true >= lower) & (Z_true <= upper)
    return np.mean(in_interval)