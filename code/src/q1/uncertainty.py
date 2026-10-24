import numpy as np

def calc_score_distribution(z_pred: np.ndarray, residuals: np.ndarray, dim: str, m: float, z0_tilde: float, M: float, n_bootstraps=200):
    """
    T4: 基于预测残差计算每个啤酒的 0-6 分概率分布和 90% 置信区间。
    """
    from .convert import to_score
    
    scores_boot = []
    # 随机抽样 200 次
    for _ in range(n_bootstraps):
        # 随机抽取残差加在预测值上，模拟不确定性
        z_boot = z_pred + np.random.choice(residuals, size=len(z_pred), replace=True)
        
        # 转换并保存
        boot_score = to_score(z_boot, dim, m, z0_tilde, M)
        scores_boot.append(boot_score)
        
    scores_boot = np.array(scores_boot) # shape: (200, 250)
    
    # 1. 计算 0-6 分各分值的概率 (p0 到 p6)
    rounded_scores = np.round(scores_boot).astype(int)
    probs = {}
    for k in range(7):
        probs[f'p{k}'] = (rounded_scores == k).mean(axis=0)
        
    # 2. 计算 90% 置信区间 (取 5% 和 95% 分位数)
    lo = np.percentile(scores_boot, 5, axis=0)
    hi = np.percentile(scores_boot, 95, axis=0)
    
    # 3. 计算最终的交付评分 (取中位数或均值)
    final_score = np.round(np.mean(scores_boot, axis=0)).astype(int)
    
    return final_score, probs, lo, hi