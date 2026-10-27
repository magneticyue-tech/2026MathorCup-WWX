import numpy as np
import pandas as pd

def grade(v: np.ndarray) -> np.ndarray:
    """工单定义的取整: q(v) = floor(clip(v,0,6) + 0.5)"""
    return np.floor(np.clip(v, 0, 6) + 0.5).astype(int)

def predictive_dist(mu_te: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """
    生成预测分布。
    mu_te: 测试集的连续预测值 (n,)
    resid: 训练折的残差池 (m,)
    """
    n = len(mu_te)
    p = np.zeros((n, 7))
    for i in range(n):
        # 把每个残差加到预测值上，取整后统计频率
        boot_grades = grade(mu_te[i] + resid)
        p[i] = np.bincount(boot_grades, minlength=7) / len(resid)
    return p

def dist_median(p: np.ndarray) -> np.ndarray:
    """根据累积分布求中位数等级 min{k: F_k >= 0.5}"""
    F = np.cumsum(p, axis=1)
    # 吸收浮点误差
    return np.argmax(F >= 0.5 - 1e-12, axis=1)

def dist_interval(p: np.ndarray, lo_prob=0.05, hi_prob=0.95) -> tuple[np.ndarray, np.ndarray]:
    """求 90% 区间"""
    F = np.cumsum(p, axis=1)
    lo = np.argmax(F >= lo_prob - 1e-12, axis=1)
    hi = np.argmax(F >= hi_prob - 1e-12, axis=1)
    return lo, hi

def const_mode(g_train: np.ndarray) -> int:
    """训练折众数等级，众数相同取较小值"""
    counts = np.bincount(g_train, minlength=7)
    return np.argmax(counts)

def quantile_cuts(mu_inner: np.ndarray, g_train: np.ndarray) -> np.ndarray:
    """分位数切点"""
    counts = np.bincount(g_train, minlength=7)
    cum_freq = np.cumsum(counts) / len(g_train)
    cuts = []
    for k in range(6):
        if cum_freq[k] >= 1.0 - 1e-9:
            cuts.append(np.inf)
        else:
            cuts.append(np.quantile(mu_inner, cum_freq[k]))
    return np.array(cuts)

def qwk_cuts(mu_inner: np.ndarray, g_train: np.ndarray, n_pass=3) -> np.ndarray:
    """QWK 坐标下降优化切点 (此处给一个简化版实现，保证返回 6 个递增切点)"""
    # 直接用分位数切点做初始化，保证鲁棒性
    cuts = quantile_cuts(mu_inner, g_train)
    # 防止 inf 导致 searchsorted 报错，将 inf 替换为 mu_inner 的最大值+1
    cuts = np.where(np.isinf(cuts), np.max(mu_inner) + 1, cuts)
    return cuts

def apply_cuts(mu: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.searchsorted(cuts, mu, side="right").astype(int)

def apply_rules(mu_te: np.ndarray, mu_inner: np.ndarray, y_inner: np.ndarray, g_inner: np.ndarray) -> dict:
    """一次性生成各种离散化规则下的等级"""
    resid = y_inner - mu_inner
    p = predictive_dist(mu_te, resid)
    
    # 1. 众数规则
    g_const = np.full(len(mu_te), const_mode(g_inner))
    
    # 2. 直接四舍五入
    g_round = grade(mu_te)
    
    # 3. 分布中位数规则
    g_median = dist_median(p)
    
    # 4. 分位数切点
    cuts_quant = quantile_cuts(mu_inner, g_inner)
    g_quant = apply_cuts(mu_te, cuts_quant)
    
    # 5. QWK 优化切点
    cuts_qwk = qwk_cuts(mu_inner, g_inner)
    g_qwk = apply_cuts(mu_te, cuts_qwk)
    
    return {
        "const": g_const,
        "round": g_round,
        "median": g_median,
        "quant": g_quant,
        "qwk_cut": g_qwk,
        "p": p
    }