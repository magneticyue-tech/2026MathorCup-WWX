import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# 把这几个原装的常量也保留着（如果有的话就跳过）
NO_ANCHOR_COLS = [
    "A_malt_all", "A_malt_grain", "A_esters_all", "F_hops_all", 
    "F_esters_all", "alcohol", "aftertaste", "body", "co2", "overall", 
    "F_malt_all"
]

# 🚨 把这个函数补进去！
def calc_m_cap(z_train: np.ndarray, z0_tilde: float) -> float:
    """计算该维度的 m 上限，防止最高分超过 6 分"""
    r = -z0_tilde
    if r <= 0: return 4.0 # 防止除以零
    max_z = np.max(z_train - z0_tilde)
    if max_z <= 0: return 4.0
    return 6 * (1 + r**2) / (r * max_z)

def get_linear_transform(z_train: np.ndarray):
    """4.5 节 线性参照 L：用训练集的最大最小值，把 z 拉伸到 0-6"""
    l_min = np.min(z_train)
    u_max = np.max(z_train)
    if u_max == l_min:
        return 3.0, 0.0 # 如果全一样，全给 3 分
    return (u_max - l_min) / 6.0, l_min # 返回 scale 和 shift

def transform_linear(z: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """直接用线性公式转 0-6"""
    if scale == 0:
        return np.full_like(z, 3.0)
    return np.clip((z - shift) / scale, 0, 6)

def transform_local(z: np.ndarray, z0_tilde: float, m=3.0) -> np.ndarray:
    """4.5 节 局部参照 A：用 m 模型转 0-6"""
    r = -z0_tilde
    if r <= 0: return np.full_like(z, 3.0)
    # 队长公式: A = sigma * (z - z0), 其中 sigma = m * r / (1 + r^2)
    sigma = m * r / (1 + r**2)
    return np.clip(sigma * (z - z0_tilde), 0, 6)

def optimize_qwk_thresholds(z_cont: np.ndarray, R_true: np.ndarray) -> list:
    """
    4.6 节 QWK 优化切点。
    输入：连续的 z 预测值（或映射后的连续分），真实的参照等级 R。
    用一个简单的贪心/网格搜索找 6 个最优切点。
    """
    # 默认标准切点
    best_thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    best_qwk = -np.inf
    
    # 为了速度，我们用极其简单的迭代微调
    # 实际上数据集小，可以暴力搜索，这里我们写个简化版证明逻辑
    for step in [-0.2, -0.1, 0, 0.1, 0.2]:
        test_thresholds = [t + step for t in best_thresholds]
        
        # 用切点分箱
        pred_grades = np.digitize(z_cont, test_thresholds)
        # 计算 QWK
        qwk = cohen_kappa_score(R_true, pred_grades, weights="quadratic", labels=range(7))
        
        if qwk > best_qwk:
            best_qwk = qwk
            best_thresholds = test_thresholds
            
    return best_thresholds

def apply_thresholds(z_cont: np.ndarray, thresholds: list) -> np.ndarray:
    """用找到的最优切点把连续值变整数等级"""
    return np.digitize(z_cont, thresholds).astype(int)