import numpy as np
# 方法论 4.2 节：无锚维度名单
NO_ANCHOR_COLS = [
    "A_malt_all", "A_malt_grain", "A_esters_all", "F_hops_all", 
    "F_esters_all", "alcohol", "aftertaste", "body", "co2", "overall", 
    "F_malt_all"
]

def to_score(
    z_pred: np.ndarray,
    dim: str,
    m: float,
    z0_tilde: float,
    M: float,
    zero: np.ndarray | float | None = None,
) -> np.ndarray:
    """
    T3: 将 z 值转换为 0-6 分。
    有锚维度使用 m 模型，无锚维度使用 C-mid。
    zero 是逐啤酒的本组本维零点；缺失时回退到训练集零点中位数。
    """
    if dim in NO_ANCHOR_COLS or np.isnan(z0_tilde):
        # 无锚：C-mid 约定
        score = 3 + 3 * z_pred / (M + 1e-9)
    else:
        # 有锚：m 模型
        r = -z0_tilde
        origin = z0_tilde if zero is None else np.where(np.isfinite(zero), zero, z0_tilde)
        score = m * (r / (1 + r**2)) * (z_pred - origin)
        
    # 截断在 0 到 6 之间
    return np.clip(score, 0, 6)

def calc_m_cap(z_train: np.ndarray, z0_tilde: float) -> float:
    """计算该维度的 m 上限，防止最高分超过 6 分"""
    r = -z0_tilde
    if r <= 0: return 4.0 # 防止除以零
    max_z = np.max(z_train - z0_tilde)
    if max_z <= 0: return 4.0
    return 6 * (1 + r**2) / (r * max_z)
