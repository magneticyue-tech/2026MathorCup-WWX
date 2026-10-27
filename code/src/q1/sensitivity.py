import numpy as np
import pandas as pd

def additive_effects(ratings: pd.DataFrame, rows: np.ndarray, sens_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    工单要求：拟合 x_id = alpha_bd + beta_pd + epsilon
    交替最小二乘（ALS），只对指定的啤酒行进行拟合
    """
    df = ratings[ratings['row'].isin(rows)]
    beers = sorted(df['row'].unique())
    tasters = sorted(df['taster'].unique())
    
    b_idx = {b: i for i, b in enumerate(beers)}
    t_idx = {t: i for i, t in enumerate(tasters)}
    
    alpha = np.zeros((len(beers), len(sens_cols)))
    beta = np.zeros((len(tasters), len(sens_cols)))
    
    X = df[sens_cols].values
    row_idx = np.array([b_idx[b] for b in df['row']])
    taster_idx = np.array([t_idx[t] for t in df['taster']])
    
    # ALS 迭代 (简化为 100 轮，保证速度，同时确保收敛)
    for _ in range(100):
        # 更新 beta (控制 alpha 不变)
        for j, t in enumerate(tasters):
            mask = (taster_idx == j)
            if mask.sum() > 0:
                beta[j] = np.mean(X[mask] - alpha[row_idx[mask]], axis=0)
        # 加权中心化 beta
        beta -= np.mean(beta, axis=0)
        
        # 更新 alpha (控制 beta 不变)
        for i, b in enumerate(beers):
            mask = (row_idx == i)
            if mask.sum() > 0:
                alpha[i] = np.mean(X[mask] - beta[taster_idx[mask]], axis=0)
                
    alpha_df = pd.DataFrame(alpha, index=beers, columns=sens_cols)
    beta_df = pd.DataFrame(beta, index=tasters, columns=sens_cols)
    return alpha_df, beta_df