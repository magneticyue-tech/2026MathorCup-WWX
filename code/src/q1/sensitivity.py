import numpy as np
import pandas as pd

def additive_effects(ratings: pd.DataFrame, rows: np.ndarray, sens_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    
    # 工单要求：交替最小二乘迭代至 max|Delta|<1e-8 或 500 轮
    for iteration in range(500):
        old_alpha = alpha.copy()
        # 更新 beta
        for j in range(len(tasters)):
            mask = (taster_idx == j)
            if mask.sum() > 0:
                beta[j] = np.mean(X[mask] - alpha[row_idx[mask]], axis=0)
        beta -= np.mean(beta, axis=0) # 加权中心化
        
        # 更新 alpha
        for i in range(len(beers)):
            mask = (row_idx == i)
            if mask.sum() > 0:
                alpha[i] = np.mean(X[mask] - beta[taster_idx[mask]], axis=0)
                
        if np.max(np.abs(alpha - old_alpha)) < 1e-8:
            break
            
    return pd.DataFrame(alpha, index=beers, columns=sens_cols), pd.DataFrame(beta, index=tasters, columns=sens_cols)