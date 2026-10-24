import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def tie_pair_counts(Z: pd.DataFrame, decimals=9) -> dict:
    cnt = {}
    for c in Z.columns:
        for idx in pd.Series(np.arange(len(Z))).groupby(Z[c].round(decimals).values).groups.values():
            idx = sorted(idx)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    cnt[(idx[a], idx[b])] = cnt.get((idx[a], idx[b]), 0) + 1
    return cnt

def infer_panels(Z: pd.DataFrame, k=3, decimals=9) -> np.ndarray:
    n = len(Z)
    cnt = tie_pair_counts(Z, decimals)
    e = [p for p, v in cnt.items() if v >= k]
    if not e: 
        return np.arange(n)
    A = coo_matrix((np.ones(len(e)), ([i for i, _ in e], [j for _, j in e])), shape=(n, n))
    return connected_components(A, directed=False)[1]

def panel_zeros(Z: pd.DataFrame, panels: np.ndarray, decimals=9) -> pd.DataFrame:
    z_round = Z.round(decimals)
    session_counts = pd.Series(panels).value_counts()
    multi_sessions = session_counts[session_counts > 1].index
    
    out_data = {}
    for c in Z.columns:
        out = {}
        for g in multi_sessions:
            v = z_round.loc[panels == g, c]
            if (v == v.min()).sum() >= 2:
                out[g] = v.min()
        out_data[c] = pd.Series(out, dtype=float)
    
    return pd.DataFrame(out_data) 

def at_zero_mask(Z: pd.DataFrame, panels: np.ndarray, zeros: pd.DataFrame) -> np.ndarray:
    mask = np.zeros_like(Z, dtype=bool)
    for g in zeros.index:
        if pd.isna(g): continue
        idx = np.where(panels == int(g))[0]
        z0_g = zeros.loc[g].values
        mask[idx, :] = np.isclose(Z.iloc[idx, :].values, z0_g, atol=1e-9)
    return mask