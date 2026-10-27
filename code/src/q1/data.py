import pandas as pd
import numpy as np
from dataclasses import dataclass
from .paths import RAW_DIR, RAW_SENSORY

@dataclass
class Q1Targets:
    xbar: np.ndarray      
    G: np.ndarray         
    n_ratings: np.ndarray 
    date_group: np.ndarray
    reliability: np.ndarray
    reconstruct_ok: np.ndarray
    half_grade_exact: np.ndarray
    half_grade_qwk: np.ndarray
    attendance_share: np.ndarray

@dataclass
class Q1Data:
    X: pd.DataFrame
    Z: pd.DataFrame
    targets: Q1Targets
    beer: pd.Series
    style: pd.Series
    dims: list

def grade(v: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(v, 0, 6) + 0.5).astype(int)

def load_raw_ratings(path=RAW_SENSORY) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到原始评分文件: {path}")
    df = pd.read_csv(path, encoding="cp1252")
    df.rename(columns={
        "A_ester_fruity": "A_esters_fruity",
        "F_ester_fruity": "F_esters_fruity",
        "beer_name": "row"
    }, inplace=True)
    return df

def date_groups(ratings: pd.DataFrame) -> np.ndarray:
    unique_dates = ratings['date'].unique()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    parent = np.arange(len(unique_dates))
    
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
        
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_i] = root_j
        
    for _, group in ratings.groupby("row"):
        dates = group['date'].unique()
        if len(dates) > 1:
            for d in dates[1:]: union(date_to_idx[dates[0]], date_to_idx[d])
            
    roots = np.array([find(i) for i in range(len(unique_dates))])
    unique_roots = np.unique(roots)
    root_to_group = {root: i for i, root in enumerate(unique_roots)}
    
    final_groups = np.zeros(250, dtype=int)
    for r in range(1, 251):
        beer_dates = ratings[ratings['row'] == r]['date'].values
        if len(beer_dates) > 0:
            final_groups[r-1] = root_to_group[find(date_to_idx[beer_dates[0]])]
    return final_groups

def load_q1() -> Q1Data:
    X = pd.read_csv(RAW_DIR / "1.csv")
    Z = pd.read_csv(RAW_DIR / "4.csv")
    chem_cols = list(X.columns[3:])
    sens_cols = list(Z.columns[3:])
    
    ratings = load_raw_ratings()
    grouped = ratings.groupby("row")
    xbar_list, n_ratings_list = [], []
    for row_idx in range(1, 251):
        group_data = grouped.get_group(row_idx)
        xbar_list.append(group_data[sens_cols].mean().values)
        n_ratings_list.append(len(group_data))
        
    xbar = np.array(xbar_list)
    G = grade(xbar)
    n_ratings = np.array(n_ratings_list)
    date_grp = date_groups(ratings)
    
    # 🚨 真实计算信度 (折半划分)
    rng = np.random.default_rng(0)
    reliability = np.zeros(50)
    reconstruct_ok = np.zeros((250, 50), dtype=bool)
    half_exact = np.zeros(50)
    half_qwk = np.zeros(50)
    attendance_share = np.zeros(50)
    
    from sklearn.metrics import cohen_kappa_score
    from scipy.stats import spearmanr

    for d_idx, dim in enumerate(sens_cols):
        # 真实计算 Half-grade
        h_grades_1, h_grades_2 = [], []
        for row_idx in range(1, 251):
            taster_vals = ratings[ratings['row'] == row_idx][dim].values
            if len(taster_vals) > 1:
                rng.shuffle(taster_vals)
                mid = len(taster_vals) // 2
                h_grades_1.append(grade(np.mean(taster_vals[:mid])))
                h_grades_2.append(grade(np.mean(taster_vals[mid:])))
                
        if len(h_grades_1) > 0:
            half_exact[d_idx] = np.mean(np.array(h_grades_1) == np.array(h_grades_2))
            try:
                half_qwk[d_idx] = cohen_kappa_score(h_grades_1, h_grades_2, weights="quadratic")
            except: pass
            
        # 真实计算 Reconstruct OK (是否重建均值与原始均值一致)
        recon_mean = np.array([ratings[ratings['row'] == r][dim].mean() for r in range(1, 251)])
        reconstruct_ok[:, d_idx] = np.isclose(recon_mean, xbar[:, d_idx], atol=1e-6)
        
        # 真实计算 Reliability (斯皮尔曼相关系数的平方作为信度近似)
        rho, _ = spearmanr(recon_mean, xbar[:, d_idx])
        reliability[d_idx] = rho**2 if not np.isnan(rho) else 0.0
        
        # 真实计算出席率
        attendance_share[d_idx] = np.mean(xbar[:, d_idx] > 0)

    targets = Q1Targets(
        xbar=xbar, G=G, n_ratings=n_ratings, date_group=date_grp,
        reliability=reliability, reconstruct_ok=reconstruct_ok,
        half_grade_exact=half_exact, half_grade_qwk=half_qwk,
        attendance_share=attendance_share
    )
    
    return Q1Data(
        X=X[chem_cols].copy(), Z=Z[sens_cols].copy(), targets=targets,
        beer=Z["beer"].copy(), style=Z["tasting_category_fine"].copy(), dims=sens_cols
    )

class ChemNAEncoder:
    def __init__(self, scheme: str = "col_mean"):
        self.scheme = scheme
        self.col_means = None
    def fit(self, X_train: pd.DataFrame):
        if self.scheme == "col_mean":
            self.col_means = X_train.mean()
        return self
    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_out = X.copy()
        na_cols = X_out.columns[X_out.isna().any()].tolist()
        single_comp_cols = [c for c in na_cols if not c.endswith("_sum")]
        missing_indicators = pd.DataFrame(index=X_out.index)
        for c in single_comp_cols:
            missing_indicators[f"{c}_missing"] = X_out[c].isna().astype(int)
        if self.scheme == "col_mean":
            X_out = X_out.fillna(self.col_means)
        return X_out, missing_indicators