import pandas as pd
import numpy as np
from dataclasses import dataclass
from .paths import RAW_DIR, RAW_SENSORY

@dataclass
class Q1Targets:
    xbar: np.ndarray      # 250x50 真实平均分
    G: np.ndarray         # 250x50 int 取整等级
    n_ratings: np.ndarray # 250 每款酒的评分人数
    date_group: np.ndarray# 250 每款酒对应的日期组

@dataclass
class Q1Data:
    X: pd.DataFrame
    Z: pd.DataFrame
    targets: Q1Targets
    beer: pd.Series
    style: pd.Series
    dims: list

def grade(v: np.ndarray) -> np.ndarray:
    """取整逻辑: q(v) = floor(clip(v,0,6) + 0.5)"""
    return np.floor(np.clip(v, 0, 6) + 0.5).astype(int)

def load_raw_ratings(path=RAW_SENSORY) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到原始评分文件: {path}")
    # 根据工单要求使用 cp1252 编码
    df = pd.read_csv(path, encoding="cp1252")
    df.rename(columns={
        "A_ester_fruity": "A_esters_fruity",
        "F_ester_fruity": "F_esters_fruity",
        "beer_name": "row"
    }, inplace=True)
    return df

def date_groups(ratings: pd.DataFrame) -> np.ndarray:
    """使用并查集合并相同啤酒的不同日期，返回 250 长度的组编号"""
    unique_dates = ratings['date'].unique()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    n_dates = len(unique_dates)

    # 并查集初始化
    parent = np.arange(n_dates)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # 遍历每款酒，如果它在多个日期出现，就把这些日期合并
    for row_idx, group in ratings.groupby("row"):
        dates = group['date'].unique()
        if len(dates) > 1:
            base_idx = date_to_idx[dates[0]]
            for d in dates[1:]:
                union(base_idx, date_to_idx[d])

    # 将根节点映射为连续的组编号 (0, 1, 2...)
    roots = np.array([find(i) for i in range(n_dates)])
    unique_roots = np.unique(roots)
    root_to_group = {root: i for i, root in enumerate(unique_roots)}

    # 为 250 款酒分配组标签
    final_groups = np.zeros(250, dtype=int)
    for r in range(1, 251):
        beer_dates = ratings[ratings['row'] == r]['date'].values
        if len(beer_dates) > 0:
            d = beer_dates[0]
            final_groups[r-1] = root_to_group[find(date_to_idx[d])]
            
    return final_groups

def load_q1() -> Q1Data:
    X = pd.read_csv(RAW_DIR / "1.csv")
    Z = pd.read_csv(RAW_DIR / "4.csv")
    chem_cols = list(X.columns[3:])
    sens_cols = list(Z.columns[3:])
    
    ratings = load_raw_ratings()
    
    grouped = ratings.groupby("row")
    xbar_list = []
    n_ratings_list = []
    for row_idx in range(1, 251):
        group_data = grouped.get_group(row_idx)
        xbar_list.append(group_data[sens_cols].mean().values)
        n_ratings_list.append(len(group_data))
        
    xbar = np.array(xbar_list)
    G = grade(xbar)
    n_ratings = np.array(n_ratings_list)
    
    # 推断日期组
    date_grp = date_groups(ratings)
    
    targets = Q1Targets(xbar=xbar, G=G, n_ratings=n_ratings, date_group=date_grp)
    
    return Q1Data(
        X=X[chem_cols].copy(),
        Z=Z[sens_cols].copy(),
        targets=targets,
        beer=Z["beer"].copy(),
        style=Z["tasting_category_fine"].copy(),
        dims=sens_cols
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
        elif self.scheme == "median_ind":
            medians = X_out[single_comp_cols].median()
            X_out[single_comp_cols] = X_out[single_comp_cols].fillna(medians)
        elif self.scheme == "native_nan":
            pass 
        elif self.scheme == "floor_ind":
            X_out[single_comp_cols] = X_out[single_comp_cols].fillna(0)
            
        return X_out, missing_indicators