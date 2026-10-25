import pandas as pd
import numpy as np
from dataclasses import dataclass
from .paths import RAW_DIR

@dataclass
class Q1Data:
    X: pd.DataFrame
    Z: pd.DataFrame
    beer: pd.Series
    style: pd.Series
    dims: list

def load_q1(raw_dir=RAW_DIR) -> Q1Data:
    """加载并切分数据"""
    X = pd.read_csv(raw_dir / "1.csv")
    Z = pd.read_csv(raw_dir / "4.csv")
    
    chem_cols = list(X.columns[3:])
    sens_cols = list(Z.columns[3:])
    
    # 按方法论 E1：行序一致，直接取列
    return Q1Data(
        X=X[chem_cols].copy(),
        Z=Z[sens_cols].copy(),
        beer=Z["beer"].copy(),
        style=Z["tasting_category_fine"].copy(),
        dims=sens_cols
    )

class ChemNAEncoder:
    def __init__(self, scheme: str = "main"):
        self.scheme = scheme
        self.col_means_main = None # 新增：保存训练折的均值
        
    def fit(self, X_train: pd.DataFrame):
        # 找出包含缺失值的单化合物列
        na_cols = X_train.columns[X_train.isna().any()].tolist()
        self.single_comp_cols = [c for c in na_cols if not c.endswith("_sum")]
        
        if self.scheme == "main":
            # 方法论要求：只用训练折的非缺失观测计算列均值
            self.col_means_main = X_train[self.single_comp_cols].mean()
        elif self.scheme == "col_mean":
            self.col_means = X_train.mean()
        return self

    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_out = X.copy()
        
        # 构建缺失指示变量
        missing_indicators = pd.DataFrame(index=X_out.index)
        for c in self.single_comp_cols:
            missing_indicators[f"{c}_missing"] = X_out[c].isna().astype(int)

        if self.scheme == "main":
            # 修复：使用训练折均值填充
            X_out[self.single_comp_cols] = X_out[self.single_comp_cols].fillna(self.col_means_main)
        elif self.scheme == "amyl_nan":
            fill_cols = [c for c in self.single_comp_cols if c != "amyl.acetate"]
            X_out[fill_cols] = X_out[fill_cols].fillna(self.col_means_main[fill_cols])
        elif self.scheme == "all_nan":
            pass 
        elif self.scheme == "col_mean":
            X_out = X_out.fillna(self.col_means)
            
        return X_out, missing_indicators