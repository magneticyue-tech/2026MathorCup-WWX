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
    """
    基于方法论 2.1 和 7.1 的缺失值编码器。
    """
    def __init__(self, scheme: str = "main"):
        self.scheme = scheme
        self.col_means = None
        
    def fit(self, X_train: pd.DataFrame):
        if self.scheme == "col_mean":
            self.col_means = X_train.mean()
        return self

    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_out = X.copy()
        
        # 找出包含缺失值的单化合物列（排除 _sum 列）
        na_cols = X_out.columns[X_out.isna().any()].tolist()
        single_comp_cols = [c for c in na_cols if not c.endswith("_sum")]
        
        # 按方法论 2.1：为 8 个受影响的列构建缺失指示变量
        missing_indicators = pd.DataFrame(index=X_out.index)
        for c in single_comp_cols:
            missing_indicators[f"{c}_missing"] = X_out[c].isna().astype(int)

        if self.scheme == "main":
            # 主设定：单化合物 NA 填 0，_sum 保留 NaN
            X_out[single_comp_cols] = X_out[single_comp_cols].fillna(0)
            
        elif self.scheme == "amyl_nan":
            # 敏感性实验：amyl.acetate 保留 NaN
            fill_cols = [c for c in single_comp_cols if c != "amyl.acetate"]
            X_out[fill_cols] = X_out[fill_cols].fillna(0)
            
        elif self.scheme == "all_nan":
            # 敏感性实验：全部单化合物保留 NaN
            pass 
            
        elif self.scheme == "col_mean":
            # 敏感性实验：均值填补
            if self.col_means is None:
                raise ValueError("必须先调用 fit() 计算均值！")
            X_out = X_out.fillna(self.col_means)
            
        return X_out, missing_indicators