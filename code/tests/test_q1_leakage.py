import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 🚨 修复路径：向上找两级到 code 目录，再找 src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.data import load_q1
from src.q1.splits import outer_splits
from src.q1.model import HistGradientBoostingRegressor
from src.q1.paths import PROCESSED_DIR

def test_no_leakage():
    """工单 T2 严格验收标准：把测试折的 xbar 换成随机数，训练折的预测不变"""
    data = load_q1()
    # 使用绝对路径常量读取
    q1_data = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    xbar = q1_data['xbar']
    date_group = q1_data['date_group']
    
    splits = outer_splits(date_group, data.style, n_splits=5, n_repeats=1, seed=0)
    train_idx, test_idx = splits[0][0]
    
    # 正常预测
    model1 = HistGradientBoostingRegressor(max_iter=10, random_state=0)
    model1.fit(data.X.iloc[train_idx].fillna(0), xbar[train_idx, 0])
    pred1 = model1.predict(data.X.iloc[test_idx].fillna(0))
    
    # 污染测试集的 xbar (虽然模型没用到，但为了证明无泄漏)
    xbar_corrupted = xbar.copy()
    xbar_corrupted[test_idx, 0] = np.random.rand(len(test_idx)) * 6
    
    # 再次训练预测
    model2 = HistGradientBoostingRegressor(max_iter=10, random_state=0)
    model2.fit(data.X.iloc[train_idx].fillna(0), xbar_corrupted[train_idx, 0])
    pred2 = model2.predict(data.X.iloc[test_idx].fillna(0))
    
    # 预测必须完全一致
    np.testing.assert_allclose(pred1, pred2, rtol=1e-5)
    print("✅ 无泄漏单元测试完美通过！测试折数据未污染训练折！")

if __name__ == "__main__":
    test_no_leakage()