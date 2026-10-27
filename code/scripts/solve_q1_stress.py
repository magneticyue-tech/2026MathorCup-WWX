import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1, ChemNAEncoder
from src.q1.splits import leave_one_style_splits, paper_split
from src.q1.model import HistGradientBoostingRegressor, monotone_cst
from src.q1.discretize import grade
from src.q1.metrics import grade_mae

def main():
    print("正在执行 T6 压力测试 (留一风格 & 70/30)...")
    data = load_q1()
    q1_data = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    xbar = q1_data['xbar']
    G_obs = q1_data['G']
    
    main_oof = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=True)
    chosen_params = json.loads(str(main_oof['chosen_params']))
    
    encoder = ChemNAEncoder(scheme="col_mean")
    encoder.fit(data.X)
    X_proc, miss_inds = encoder.transform(data.X)
    X_arr = pd.concat([X_proc, miss_inds], axis=1).values
    
    results = {}
    
    # 1. 留一风格
    style_splits = leave_one_style_splits(data.style)
    for train_idx, test_idx in style_splits:
        style_name = data.style.iloc[test_idx[0]]
        if len(test_idx) < 3: continue # 只测 >=3 款的
        
        arm_G = np.zeros((len(test_idx), 50), dtype=int)
        for dim_idx, dim in enumerate(data.dims):
            model = HistGradientBoostingRegressor(**chosen_params[dim], random_state=0, monotonic_cst=monotone_cst(data.X.columns.tolist() + list(pd.concat([X_proc, miss_inds], axis=1).columns[data.X.shape[1]:]), dim, True))
            model.fit(X_arr[train_idx], xbar[train_idx, dim_idx])
            arm_G[:, dim_idx] = grade(model.predict(X_arr[test_idx]))
            
        mae = grade_mae(G_obs[test_idx], arm_G)
        results[f"LOSO_{style_name}"] = float(mae)
        print(f"  留一风格 {style_name}: MAE = {mae:.4f}")

    # 2. 论文 70/30 划分
    train_idx, test_idx = paper_split(data.style, seed=0)
    arm_G = np.zeros((len(test_idx), 50), dtype=int)
    for dim_idx, dim in enumerate(data.dims):
        model = HistGradientBoostingRegressor(**chosen_params[dim], random_state=0)
        model.fit(X_arr[train_idx], xbar[train_idx, dim_idx])
        arm_G[:, dim_idx] = grade(model.predict(X_arr[test_idx]))
    mae_7030 = grade_mae(G_obs[test_idx], arm_G)
    results["paper_70_30"] = float(mae_7030)
    
    np.savez_compressed(PROCESSED_DIR / "q1_stress.npz", **{k: json.dumps(v) for k, v in results.items()})
    print("✅ T6 压力测试完成。")

if __name__ == "__main__":
    main()