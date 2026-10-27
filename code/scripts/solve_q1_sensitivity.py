import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED, RAW_DIR
from src.q1.data import load_q1, ChemNAEncoder, load_raw_ratings
from src.q1.splits import outer_splits
from src.q1.model import HistGradientBoostingRegressor, monotone_cst
from src.q1.discretize import grade
from src.q1.metrics import grade_mae, qwk_by_dim

def bootstrap_ci(a, b, groups, n_boot=1000):
    """按品评团组聚类 Bootstrap 计算差值 CI"""
    unique_groups = np.unique(groups)
    diffs = []
    for _ in range(n_boot):
        boot_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in boot_groups])
        diffs.append(grade_mae(a[idx], b[idx]))
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def main():
    print("正在执行 T6 全量敏感性分析...")
    data = load_q1()
    q1_data = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    xbar = q1_data['xbar']
    G_obs = q1_data['G']
    date_group = q1_data['date_group']
    
    main_oof = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=True)
    chosen_params = json.loads(str(main_oof['chosen_params']))
    main_G = np.load(PROCESSED_DIR / "q1_scores.npz", allow_pickle=True)['G_hat']
    
    splits = outer_splits(date_group, data.style, n_splits=5, n_repeats=1, seed=SEED)
    outer_folds = splits[0]
    
    arms = {
        "median_ind": ("ChemNAEncoder", "median_ind", "all"),
        "native_nan": ("ChemNAEncoder", "native_nan", "all"),
        "floor_ind": ("ChemNAEncoder", "floor_ind", "all"),
        "no_barnyard": ("DropCol", None, "barnyard"),
        "no_sum": ("DropCol", None, "sum"),
        "mono_off": ("Mono", False, "all")
    }
    
    results = {}
    
    for arm_name, (arm_type, scheme, target) in arms.items():
        print(f"  正在测试对照臂: {arm_name} ...")
        
        # 准备数据
        if arm_type == "ChemNAEncoder":
            encoder = ChemNAEncoder(scheme=scheme)
            encoder.fit(data.X)
            X_proc, miss_inds = encoder.transform(data.X)
            X_proc = pd.concat([X_proc, miss_inds], axis=1)
            X_arr = X_proc.values
            dims_to_run = data.dims
            mono_flag = True
            
        elif arm_type == "DropCol":
            X_proc = data.X.copy()
            dims_to_run = data.dims
            if target == "barnyard":
                dims_to_run = [d for d in dims_to_run if d != "barnyard"]
            elif target == "sum":
                cols_to_drop = [c for c in X_proc.columns if c.endswith("_sum")]
                X_proc = X_proc.drop(columns=cols_to_drop)
            # 简单均值填补
            X_proc = X_proc.fillna(X_proc.mean())
            X_arr = X_proc.values
            mono_flag = True
            
        elif arm_type == "Mono":
            encoder = ChemNAEncoder(scheme="col_mean")
            encoder.fit(data.X)
            X_proc, miss_inds = encoder.transform(data.X)
            X_proc = pd.concat([X_proc, miss_inds], axis=1)
            X_arr = X_proc.values
            dims_to_run = data.dims
            mono_flag = False
            
        arm_G = np.zeros((250, 50), dtype=int)
        
        for dim in dims_to_run:
            dim_idx = data.dims.index(dim)
            best_params = chosen_params[dim]
            mono_cst = monotone_cst(X_proc.columns.tolist(), dim, mono_flag)
            
            for train_idx, test_idx in outer_folds:
                model = HistGradientBoostingRegressor(**best_params, random_state=0, monotonic_cst=mono_cst)
                model.fit(X_arr[train_idx], xbar[train_idx, dim_idx])
                pred = model.predict(X_arr[test_idx])
                arm_G[test_idx, dim_idx] = grade(pred)
                
        if target == "barnyard":
            arm_G[:, data.dims.index("barnyard")] = main_G[:, data.dims.index("barnyard")]
            
        mae = grade_mae(G_obs, arm_G)
        change = np.mean(arm_G != main_G)
        
        # 计算配对差 CI
        diff_main = np.abs(G_obs - main_G).mean(axis=1)
        diff_arm = np.abs(G_obs - arm_G).mean(axis=1)
        lo, hi = bootstrap_ci(diff_main, diff_arm, date_group, n_boot=200)
        
        results[arm_name] = {
            "mae": float(mae),
            "change_ratio": float(change),
            "delta_ci": [float(lo), float(hi)]
        }
        print(f"    -> MAE: {mae:.4f}, 改变比例: {change:.2%}, CI: [{lo:.4f}, {hi:.4f}]")

    np.savez_compressed(PROCESSED_DIR / "q1_sensitivity.npz", **{k: json.dumps(v) for k, v in results.items()})
    print("✅ T6 敏感性分析完成。")

if __name__ == "__main__":
    main()