import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1, ChemNAEncoder
from src.q1.splits import outer_splits, inner_splits
from src.q1.model import PARAM_GRID, HistGradientBoostingRegressor, monotone_cst

def main():
    print("正在加载 T1 的数据产物与化学特征...")
    data = load_q1()
    
    # 加载 T1 算好的真实目标与日期组
    q1_data = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    xbar = q1_data['xbar'] # (250, 50)
    date_group = q1_data['date_group'] # (250,)
    
    # 化学特征预处理
    encoder = ChemNAEncoder(scheme="col_mean")
    encoder.fit(data.X)
    X_proc, missing_inds = encoder.transform(data.X)
    X_proc = pd.concat([X_proc, missing_inds], axis=1) 
    
    # 获取外层 5 次重复、每次 5 折的划分
    splits = outer_splits(date_group, data.style, n_splits=5, n_repeats=5, seed=SEED)
    
    DIM_GROUPS = {
        "Aroma": [c for c in data.dims if c.startswith("A_")],
        "Flavour": [c for c in data.dims if c.startswith("F_")],
        "Defect": ["dms", "metallic", "X4vg", "t2n", "diacetyl", "stale_hops"],
        "Spice": ["orange", "coriander", "clove"],
        "Sour": ["lactic", "acetic", "barnyard"],
        "Basic": ["body", "alcohol", "bitternes", "sweetness", "co2", "acidity", "aftertaste", "overall"]
    }
    
    oof_mu = np.zeros((5, 250, 50)) # 5次重复的外层预测
    inner_mu = np.full((5, 5, 250, 50), np.nan) # 5次重复，5个折，内层预测
    fold_id = np.zeros((5, 250), dtype=int)
    chosen_params_all = {}
    
    start_time = time.time()
    
    # 预先填入 fold_id
    for run in range(5):
        for fold_idx, (_, test_idx) in enumerate(splits[run]):
            fold_id[run, test_idx] = fold_idx

    print("\n开始嵌套交叉验证训练...")
    feature_names = X_proc.columns.tolist()

    for group_name, dims in DIM_GROUPS.items():
        print(f"\n>>> 正在处理维度组: {group_name} ({len(dims)} 个维度) <<<")
        
        # 1. 组内选参 (通过内层平均 RMSE/SD)
        best_score = np.inf
        best_params = PARAM_GRID[0]
        best_use_mono = False
        
        # 取第一折训练集作为代表来搜参（节省时间）
        train_idx_0, _ = splits[0][0]
        inner_folds = inner_splits(train_idx_0, date_group, data.style, SEED)
        
        for params in PARAM_GRID:
            for use_mono in [True, False]:
                group_inner_scores = []
                for in_train_idx, in_val_idx in inner_folds:
                    real_in_train = train_idx_0[in_train_idx]
                    real_in_val = train_idx_0[in_val_idx]
                    
                    X_in_train, X_in_val = X_proc.iloc[real_in_train], X_proc.iloc[real_in_val]
                    
                    for dim in dims:
                        dim_idx = data.dims.index(dim)
                        y_in_train = xbar[real_in_train, dim_idx]
                        y_in_val = xbar[real_in_val, dim_idx]
                        
                        mono_cst = monotone_cst(feature_names, dim, use_mono)
                        model = HistGradientBoostingRegressor(**params, random_state=0, monotonic_cst=mono_cst)
                        model.fit(X_in_train, y_in_train)
                        pred = model.predict(X_in_val)
                        
                        rmse = np.sqrt(np.mean((y_in_val - pred)**2))
                        sd = np.std(y_in_val)
                        group_inner_scores.append(rmse / (sd + 1e-9))
                        
                mean_score = np.mean(group_inner_scores)
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params
                    best_use_mono = use_mono
                    
        print(f"  -> 组最优参数选定: {best_params}, 单调约束: {best_use_mono}")
        
        # 2. 用选好的参数跑外层预测
        for dim in dims:
            print(f"    正在跑维度: {dim} ...")
            dim_idx = data.dims.index(dim)
            mono_cst = monotone_cst(feature_names, dim, best_use_mono)
            chosen_params_all[dim] = best_params
            
            for run in range(5):
                for fold_idx, (train_idx, test_idx) in enumerate(splits[run]):
                    X_train, X_test = X_proc.iloc[train_idx], X_proc.iloc[test_idx]
                    y_train = xbar[train_idx, dim_idx]
                    
                    # 外层预测
                    model = HistGradientBoostingRegressor(**best_params, random_state=0, monotonic_cst=mono_cst)
                    model.fit(X_train, y_train)
                    oof_mu[run, test_idx, dim_idx] = model.predict(X_test)
                    
                    # 内层预测 (用于 T3 的残差池)
                    inner_folds_run = inner_splits(train_idx, date_group, data.style, SEED + 100 + run)
                    for in_train_idx, in_val_idx in inner_folds_run:
                        real_in_train = train_idx[in_train_idx]
                        real_in_val = train_idx[in_val_idx]
                        
                        X_in_train = X_proc.iloc[real_in_train]
                        X_in_val = X_proc.iloc[real_in_val]
                        y_in_train = xbar[real_in_train, dim_idx]
                        
                        m_in = HistGradientBoostingRegressor(**best_params, random_state=0, monotonic_cst=mono_cst)
                        m_in.fit(X_in_train, y_in_train)
                        inner_mu[run, fold_idx, real_in_val, dim_idx] = m_in.predict(X_in_val)

    # 保存 T2 产物
    save_path = PROCESSED_DIR / "q1_main_oof.npz"
    meta_info = {"python": sys.version, "sklearn": sklearn.__version__}
    
    np.savez_compressed(
        save_path, 
        oof_mu=oof_mu,
        inner_mu=inner_mu,
        fold_id=fold_id,
        chosen_params=json.dumps(chosen_params_all),
        seed=SEED, 
        meta=json.dumps(meta_info)
    )
    
    print(f"\n✅ T2 运行完毕！耗时: {time.time() - start_time:.1f} 秒")
    print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    main()