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
from src.q1.model import find_best_params_for_group, run_outer_cv

def main():
    print("正在加载数据...")
    data = load_q1()
    
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    
    encoder = ChemNAEncoder(scheme="main")
    encoder.fit(data.X)
    X_proc, missing_inds = encoder.transform(data.X)
    X_proc = pd.concat([X_proc, missing_inds], axis=1) 
    
    splits = outer_splits(panels, data.style, n_splits=5, n_repeats=5, seed=SEED)
    inner_splits_func = lambda train_idx: inner_splits(train_idx, panels, data.style, SEED)
    
    # 定义 6 个维度组
    DIM_GROUPS = {
        "Aroma": [c for c in data.dims if c.startswith("A_")],
        "Flavour": [c for c in data.dims if c.startswith("F_")],
        "Defect": ["dms", "metallic", "X4vg", "t2n", "diacetyl", "stale_hops"],
        "Spice": ["orange", "coriander", "clove"],
        "Sour": ["lactic", "acetic", "barnyard"],
        "Basic": ["body", "alcohol", "bitternes", "sweetness", "co2", "acidity", "aftertaste", "overall"]
    }
    
    oof_results = {}
    chosen_params_all = {}
    
    start_time = time.time()
    
    for group_name, dims in DIM_GROUPS.items():
        print(f"\n>>> 正在处理维度组: {group_name} (包含 {len(dims)} 个维度) <<<")
        
        # 1. 该组只搜一次参数 (用第一个维度代表)
        print(f"  -> 正在为该组进行网格搜索...")
        best_params, best_use_mono = find_best_params_for_group(
            X_proc, data.Z, panels, splits[0], inner_splits_func, dims
        )
        print(f"  -> 组最优参数选定: {best_params}, 单调约束: {best_use_mono}")
        
        # 2. 把参数应用到该组的所有维度
        for dim in dims:
            print(f"    正在跑维度: {dim} ...")
            preds_list = []
            
            for run in range(5):
                outer_folds = splits[run]
                pred_run = run_outer_cv(
                    X_proc, data.Z, panels, outer_folds, dim, best_params, best_use_mono
                )
                preds_list.append(pred_run)
                
            oof_results[dim] = np.mean(preds_list, axis=0)
            chosen_params_all[dim] = best_params
            
    save_path = PROCESSED_DIR / "q1_main_oof.npz"
    meta_info = {"python": sys.version, "sklearn": sklearn.__version__}
    
    np.savez_compressed(
        save_path, 
        **oof_results, 
        chosen_params=json.dumps(chosen_params_all),
        seed=SEED, 
        meta=json.dumps(meta_info)
    )
    
    print(f"\n✅ T2 极速版运行完毕！耗时: {time.time() - start_time:.1f} 秒")

if __name__ == "__main__":
    main()