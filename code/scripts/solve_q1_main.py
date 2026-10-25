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
    
    DIM_GROUPS = {
        "Aroma": [c for c in data.dims if c.startswith("A_")],
        "Flavour": [c for c in data.dims if c.startswith("F_")],
        "Defect": ["dms", "metallic", "X4vg", "t2n", "diacetyl", "stale_hops"],
        "Spice": ["orange", "coriander", "clove"],
        "Sour": ["lactic", "acetic", "barnyard"],
        "Basic": ["body", "alcohol", "bitternes", "sweetness", "co2", "acidity", "aftertaste", "overall"]
    }
    
    # 初始化存储矩阵
    oof_z_all = np.zeros((5, 250, 50))
    inner_z_all = np.full((5, 250, 50), np.nan) # 存储内层预测
    fold_id = np.zeros((5, 250), dtype=int)
    final_zhat = np.zeros((250, 50))
    chosen_params_all = {}
    
    start_time = time.time()
    
    # 填入 fold_id
    for run in range(5):
        for fold_idx, (_, test_idx) in enumerate(splits[run]):
            fold_id[run, test_idx] = fold_idx
    
    for group_name, dims in DIM_GROUPS.items():
        print(f"\n>>> 正在处理维度组: {group_name} <<<")
        best_params, best_use_mono = find_best_params_for_group(
            X_proc, data.Z, panels, splits[0], inner_splits_func, dims
        )
        
        for dim in dims:
            print(f"    正在跑维度: {dim} ...")
            dim_idx = data.dims.index(dim)
            preds_list = []
            
            for run in range(5):
                # 注意：这里接收两个返回值
                pred_run, inner_run = run_outer_cv(
                    X_proc, data.Z, panels, splits[run], inner_splits_func, dim, best_params, best_use_mono
                )
                preds_list.append(pred_run)
                oof_z_all[run, :, dim_idx] = pred_run 
                inner_z_all[run, :, dim_idx] = inner_run # 存入内层预测
                
            # 5次平均作为最终 zhat
            final_zhat[:, dim_idx] = np.mean(preds_list, axis=0)
            chosen_params_all[dim] = best_params
            
    save_path = PROCESSED_DIR / "q1_main_oof_v2.npz"
    np.savez_compressed(
        save_path, 
        zhat=final_zhat,
        oof_z=oof_z_all,
        inner_z=inner_z_all, # 终于存进去了！
        fold_id=fold_id,
        chosen_params=json.dumps(chosen_params_all),
        dims=np.array(data.dims),
        seed=SEED, 
        meta=json.dumps({"python": sys.version, "sklearn": sklearn.__version__})
    )
    print(f"\n✅ 契约版 T2 运行完毕！耗时: {time.time() - start_time:.1f} 秒")

if __name__ == "__main__":
    main()