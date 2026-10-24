import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn

# 添加路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1, ChemNAEncoder
from src.q1.splits import outer_splits, inner_splits
from src.q1.model import nested_cv_predict

def main():
    print("正在加载数据...")
    data = load_q1()
    
    # 加载 T1 生成的场次和零点数据
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    zeros = panels_data['zeros']
    zero_panel_ids = panels_data['zero_panel_ids']
    at_zero = panels_data['at_zero']
    
    # 预处理
    encoder = ChemNAEncoder(scheme="main")
    encoder.fit(data.X)
    X_proc, missing_inds = encoder.transform(data.X)
    X_proc = pd.concat([X_proc, missing_inds], axis=1) # 拼上缺失指示器
    
    # 划分外层 CV (队长要求的 5 次重复)
    splits = outer_splits(panels, data.style, n_splits=5, n_repeats=1, seed=SEED)
    outer_folds = splits[0]
  
    # 定义内层划分的 lambda 函数
    inner_splits_func = lambda train_idx: inner_splits(train_idx, panels, data.style, SEED)
    
    dims_to_run = data.dims 
    
    oof_results = {}
    
    start_time = time.time()
    for i, dim in enumerate(dims_to_run):
        print(f"正在训练维度 {i+1}/{len(dims_to_run)}: {dim} ...")
        
        # 判断是否使用 Hurdle 模型 (工单 3.4)
        use_hurdle = False
        z0_tilde = np.nanmedian(zeros[:, data.dims.index(dim)]) if len(zero_panel_ids)>0 else 0
        
        # 提取当前维度的一维零点状态数组
        dim_idx = data.dims.index(dim)
        at_zero_col = at_zero[:, dim_idx]

        # 简化版逻辑：如果该维度处于零点的比例 > 30%，启用 Hurdle
        if at_zero_col.mean() > 0.3:
            use_hurdle = True
            print(f"  -> 发现维度 {dim} 零膨胀严重，启用 Hurdle 模型！")
        
        pred = nested_cv_predict(
            X_proc, data.Z, panels, outer_folds, inner_splits_func, 
            dim=dim, use_hurdle=use_hurdle, at_zero=at_zero_col, z0_tilde=z0_tilde # <-- 这里传一维的 at_zero_col
        )
        oof_results[dim] = pred

    # 保存结果
    save_path = PROCESSED_DIR / "q1_main_oof_demo.npz"
    
    meta_info = {"python": sys.version, "sklearn": sklearn.__version__}
    
    np.savez_compressed(
        save_path, 
        **oof_results, 
        seed=SEED, 
        meta=json.dumps(meta_info)
    )
    
    print(f"\n✅ T2 演示运行完毕！耗时: {time.time() - start_time:.1f} 秒")
    print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    main()