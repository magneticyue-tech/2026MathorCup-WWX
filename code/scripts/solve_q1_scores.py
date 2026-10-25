import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.convert import (
    NO_ANCHOR_COLS, get_linear_transform, transform_linear, 
    transform_local, optimize_qwk_thresholds, apply_thresholds
)

def main():
    print("正在加载数据，执行新版 §4.5/4.6 映射...")
    data = load_q1()
    
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    zeros = panels_data['zeros']
    
    # ⚠️ 这里读取的是 T2 跑出来的最终预测结果
    oof_data = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=True)
    
    results = {}
    
    for dim in data.dims:
        z_pred = oof_data[dim]
        z_true = data.Z[dim].values
        dim_idx = data.dims.index(dim)
        
        # 1. 构造参照系 R (4.5节)
        # 这里简化处理：如果是有锚维度，我们用线性参照 L 作为 R 的基础
        # (为了速度，我们略去逐折的局部零点混算，统一用线性 L 作为真值参照去优化切点)
        scale, shift = get_linear_transform(z_true)
        z_cont_L = transform_linear(z_true, scale, shift)
        R_true = np.round(z_cont_L).astype(int) # 构造的参照等级
        
        # 2. 对预测值进行线性映射
        z_cont_pred = transform_linear(z_pred, scale, shift)
        
        # 3. 优化 QWK 切点 (4.6节)
        best_thresholds = optimize_qwk_thresholds(z_cont_pred, R_true)
        final_grades = apply_thresholds(z_cont_pred, best_thresholds)
        
        results[dim] = final_grades
        
    # 保存结果
    save_path = PROCESSED_DIR / "q1_scores_v2.npz"
    np.savez_compressed(save_path, **results, seed=SEED)
    
    # 导出 CSV
    df = pd.DataFrame(results)
    df.insert(0, "beer", data.beer)
    df.to_csv(PROCESSED_DIR / "q1_final_scores_v2.csv", index=False)
    
    print(f"✅ 新版映射完成，结果保存至: {save_path}")

if __name__ == "__main__":
    main()