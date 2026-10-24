import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.convert import to_score, NO_ANCHOR_COLS
from src.q1.uncertainty import calc_score_distribution

def main():
    print("正在加载 Z 值预测结果和零点参数...")
    data = load_q1()
    
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    zeros = panels_data['zeros']
    
    oof_data = np.load(PROCESSED_DIR / "q1_main_oof_demo.npz", allow_pickle=True)
    
    print("正在进行 T3 转换与 T4 区间估计 (m=3)...")
    m = 3.0
    
    # 存储容器
    final_scores = {}
    ci_lower = {}
    ci_upper = {}
    all_probs = {} # 存概率
    
    np.random.seed(SEED) # 固定随机种子
    
    for dim in data.dims:
        z_pred = oof_data[dim]
        z_true = data.Z[dim].values
        
        # 提取零点与 M
        dim_idx = data.dims.index(dim)
        col_zeros = zeros[:, dim_idx]
        col_zeros = col_zeros[~np.isnan(col_zeros)]
        z0_tilde = np.median(col_zeros) if len(col_zeros) > 0 else np.nan
        M = np.max(np.abs(z_true))
        
        # 计算残差 (真实值 - 预测值)
        residuals = z_true - z_pred
        
        # T4: 计算概率、置信区间和最终评分
        score, probs, lo, hi = calc_score_distribution(
            z_pred, residuals, dim, m, z0_tilde, M
        )
        
        final_scores[dim] = score
        ci_lower[dim] = lo
        ci_upper[dim] = hi
        for k, v in probs.items():
            all_probs[f"{dim}_{k}"] = v
            
    # 保存 .npz
    save_path = PROCESSED_DIR / "q1_scores_full.npz"
    np.savez_compressed(
        save_path, 
        **final_scores, 
        **{f"lo_{k}": v for k, v in ci_lower.items()},
        **{f"hi_{k}": v for k, v in ci_upper.items()},
        **all_probs,
        seed=SEED
    )
    
    # 导出交付用的 CSV
    df_scores = pd.DataFrame(final_scores)
    df_scores.insert(0, "beer", data.beer)
    df_scores.to_csv(PROCESSED_DIR / "q1_final_scores.csv", index=False)
    
    print(f"✅ T3 & T4 运行完毕！")
    print(f"带置信区间和概率的数据已保存至: {save_path}")

if __name__ == "__main__":
    main()