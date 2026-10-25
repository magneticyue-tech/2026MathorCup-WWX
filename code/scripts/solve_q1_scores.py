import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.convert import to_score, calc_m_cap, NO_ANCHOR_COLS
from src.q1.uncertainty import calc_score_distribution, conformal_halfwidth, score_interval, coverage

def main():
    print("正在加载 Z 值预测结果和零点参数...")
    data = load_q1()
    
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    zeros = panels_data['zeros']
    
    # 注意：这里读取的是刚跑完的 T2 全量结果
    oof_data = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=True)
    
    rng = np.random.default_rng(SEED)
    
    # m 的网格，加上一个 0 代表用 cap_d 算
    m_grid = [2.5, 3.0, 4.0, 0.0]
    
    for m_val in m_grid:
        m_name = "cap" if m_val == 0.0 else str(m_val)
        print(f"\n正在计算 m={m_name} 的转换...")
        
        final_scores = {}
        ci_lower = {}
        ci_upper = {}
        ref_scores = {}
        coverage_list = []
        
        for dim in data.dims:
            z_pred = oof_data[dim]
            z_true = data.Z[dim].values
            
            dim_idx = data.dims.index(dim)
            col_zeros = zeros[:, dim_idx]
            col_zeros = col_zeros[~np.isnan(col_zeros)]
            z0_tilde = np.median(col_zeros) if len(col_zeros) > 0 else np.nan
            M = np.max(np.abs(z_true))
            
            # 处理 cap
            current_m = m_val
            if current_m == 0.0:
                current_m = calc_m_cap(z_true, z0_tilde)
            
            # 1. 用 T4 的 Bootstrap 算概率
            residuals = z_true - z_pred
            score, probs = calc_score_distribution(z_pred, residuals, dim, current_m, z0_tilde, M, rng=rng)
            final_scores[dim] = score
            
            # 2. 用 T4 的 Split-Conformal 算区间
            q = conformal_halfwidth(residuals, alpha=0.1)
            lo_arr = np.zeros(len(z_pred))
            hi_arr = np.zeros(len(z_pred))
            for i in range(len(z_pred)):
                lo, hi = score_interval(z_pred[i], q, dim, current_m, z0_tilde, M)
                lo_arr[i] = lo
                hi_arr[i] = hi
                
            ci_lower[dim] = lo_arr
            ci_upper[dim] = hi_arr
            
            # 计算 z 空间覆盖率
            cov = coverage(z_true, z_pred, q)
            coverage_list.append(cov)
            
            # 3. 参考评分（用真实 Z 转换）
            ref_scores[dim] = to_score(z_true, dim, current_m, z0_tilde, M)
            
        # 保存结果
        save_path = PROCESSED_DIR / f"q1_scores_m{m_name}.npz"
        np.savez_compressed(
            save_path, 
            **final_scores, 
            **{f"lo_{k}": v for k, v in ci_lower.items()},
            **{f"hi_{k}": v for k, v in ci_upper.items()},
            **{f"ref_{k}": v for k, v in ref_scores.items()}, # <--- 修改这里
            seed=SEED
        )
        
        # 导出 CSV
        df_scores = pd.DataFrame(final_scores)
        df_scores.insert(0, "beer", data.beer)
        df_scores.to_csv(PROCESSED_DIR / f"q1_final_scores_m{m_name}.csv", index=False)
        
        print(f"✅ m={m_name} 完成！平均 z 空间覆盖率: {np.mean(coverage_list):.3f}")

if __name__ == "__main__":
    main()