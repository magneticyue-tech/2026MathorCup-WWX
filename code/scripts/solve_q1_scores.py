import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.convert import to_score, calc_m_cap, NO_ANCHOR_COLS

def main():
    print("正在加载 Z 值预测结果和零点参数...")
    data = load_q1()
    
    # 读取 T1 的结果 (零点)
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    zeros = panels_data['zeros']
    zero_panel_ids = panels_data['zero_panel_ids']
    
    # 读取 T2 的结果 (Z值预测)
    # 注意：如果刚才跑出的文件名叫 q1_main_oof_demo.npz，这里就改一下
    oof_data = np.load(PROCESSED_DIR / "q1_main_oof_demo.npz", allow_pickle=True)
    
    print("正在进行 0-6 分转换 (m=3)...")
    m = 3.0
    scores_dict = {}
    
    for dim in data.dims:
        z_pred = oof_data[dim]
        z_true = data.Z[dim].values
        
        # 提取该维度的中位零点
        dim_idx = data.dims.index(dim)
        col_zeros = zeros[:, dim_idx]
        col_zeros = col_zeros[~np.isnan(col_zeros)]
        z0_tilde = np.median(col_zeros) if len(col_zeros) > 0 else np.nan
        
        # 计算 M (训练集最大绝对值)
        M = np.max(np.abs(z_true))
        
        # 转换！
        raw_score = to_score(z_pred, dim, m=m, z0_tilde=z0_tilde, M=M)
        
        # 四舍五入为 0-6 的整数
        scores_dict[dim] = np.round(raw_score).astype(int)
        
    # 保存为 npz
    save_path = PROCESSED_DIR / "q1_scores_m3.npz"
    np.savez_compressed(save_path, **scores_dict, seed=SEED)
    
    # 顺便导出一份 CSV 方便你直接用 Excel 打开看！
    df_scores = pd.DataFrame(scores_dict)
    df_scores.insert(0, "beer", data.beer)
    csv_path = PROCESSED_DIR / "q1_final_scores.csv"
    df_scores.to_csv(csv_path, index=False)
    
    print(f"✅ T3 转换完毕！")
    print(f"最终评分已保存至: {save_path}")
    print(f"CSV 版本 (可用 Excel 查看): {csv_path}")

if __name__ == "__main__":
    main()