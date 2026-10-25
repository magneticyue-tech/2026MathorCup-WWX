import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.convert import NO_ANCHOR_COLS

def main():
    print("正在生成最终交付产物 (含概率与区间)...")
    data = load_q1()
    
    # 加载刚跑出来的数据
    main_data = np.load(PROCESSED_DIR / "q1_main_oof_v2.npz", allow_pickle=True)
    zhat = main_data['zhat'] # 5次重复平均后的预测值
    oof_z = main_data['oof_z'] 
    
    rng = np.random.default_rng(SEED)
    N_BOOTSTRAPS = 200
    
    final_scores = {}
    probs_all = {}
    lo_all = {}
    hi_all = {}
    
    for dim_idx, dim in enumerate(data.dims):
        z_true = data.Z[dim].values
        z_pred = zhat[:, dim_idx]
        
        # 1. 构造线性参照 L (全量)
        low, high = np.min(z_true), np.max(z_true)
        z_cont_true = 6 * (z_true - low) / (high - low + 1e-9)
        
        # 2. 预测连续值
        z_cont_pred = 6 * (z_pred - low) / (high - low + 1e-9)
        
        # 3. 计算残差 (用于概率和区间)
        residuals = z_true - z_pred
        
        # 4. Bootstrap 计算概率分布
        boot_scores = []
        for _ in range(N_BOOTSTRAPS):
            z_boot = z_pred + rng.choice(residuals, size=len(z_pred), replace=True)
            # 映射到连续分并截断
            z_cont_boot = 6 * (z_boot - low) / (high - low + 1e-9)
            # 直接四舍五入到整数
            boot_scores.append(np.clip(np.round(z_cont_boot), 0, 6).astype(int))
            
        boot_scores = np.array(boot_scores) # (200, 250)
        
        # 最终得分 = Bootstrap 众数或均值（这里用均值取整，与 csv 一致）
        final_sc = np.round(np.mean(boot_scores, axis=0)).astype(int)
        final_scores[dim] = final_sc
        
        # 概率 p0-p6
        for k in range(7):
            probs_all[f"{dim}_p{k}"] = (boot_scores == k).mean(axis=0)
            
        # 90% 置信区间 (取 5% 和 95% 分位数)
        lo_all[f"{dim}_lo"] = np.percentile(boot_scores, 5, axis=0)
        hi_all[f"{dim}_hi"] = np.percentile(boot_scores, 95, axis=0)

    # 保存最终的 npz（包含得分、概率、区间）
    save_path = PROCESSED_DIR / "q1_scores.npz"
    np.savez_compressed(
        save_path,
        **final_scores,
        **probs_all,
        **lo_all,
        **hi_all,
        seed=SEED
    )
    print(f"✅ 终极交付文件 q1_scores.npz 已生成（含 p0-p6 和 90% 区间）。")

    # 顺手把带区间的长表导出，满足工单要求
    long_rows = []
    for i, beer in enumerate(data.beer):
        for dim_idx, dim in enumerate(data.dims):
            row = {
                "beer": beer,
                "dim": dim,
                "score": final_scores[dim][i],
                "lo": lo_all[f"{dim}_lo"][i],
                "hi": hi_all[f"{dim}_hi"][i],
                "anchored": 0 if dim in NO_ANCHOR_COLS else 1
            }
            # 加上 p0-p6
            for k in range(7):
                row[f"p{k}"] = probs_all[f"{dim}_p{k}"][i]
            long_rows.append(row)
            
    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(PROCESSED_DIR / "q1_scores_long.csv", index=False)
    print(f"✅ 终极长表 q1_scores_long.csv 已生成。")

if __name__ == "__main__":
    main()