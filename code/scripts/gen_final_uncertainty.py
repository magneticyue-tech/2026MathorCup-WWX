import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.convert import NO_ANCHOR_COLS

def main():
    print("正在生成最终交付产物 (修复版：使用纯内层折外残差防泄漏)...")
    data = load_q1()
    
    # 读取主模型产物
    main_data = np.load(PROCESSED_DIR / "q1_main_oof_v2.npz", allow_pickle=True)
    zhat = main_data['zhat'] # 5次重复平均后的预测值 (250, 50)
    inner_z_all = main_data['inner_z'] # 内层交叉验证预测值 (5, 250, 50)
    
    rng = np.random.default_rng(SEED)
    N_BOOTSTRAPS = 200
    
    final_scores = {}
    probs_all = {}
    lo_all = {}
    hi_all = {}
    
    for dim_idx, dim in enumerate(data.dims):
        z_true = data.Z[dim].values
        z_pred = zhat[:, dim_idx]
        
        # ==========================================
        # 核心修复点：提取该维度的“内层折外残差池”
        # ==========================================
        # 取该维度5次重复的所有内层预测 (5, 250)
        inner_z_dim = inner_z_all[:, :, dim_idx] 
        
        # 找到不是 NaN 的位置（因为内层预测只针对外层训练集，测试集是 NaN）
        valid_mask = ~np.isnan(inner_z_dim)
        
        # 真实值扩展成 (5, 250) 去对齐，提取残差
        z_true_expanded = np.tile(z_true, (5, 1))
        inner_residuals = z_true_expanded[valid_mask] - inner_z_dim[valid_mask]
        
        # 构造线性参照
        low, high = np.min(z_true), np.max(z_true)
        z_cont_pred = 6 * (z_pred - low) / (high - low + 1e-9)
        
        # ==========================================
        # Bootstrap 采样 (从内层残差池中抽)
        # ==========================================
        boot_scores = []
        for _ in range(N_BOOTSTRAPS):
            # 这里抽样用的是纯粹的、无泄漏的内层残差
            z_boot = z_pred + rng.choice(inner_residuals, size=len(z_pred), replace=True)
            z_cont_boot = 6 * (z_boot - low) / (high - low + 1e-9)
            boot_scores.append(np.clip(np.round(z_cont_boot), 0, 6).astype(int))
            
        boot_scores = np.array(boot_scores) # (200, 250)
        
        final_sc = np.round(np.mean(boot_scores, axis=0)).astype(int)
        final_scores[dim] = final_sc
        
        for k in range(7):
            probs_all[f"{dim}_p{k}"] = (boot_scores == k).mean(axis=0)
            
        lo_all[f"{dim}_lo"] = np.percentile(boot_scores, 5, axis=0)
        hi_all[f"{dim}_hi"] = np.percentile(boot_scores, 95, axis=0)

    # 保存最终的 npz
    save_path = PROCESSED_DIR / "q1_scores.npz"
    np.savez_compressed(
        save_path,
        **final_scores,
        **probs_all,
        **lo_all,
        **hi_all,
        seed=SEED
    )
    print(f"✅ 终极防泄漏交付文件 q1_scores.npz 已生成。")

    # 导出长表
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
            for k in range(7):
                row[f"p{k}"] = probs_all[f"{dim}_p{k}"][i]
            long_rows.append(row)
            
    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(PROCESSED_DIR / "q1_scores_long.csv", index=False)
    print(f"✅ 终极长表 q1_scores_long.csv 已重新生成。")

if __name__ == "__main__":
    main()