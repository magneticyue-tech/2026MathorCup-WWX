import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.convert import NO_ANCHOR_COLS

def main():
    print("正在生成最终交付产物 (修复版：概率和区间完全对齐获选切点映射)...")
    data = load_q1()
    
    main_data = np.load(PROCESSED_DIR / "q1_main_oof_v2.npz", allow_pickle=True)
    zhat = main_data['zhat']
    inner_z_all = main_data['inner_z']
    
    # 🚨 核心修复：读取刚刚 solve_q1_scores.py 跑出来的获选切点
    with open(PROCESSED_DIR / "q1_mapping_selection.json", "r") as f:
        selection_data = json.load(f)
    selection_params = selection_data["params"]
    
    rng = np.random.default_rng(SEED)
    N_BOOTSTRAPS = 200
    
    final_scores = {}
    probs_all = {}
    lo_all = {}
    hi_all = {}
    
    for dim_idx, dim in enumerate(data.dims):
        z_true = data.Z[dim].values
        z_pred = zhat[:, dim_idx]
        
        # 获取该维度的残差池
        inner_z_dim = inner_z_all[:, :, dim_idx] 
        valid_mask = ~np.isnan(inner_z_dim)
        z_true_expanded = np.tile(z_true, (5, 1))
        inner_residuals = z_true_expanded[valid_mask] - inner_z_dim[valid_mask]
        
        # 获取该维度的切点参数
        low = selection_params[dim]["low"]
        high = selection_params[dim]["high"]
        thresholds = np.array(selection_params[dim]["thresholds"])
        
        z_cont_pred = 6 * (z_pred - low) / (high - low + 1e-9)
        
        # Bootstrap 采样
        boot_scores = []
        for _ in range(N_BOOTSTRAPS):
            z_boot = z_pred + rng.choice(inner_residuals, size=len(z_pred), replace=True)
            z_cont_boot = 6 * (z_boot - low) / (high - low + 1e-9)
            # 🚨 核心修复：使用 digitize 和训练好的切点，而不是 np.round
            boot_scores.append(np.digitize(z_cont_boot, thresholds))
            
        boot_scores = np.array(boot_scores) # (200, 250)
        
        # 最终得分必须和 csv 里的交付得分一致 (即按切点算出来的真实得分)
        final_scores[dim] = np.digitize(z_cont_pred, thresholds)
        
        for k in range(7):
            probs_all[f"{dim}_p{k}"] = (boot_scores == k).mean(axis=0)
            
        lo_all[f"{dim}_lo"] = np.percentile(boot_scores, 5, axis=0)
        hi_all[f"{dim}_hi"] = np.percentile(boot_scores, 95, axis=0)

    # 保存最终的 npz
    save_path = PROCESSED_DIR / "q1_scores.npz"
    np.savez_compressed(save_path, **final_scores, **probs_all, **lo_all, **hi_all, seed=SEED)
    print(f"✅ 终极交付文件 q1_scores.npz 已生成（概率区间对齐切点）。")

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
    print(f"✅ 终极长表 q1_scores_long.csv 已重新生成（完全对齐切点）。")

if __name__ == "__main__":
    main()