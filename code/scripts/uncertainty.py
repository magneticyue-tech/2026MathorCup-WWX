import sys
import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.convert import NO_ANCHOR_COLS

def main():
    print("正在生成最终交付产物 (概率区间对齐切点及强断言校验)...")
    data = load_q1()
    
    main_data = np.load(PROCESSED_DIR / "q1_main_oof_v2.npz", allow_pickle=True)
    zhat = main_data['zhat']
    inner_z_all = main_data['inner_z']
    
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
        
        inner_z_dim = inner_z_all[:, :, dim_idx] 
        valid_mask = ~np.isnan(inner_z_dim)
        z_true_expanded = np.tile(z_true, (5, 1))
        inner_residuals = z_true_expanded[valid_mask] - inner_z_dim[valid_mask]
        
        low = selection_params[dim]["low"]
        high = selection_params[dim]["high"]
        thresholds = np.array(selection_params[dim]["thresholds"])
        
        z_cont_pred = 6 * (z_pred - low) / (high - low + 1e-9)
        
        boot_scores = []
        for _ in range(N_BOOTSTRAPS):
            z_boot = z_pred + rng.choice(inner_residuals, size=len(z_pred), replace=True)
            z_cont_boot = 6 * (z_boot - low) / (high - low + 1e-9)
            boot_scores.append(np.digitize(z_cont_boot, thresholds))
            
        boot_scores = np.array(boot_scores) 
        
        final_scores[dim] = np.digitize(z_cont_pred, thresholds)
        
        for k in range(7):
            probs_all[f"{dim}_p{k}"] = (boot_scores == k).mean(axis=0)
            
        lo_all[f"{dim}_lo"] = np.percentile(boot_scores, 5, axis=0)
        hi_all[f"{dim}_hi"] = np.percentile(boot_scores, 95, axis=0)

    # 断言校验
    for dim in data.dims:
        p_sum = sum(probs_all[f"{dim}_p{k}"] for k in range(7))
        assert np.allclose(p_sum, 1.0), f"{dim} 概率和不为1！"
        assert np.all(lo_all[f"{dim}_lo"] <= final_scores[dim]), f"{dim} 区间下限错误！"
        assert np.all(final_scores[dim] <= hi_all[f"{dim}_hi"]), f"{dim} 区间上限错误！"
    
    print("✅ 断言校验通过：概率(p0-p6)行和为1，且 lo <= score <= hi 严格成立。")

    save_path = PROCESSED_DIR / "q1_scores.npz"
    np.savez_compressed(save_path, **final_scores, **probs_all, **lo_all, **hi_all, seed=SEED)
    print(f"✅ 终极交付文件 q1_scores.npz 已生成。")

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