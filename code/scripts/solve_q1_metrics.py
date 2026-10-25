import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.splits import outer_splits
from src.q1.baselines import b0_mean, b1_style_shrunk

def main():
    print("正在加载数据并计算基线指标 (T5 修复版)...")
    data = load_q1()
    
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    
    oof_data = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=True)
    
    # 获取 5 次重复的外层折
    splits = outer_splits(panels, data.style, n_splits=5, n_repeats=5, seed=SEED)
    
    results = {"main": [], "B0": [], "B1": []}
    
    for dim in data.dims:
        y_true = data.Z[dim].values
        z_main = oof_data[dim]
        
        # 对 B0 和 B1 也进行 5 次重复预测并求平均
        z_b0_runs = []
        z_b1_runs = []
        
        for run in range(5):
            outer_folds = splits[run]
            z_b0 = np.zeros(len(y_true))
            z_b1 = np.zeros(len(y_true))
            
            for train_idx, test_idx in outer_folds:
                Z_dim_df = data.Z[[dim]]
                z_b0[test_idx] = b0_mean(train_idx, test_idx, Z_dim_df).flatten()
                z_b1[test_idx] = b1_style_shrunk(train_idx, test_idx, Z_dim_df, data.style).flatten()
                
            z_b0_runs.append(z_b0)
            z_b1_runs.append(z_b1)
            
        z_b0_avg = np.mean(z_b0_runs, axis=0)
        z_b1_avg = np.mean(z_b1_runs, axis=0)
        
        rho_main = spearmanr(y_true, z_main)[0]
        rho_b0 = spearmanr(y_true, z_b0_avg)[0]
        rho_b1 = spearmanr(y_true, z_b1_avg)[0]
        
        results["main"].append(rho_main)
        results["B0"].append(rho_b0)
        results["B1"].append(rho_b1)
        
    avg_main = np.nanmean(results["main"])
    avg_b1 = np.nanmean(results["B1"])
    wins = np.sum(np.array(results["main"]) > np.array(results["B1"]))
    
    print("\n" + "="*40)
    print("📊 T5 基线对比报告 (5次重复对齐版)")
    print("="*40)
    print(f"主模型平均 Spearman: {avg_main:.4f}")
    print(f"B1(风格基线)平均 Spearman: {avg_b1:.4f}")
    print(f"主模型打败 B1 的维度数: {wins} / {len(data.dims)}")
    print("="*40)

if __name__ == "__main__":
    main()