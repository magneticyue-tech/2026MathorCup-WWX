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
    print("正在加载数据并计算基线指标 (T5)...")
    data = load_q1()
    
    # 加载 T1 的场次划分
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    
    # 加载 T2 主模型预测的 OOF 结果
    oof_data = np.load(PROCESSED_DIR / "q1_main_oof_demo.npz", allow_pickle=True)
    
    # 获取外层折（使用1次重复）
    splits = outer_splits(panels, data.style, n_splits=5, n_repeats=1, seed=SEED)
    outer_folds = splits[0]
    
    # 存储指标
    results = {"main": [], "B0": [], "B1": []}
    
    # 我们直接对 50 维计算
    for dim in data.dims:
        y_true = data.Z[dim].values
        
        # 主模型预测值
        z_main = oof_data[dim]
        
        # 计算基线的折外预测
        z_b0 = np.zeros(len(y_true))
        z_b1 = np.zeros(len(y_true))
        
        for train_idx, test_idx in outer_folds:
            # 我们需要取单维的 Z 算基线
            Z_dim_df = data.Z[[dim]]
            z_b0[test_idx] = b0_mean(train_idx, test_idx, Z_dim_df).flatten()
            z_b1[test_idx] = b1_style_shrunk(train_idx, test_idx, Z_dim_df, data.style).flatten()
            
        # 计算 Spearman
        rho_main = spearmanr(y_true, z_main)[0]
        rho_b0 = spearmanr(y_true, z_b0)[0]
        rho_b1 = spearmanr(y_true, z_b1)[0]
        
        results["main"].append(rho_main)
        results["B0"].append(rho_b0)
        results["B1"].append(rho_b1)
        
    # 结果汇总
    avg_main = np.nanmean(results["main"])
    avg_b1 = np.nanmean(results["B1"])
    
    # 计算赢过 B1 的维度数
    wins = np.sum(np.array(results["main"]) > np.array(results["B1"]))
    
    print("\n" + "="*40)
    print("📊 T5 基线对比报告")
    print("="*40)
    print(f"主模型平均 Spearman: {avg_main:.4f}")
    print(f"B1(风格基线)平均 Spearman: {avg_b1:.4f}")
    print(f"主模型打败 B1 的维度数: {wins} / {len(data.dims)}")
    
    if avg_main > avg_b1 and wins >= len(data.dims)//2:
        print("🏆 结论: 化学模型显著优于风格基线，成功标准达成！")
    else:
        print("⚠️ 结论: 未达到事先声明的成功标准，需要向队长报告。")
    print("="*40)
    
    # 保存指标到 json
    with open(PROCESSED_DIR / "q1_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "main_spearman": avg_main,
            "b1_spearman": avg_b1,
            "wins_over_b1": int(wins)
        }, f, indent=4)
    
    print(f"指标已保存至: {PROCESSED_DIR / 'q1_metrics.json'}")

if __name__ == "__main__":
    main()