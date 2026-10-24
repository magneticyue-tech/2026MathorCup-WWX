import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, FIG_DIR
from src.q1.data import load_q1
from src.q1.baselines import b0_mean, b1_style_shrunk
from src.q1.splits import outer_splits
from src.q1.paths import SEED
from scipy.stats import spearmanr

# 设置中文字体，防止图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("正在生成 T7 图表...")
    data = load_q1()
    
    # 提取数据
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    
    oof_data = np.load(PROCESSED_DIR / "q1_main_oof_demo.npz", allow_pickle=True)
    
    splits = outer_splits(panels, data.style, n_splits=5, n_repeats=1, seed=SEED)
    outer_folds = splits[0]
    
    rho_main_list = []
    rho_b1_list = []
    
    for dim in data.dims:
        y_true = data.Z[dim].values
        z_main = oof_data[dim]
        z_b1 = np.zeros(len(y_true))
        
        for train_idx, test_idx in outer_folds:
            Z_dim_df = data.Z[[dim]]
            z_b1[test_idx] = b1_style_shrunk(train_idx, test_idx, Z_dim_df, data.style).flatten()
            
        rho_main_list.append(spearmanr(y_true, z_main)[0])
        rho_b1_list.append(spearmanr(y_true, z_b1)[0])

    # --- 开始画图 ---
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    x = np.arange(len(data.dims))
    width = 0.35
    
    # 画柱子
    bars1 = ax.bar(x - width/2, rho_main_list, width, label='Main Model (Chem)', color='#1f77b4')
    bars2 = ax.bar(x + width/2, rho_b1_list, width, label='B1 Baseline (Style)', color='#ff7f0e', alpha=0.7)
    
    # 设置图表元素
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Main Model vs B1 Baseline across 50 Sensory Dimensions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data.dims, rotation=90, fontsize=8)
    
    # 添加平均线
    mean_main = np.nanmean(rho_main_list)
    mean_b1 = np.nanmean(rho_b1_list)
    ax.axhline(mean_main, color='#1f77b4', linestyle='--', linewidth=1.5, label=f'Main Mean ({mean_main:.3f})')
    ax.axhline(mean_b1, color='#ff7f0e', linestyle='--', linewidth=1.5, label=f'B1 Mean ({mean_b1:.3f})')
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # 保存
    save_path = FIG_DIR / "q1_main_vs_b1.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 绝美图表已保存至: {save_path}")

if __name__ == "__main__":
    main()