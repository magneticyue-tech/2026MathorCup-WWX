import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, FIG_DIR, SEED
from src.q1.data import load_q1
from src.q1.splits import outer_splits
from src.q1.baselines import b1_style_shrunk

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("正在生成 T7 补充图表...")
    data = load_q1()
    
    # 1. 画品评团大小分布图 (fig_q1_panels.png)
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    
    counts = pd.Series(panels).value_counts().values
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    # 画直方图
    ax.hist(counts, bins=np.arange(0.5, max(counts)+1.5, 1), edgecolor='black', color='#2ca02c', alpha=0.7)
    ax.set_title("品评团大小分布 (Panel Size Distribution)", fontsize=14)
    ax.set_xlabel("品评团包含的啤酒数量", fontsize=12)
    ax.set_ylabel("品评团个数", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 标出 1 款的单例组
    single_count = sum(counts == 1)
    ax.text(1, ax.get_ylim()[1]*0.9, f"单款组: {single_count} 个", color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_q1_panels.png", bbox_inches='tight')
    plt.close()
    print(f"✅ 品评团分布图已保存: {FIG_DIR / 'fig_q1_panels.png'}")
    
    # 2. 画 Spearman 对比图 (替换之前那个柱状图，这个更像工单要求的按维度组排列)
    # 为了测试，我们依然用 demo 数据
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
        
    fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
    x = np.arange(len(data.dims))
    width = 0.35
    
    ax.bar(x - width/2, rho_main_list, width, label='Main Model', color='#1f77b4')
    ax.bar(x + width/2, rho_b1_list, width, label='B1 Baseline', color='#ff7f0e', alpha=0.7)
    
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('逐维 Spearman 对比 (Main vs B1)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data.dims, rotation=90, fontsize=8)
    ax.axhline(np.nanmean(rho_main_list), color='#1f77b4', linestyle='--', label='Main Mean')
    ax.axhline(np.nanmean(rho_b1_list), color='#ff7f0e', linestyle='--', label='B1 Mean')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_q1_spearman.png", bbox_inches='tight')
    plt.close()
    print(f"✅ Spearman对比图已保存: {FIG_DIR / 'fig_q1_spearman.png'}")

if __name__ == "__main__":
    main()