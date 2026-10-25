import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, FIG_DIR, SEED
from src.q1.data import load_q1, ChemNAEncoder
from src.q1.splits import outer_splits

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("正在启动极速版 T6 敏感性分析...")
    data = load_q1()
    
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    
    # 获取一套划分即可
    splits = outer_splits(panels, data.style, n_splits=5, n_repeats=1, seed=SEED)
    outer_folds = splits[0]
    
    # 测试 0%, 5%, 10%, 15% 的破坏率，画出优美的折线图
    masks = [0.0, 0.05, 0.10, 0.15]
    mean_spearman = []
    
    # 为了快，我们抽取 10 个核心维度来做这个压力测试
    test_dims = data.dims[:10] 
    
    for frac in masks:
        print(f"  -> 正在模拟 {frac*100:.0f}% 数据缺失...")
        dim_rhos = []
        
        for dim in test_dims:
            y = data.Z[dim].values
            preds = np.zeros(len(y))
            
            for train_idx, test_idx in outer_folds:
                X_train_raw = data.X.iloc[train_idx]
                X_test_raw = data.X.iloc[test_idx]
                y_train = y[train_idx]
                
                # 人为破坏测试集数据
                if frac > 0:
                    rng = np.random.default_rng(SEED)
                    # 只破坏原本有值的单元格
                    valid_mask = X_test_raw.notna()
                    random_mask = rng.random(X_test_raw.shape) < frac
                    mask = valid_mask & random_mask
                    X_test_raw = X_test_raw.copy()
                    X_test_raw[mask] = np.nan
                
                # 编码器（主设定：均值填充）
                encoder = ChemNAEncoder(scheme="main")
                encoder.fit(X_train_raw)
                X_train_proc, ind_train = encoder.transform(X_train_raw)
                X_test_proc, ind_test = encoder.transform(X_test_raw)
                
                X_train_final = pd.concat([X_train_proc, ind_train], axis=1)
                X_test_final = pd.concat([X_test_proc, ind_test], axis=1)
                # 防止列不匹配
                X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)
                
                # 极速模型，不调参
                model = HistGradientBoostingRegressor(max_depth=3, max_iter=100, random_state=SEED)
                model.fit(X_train_final, y_train)
                preds[test_idx] = model.predict(X_test_final)
                
            rho = spearmanr(y, preds)[0]
            if not np.isnan(rho):
                dim_rhos.append(rho)
                
        mean_rho = np.mean(dim_rhos)
        mean_spearman.append(mean_rho)
        print(f"     {frac*100:.0f}% 缺失下的平均 Spearman: {mean_rho:.4f}")

    # 画图
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot([f"{int(m*100)}%" for m in masks], mean_spearman, marker='o', linestyle='-', color='#d62728', linewidth=2.5)
    
    # 标注具体的数字
    for i, v in enumerate(mean_spearman):
        ax.text(i, v + 0.002, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        
    ax.set_title("敏感性分析：随机缺失对模型性能的影响", fontsize=14)
    ax.set_xlabel("化学数据缺失比例 (MCAR)", fontsize=12)
    ax.set_ylabel("平均 Spearman 相关系数", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(min(mean_spearman)-0.05, max(mean_spearman)+0.05)
    
    plt.tight_layout()
    save_path = FIG_DIR / "fig_q1_sensitivity.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 敏感性图表已生成: {save_path}")
    print("请前往 figures 文件夹查看！")

if __name__ == "__main__":
    main()