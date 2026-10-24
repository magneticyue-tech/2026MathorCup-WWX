import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1, ChemNAEncoder
from src.q1.splits import outer_splits

def mask_cells(X_train, X_test, frac, seed):
    """人为地在测试集里制造缺失值 (MCAR)"""
    rng = np.random.default_rng(seed)
    X_test_masked = X_test.copy()
    
    # 找到原本有数据的单元格
    valid_mask = X_test_masked.notna()
    
    # 随机生成遮罩
    random_mask = rng.random(X_test_masked.shape) < frac
    mask = valid_mask & random_mask
    
    X_test_masked[mask] = np.nan
    return X_test_masked, mask.sum()

def main():
    print("正在加载数据，启动 T6 压力测试...")
    data = load_q1()
    
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    oof_data = np.load(PROCESSED_DIR / "q1_main_oof_demo.npz", allow_pickle=True)
    
    splits = outer_splits(panels, data.style, n_splits=5, n_repeats=1, seed=SEED)
    outer_folds = splits[0]
    
    # 挑 3 个代表性维度速测
    test_dims = ["sweetness", "bitternes", "body"]
    
    # 实验设置：0% (基准), 5%, 10%
    masks = [0.0, 0.05, 0.10]
    
    results = {m: [] for m in masks}
    
    for dim in test_dims:
        print(f"\n--- 测试维度: {dim} ---")
        y = data.Z[dim].values
        
        for frac in masks:
            preds = np.zeros(len(y))
            
            for train_idx, test_idx in outer_folds:
                # 1. 切分原始数据
                X_train_raw = data.X.iloc[train_idx]
                X_test_raw = data.X.iloc[test_idx]
                y_train = y[train_idx]
                
                # 2. 如果是压力测试，对测试集搞破坏
                if frac > 0:
                    X_test_raw, _ = mask_cells(X_train_raw, X_test_raw, frac, seed=SEED)
                
                # 3. 编码缺失值 (使用主设定: 填0+指示器)
                encoder = ChemNAEncoder(scheme="main")
                encoder.fit(X_train_raw)
                X_train_proc, ind_train = encoder.transform(X_train_raw)
                X_test_proc, ind_test = encoder.transform(X_test_raw)
                
                X_train_final = pd.concat([X_train_proc, ind_train], axis=1)
                X_test_final = pd.concat([X_test_proc, ind_test], axis=1)
                
                # 🚨 加上这一行：把测试集的列强行对齐到训练集！
                X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)                
                # 4. 快速训练模型 (不再调参，直接用固定参数)
                model = HistGradientBoostingRegressor(
                    max_depth=3, min_samples_leaf=20, learning_rate=0.05, random_state=SEED
                )
                model.fit(X_train_final, y_train)
                preds[test_idx] = model.predict(X_test_final)
            
            # 计算 Spearman
            rho = spearmanr(y, preds)[0]
            results[frac].append(rho)
            print(f"  缺失率 {frac*100:>2.0f}% | Spearman: {rho:.4f}")
            
    # 计算平均跌幅
    print("\n" + "="*40)
    print("📊 T6 敏感性分析报告 (破坏测试)")
    print("="*40)
    base_mean = np.mean(results[0.0])
    for frac in masks[1:]:
        current_mean = np.mean(results[frac])
        drop = base_mean - current_mean
        print(f"缺失率 {frac*100:>2.0f}%: 平均 Spearman 下降 {drop:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()