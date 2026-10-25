import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.panels import panel_zeros

def optimize_qwk_thresholds(z_cont, R_true):
    """优化切点：使用分位数初始化，然后迭代微调"""
    # 初始切点
    best_thresholds = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    best_qwk = -np.inf
    
    # 简单的坐标下降法微调
    for iteration in range(3): # 跑3轮
        for i in range(6):
            # 尝试在这个切点左右移动，找最高分
            for step in np.linspace(-0.2, 0.2, 9):
                test_thresholds = best_thresholds.copy()
                test_thresholds[i] += step
                
                # 确保切点是递增的
                if i > 0 and test_thresholds[i] <= test_thresholds[i-1]: continue
                if i < 5 and test_thresholds[i] >= test_thresholds[i+1]: continue
                
                pred = np.digitize(z_cont, test_thresholds)
                qwk = cohen_kappa_score(R_true, pred, weights="quadratic", labels=range(7))
                
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_thresholds = test_thresholds
    return best_thresholds

def main():
    print("正在执行 T3/T4: 混合参照与 QWK 切点选型 (修复版)...")
    data = load_q1()
    
    main_data = np.load(PROCESSED_DIR / "q1_main_oof_v2.npz", allow_pickle=True)
    zhat = main_data['zhat'] # (250, 50) 五次平均预测
    panels_data = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    panels = panels_data['panels']
    
    qwk_linear_list = []
    qwk_qwk_list = []
    final_scores_dict = {}
    
    for dim_idx, dim in enumerate(data.dims):
        z_true = data.Z[dim].values
        
        # =====================================
        # 1. 构造 L (线性参照) 和 R (混合参照)
        # =====================================
        low, high = np.min(z_true), np.max(z_true)
        z_cont_L = 6 * (z_true - low) / (high - low + 1e-9)
        L = np.round(z_cont_L).astype(int)
        
        zeros_df = panel_zeros(data.Z.iloc[:], panels)
        if zeros_df.shape[1] > dim_idx:
            zeros = zeros_df.iloc[:, dim_idx].dropna().values
            if len(zeros) > 0 and dim != "F_malt_all":
                z0_tilde = np.median(zeros)
                r = -z0_tilde
                sigma = 3 * r / (1 + r**2)
                A = np.round(np.clip(sigma * (z_true - z0_tilde), 0, 6)).astype(int)
                I = np.zeros(len(z_true), dtype=bool)
                for g in zeros_df.index:
                    if not np.isnan(zeros_df.loc[g, dim]):
                        idx = np.where(panels == int(g))[0]
                        I[idx] = True
                R = np.where(I, A, L)
            else:
                R = L
        else:
            R = L
            
        # =====================================
        # 2. 比较映射方法（预测值必须用 zhat）
        # =====================================
        # 方法一：直接线性映射 (用 zhat 算)
        z_cont_pred = 6 * (zhat[:, dim_idx] - low) / (high - low + 1e-9)
        pred_lin = np.clip(np.round(z_cont_pred), 0, 6).astype(int)
        qwk_lin = cohen_kappa_score(R, pred_lin, weights="quadratic", labels=range(7))
        
        # 方法二：QWK优化切点 (用 zhat 算)
        best_thresholds = optimize_qwk_thresholds(z_cont_pred, R)
        pred_qwk = np.digitize(z_cont_pred, best_thresholds)
        qwk_qwk = cohen_kappa_score(R, pred_qwk, weights="quadratic", labels=range(7))
        
        qwk_linear_list.append(qwk_lin)
        qwk_qwk_list.append(qwk_qwk)
        
        # 选型：用 QWK 高的作为最终输出
        final_scores_dict[dim] = pred_qwk if qwk_qwk > qwk_lin else pred_lin

    print(f"\n📊 映射选型结果：")
    print(f"直接线性映射平均 QWK: {np.nanmean(qwk_linear_list):.4f}")
    print(f"QWK切点优化平均 QWK: {np.nanmean(qwk_qwk_list):.4f}")
    
    save_path = PROCESSED_DIR / "q1_scores_v3.npz"
    np.savez_compressed(save_path, **final_scores_dict, seed=SEED)
    
    df = pd.DataFrame(final_scores_dict)
    df.insert(0, "beer", data.beer)
    df.to_csv(PROCESSED_DIR / "q1_final_scores_v3.csv", index=False)
    print(f"✅ 最终评分已保存至: {save_path}")

if __name__ == "__main__":
    main()