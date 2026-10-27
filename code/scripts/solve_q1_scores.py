import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.discretize import apply_rules, dist_median, dist_interval

def main():
    print("正在加载 T2 预测结果与 T1 目标...")
    data = load_q1()
    
    q1_data = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    G_obs = q1_data['G'] # 250x50 真实等级
    
    main_data = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=True)
    oof_mu = main_data['oof_mu'] # (5, 250, 50)
    inner_mu = main_data['inner_mu'] # (5, 5, 250, 50)
    
    print("正在执行 T3 离散化与交付生成...")
    
    # 存储最终结果的容器
    p_final = np.zeros((250, 50, 7))
    G_hat = np.zeros((250, 50), dtype=int)
    lo_final = np.zeros((250, 50), dtype=int)
    hi_final = np.zeros((250, 50), dtype=int)
    G_hat_Q = np.zeros((250, 50), dtype=int) # 分位数规则的中位数
    
    # 为了后续对比，把各种规则的结果都存下来
    rules_dict = {r: np.zeros((5, 250, 50), dtype=int) for r in ["const", "round", "median", "quant", "qwk_cut"]}
    
    for dim_idx, dim in enumerate(data.dims):
        # 提取该维度的所有数据
        mu_run = oof_mu[:, :, dim_idx] # (5, 250)
        inner_mu_run = inner_mu[:, :, :, dim_idx] # (5, 5, 250)
        g_inner_run = G_obs[:, dim_idx] # (250,)
        y_inner_run = q1_data['xbar'][:, dim_idx] # 连续均值
        
        p_runs = []
        quant_medians = []
        
        for run in range(5):
            # 把5折的内层预测拼起来（去掉 NaN，只保留有效的内层预测作为残差池）
            inner_preds = inner_mu_run[run] # (5, 250)
            valid_mask = ~np.isnan(inner_preds)
            
            # 提取有效的连续预测、真实均值和真实等级
            mu_inner = inner_preds[valid_mask]
            y_inner = np.tile(y_inner_run, (5, 1))[valid_mask]
            g_inner = np.tile(g_inner_run, (5, 1))[valid_mask]
            
            # 调用离散化函数
            res = apply_rules(mu_run[run], mu_inner, y_inner, g_inner)
            
            p_runs.append(res['p'])
            for rule in ["const", "round", "median", "quant", "qwk_cut"]:
                rules_dict[rule][run, :, dim_idx] = res[rule]
                
            quant_medians.append(res['quant'])
            
        # 按照工单 §3.5 合并 5 次重复
        p_final[:, dim_idx] = np.mean(p_runs, axis=0) # p_final 为 5 次 p 的平均
        G_hat[:, dim_idx] = dist_median(p_final[:, dim_idx]) # 由 p_final 取中位数
        lo_final[:, dim_idx], hi_final[:, dim_idx] = dist_interval(p_final[:, dim_idx]) # 由 p_final 算区间
        
        # G_hat_Q 为 5 次 quant 规则结果的中位数
        G_hat_Q[:, dim_idx] = np.median(np.array(quant_medians), axis=0).astype(int)

    print("正在保存 q1_scores.npz...")
    save_path = PROCESSED_DIR / "q1_scores.npz"
    np.savez_compressed(
        save_path,
        p_final=p_final,
        G_hat=G_hat,
        lo=lo_final,
        hi=hi_final,
        G_hat_Q=G_hat_Q,
        **{f"grade_{r}": rules_dict[r] for r in rules_dict}
    )
    print(f"✅ T3 运行完毕，最终评分已存入: {save_path}")

if __name__ == "__main__":
    main()