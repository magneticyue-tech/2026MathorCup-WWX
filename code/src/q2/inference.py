# code/src/q2/inference.py
"""Q2 推断层：置换检验，计算显著性门槛 q_d。"""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd

import config as C
from data import load_q2_dataset


def run_permutation_test():
    print("="*50)
    print("开始计算置换零分布门槛（q_d）")
    print("="*50)

    # 1. 加载数据
    ds = load_q2_dataset()
    X_mat = ds.X_rank.values  # 250 x 225
    Y_mat = ds.Y_rank.values  # 250 x 50
    dim_cols = ds.Y_rank.columns
    n = len(ds.sample_ids)

    # 设置随机种子以保证可复现
    np.random.seed(C.SEED)
    
    # 运行次数：正式 1000，这里为了快速看效果设为 1000，如果嫌慢可以改成 200
    N_PERM = 1000 
    
    q_d_dict = {}

    print(f"正在进行 {N_PERM} 次置换检验，请稍候（可能需要十几秒）...")

    # 2. 对每个维度计算门槛 q_d
    for d_idx, d_name in enumerate(dim_cols):
        y = Y_mat[:, d_idx]
        
        # 生成 N_PERM 个打乱的 y
        y_perms = np.array([np.random.permutation(y) for _ in range(N_PERM)]).T  # 250 x 1000
        
        # 计算所有候选化合物与所有打乱 y 的相关系数 (225 x 250 乘以 250 x 1000)
        # 注意：X 已经标准化，Y 也是秩，所以点积除以 n 就是相关系数
        r_perms = (X_mat.T @ y_perms) / n  # 结果形状: 225 x 1000
        
        # 找出每次置换中（即每一列中）绝对值最大的 r
        max_abs_r_perms = np.max(np.abs(r_perms), axis=0)  # 形状: 1000
        
        # 取 95% 分位数
        q_d = np.percentile(max_abs_r_perms, 95)
        q_d_dict[d_name] = q_d

    # 3. 将 q_d 合并到 Top-3 结果中
    print("\n正在将门槛合并至结果表...")
    results_file = C.Q2_DIR / "top3_results.csv"
    if not results_file.exists():
        raise FileNotFoundError("找不到 top3_results.csv，请先运行 association.py")
        
    df_top3 = pd.read_csv(results_file)
    
    # 添加 q_d 和 是否显著 的列
    df_top3["q_d"] = df_top3["dimension"].map(q_d_dict)
    df_top3["is_significant"] = df_top3["score"].abs() > df_top3["q_d"]
    
    # 保存最终结果
    out_file = C.Q2_DIR / "top3_results_with_sig.csv"
    df_top3.to_csv(out_file, index=False)
    
    print("\n✅ 显著性检验完成！")
    print(f"结果已保存至: {out_file}")
    
    # 打印预览
    print("\n【预览】加入显著性后的结果：")
    for dim in ["A_malt_all", "lactic", "acetic", "t2n"]:
        if dim in dim_cols:
            print(f"\n--- {dim} ---")
            sub = df_top3[df_top3["dimension"] == dim]
            qd = sub["q_d"].iloc[0]
            print(f"门槛 q_d = {qd:.3f}")
            for _, row in sub.iterrows():
                sig_mark = "✅显著" if row["is_significant"] else "❌不显著"
                print(f"Top {row['rank']}: {row['compound']:<25} r={row['score']:>6.3f}  {sig_mark}")

if __name__ == "__main__":
    run_permutation_test()