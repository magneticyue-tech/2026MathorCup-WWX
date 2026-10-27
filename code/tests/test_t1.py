import sys
from pathlib import Path

# 🚨 动态把 code 目录加入环境变量
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.q1.data import load_q1
import numpy as np
import pandas as pd

print("正在加载 T1 数据...")
data = load_q1()

print("\n================ T1 验收报告 ================")
print(f"目标 xbar 形状: {data.targets.xbar.shape} (应为 250x50)")
print(f"目标 G 形状: {data.targets.G.shape} (应为 250x50)")
print(f"每款酒的评分人数 (前5款): {data.targets.n_ratings[:5]}")

# 工单要求的死亡核对：G 的等级计数
flat_G = data.targets.G.flatten()
counts = np.bincount(flat_G, minlength=7)
print(f"\nG 的等级计数 (核心校验): {counts.tolist()}")
print("工单期望值: [4674, 3579, 2057, 1545, 524, 112, 9]")
print("==============================================")