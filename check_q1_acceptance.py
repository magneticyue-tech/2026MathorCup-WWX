#!/usr/bin/env python3
"""Q1 验收检查脚本"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

PROC = Path("code/data/processed")

print("=" * 60)
print("Q1 验收检查")
print("=" * 60)

# T1: panels
print("\n[T1] 品评团推断")
p = np.load(PROC / "q1_panels.npz", allow_pickle=True)
panels = p['panels']
counts = pd.Series(panels).value_counts()
multi = counts[counts > 1]
print(f"  多啤酒组: {len(multi)} 个 (期望41)")
print(f"  覆盖啤酒: {multi.sum()} 款 (期望228)")
print(f"  最大组: {multi.max()} 款 (期望11)")
print(f"  ✓ T1a PASS" if len(multi)==41 and multi.sum()==228 and multi.max()==11 else "  ✗ T1a FAIL")

zeros = p['zeros']
print(f"  zeros shape: {zeros.shape} (期望 (41, 50))")
print(f"  ✓ T1b PASS" if zeros.shape == (41, 50) else "  ✗ T1b FAIL")

# T2: main OOF
print("\n[T2] 主模型 OOF")
oof = np.load(PROC / "q1_main_oof_demo.npz", allow_pickle=True)
dim_keys = [k for k in oof.keys() if k not in ['seed', 'meta']]
print(f"  预测维度数: {len(dim_keys)} (期望50)")
print(f"  seed: {oof['seed']} (期望0)")
print(f"  ✓ T2 PASS" if len(dim_keys) == 50 and oof['seed'] == 0 else "  ✗ T2 FAIL")

# T3: scores
print("\n[T3] z → 0–6 评分")
scores = np.load(PROC / "q1_scores_full.npz", allow_pickle=True)
dims_in_scores = [k for k in scores.keys() if not k.startswith('lo_') and not k.startswith('hi_') and '_p' not in k and k != 'seed']
print(f"  评分维度数: {len(dims_in_scores)} (期望50)")

df = pd.read_csv(PROC / "q1_final_scores.csv")
print(f"  CSV shape: {df.shape} (期望 (250, 51))")
print(f"  ✓ T3 PASS" if len(dims_in_scores) == 50 and df.shape == (250, 51) else "  ✗ T3 FAIL")

# T5: metrics
print("\n[T5] 对照与指标")
with open(PROC / "q1_metrics.json") as f:
    met = json.load(f)
main_sp = met['main_spearman']
b1_sp = met['b1_spearman']
wins = met['wins_over_b1']
print(f"  main_spearman: {main_sp:.4f}")
print(f"  b1_spearman: {b1_sp:.4f}")
print(f"  wins_over_b1: {wins}")
success = main_sp > b1_sp and wins >= 25
print(f"  成功判定: {success}")
print(f"  ✓ T5 PASS" if success else "  ✗ T5 FAIL")

print("\n" + "=" * 60)
print("快速验收检查完成")
print("=" * 60)
