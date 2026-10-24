"""
Q1 核心训练流水线 (V2.1 适配本地目录版)
包含：数据预处理、品评团推断、零点计算、HGBR模型训练、单调约束、5次重复CV
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 路径与数据准备
# ==========================================
# 自动获取当前工作目录（确保你在 MathorCupMX 根目录运行）
ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results" / "q1"
RESULTS.mkdir(parents=True, exist_ok=True)

print(f"正在从 {RAW} 读取数据...")
y = pd.read_csv(RAW / "4.csv")
X = pd.read_csv(RAW / "1.csv")

# 提取维度名和特征名 (跳过前三列，通常是ID等)
SENS_COLS = list(y.columns[3:])
CHEM_COLS = list(X.columns[3:])

# ==========================================
# 1. 预处理化学数据 (对应工单 2.1)
# ==========================================
print("预处理化学特征...")
X_proc = X[CHEM_COLS].copy()
# 找出有缺失值的列，排除 aroma_hops_sum
na_cols = X_proc.columns[X_proc.isna().any()].tolist()
fill_zero_cols = [c for c in na_cols if c != "aroma_hops_sum"]

# 主设定：单化合物 NA 编码为 0
X_proc[fill_zero_cols] = X_proc[fill_zero_cols].fillna(0)

# 为有缺失值的列增加缺失指示变量 (0或1)
for c in fill_zero_cols:
    X_proc[f"{c}_isNA"] = X[CHEM_COLS][c].isna().astype(int)

# ==========================================
# 2. 推断品评团 (对应工单 2.4 & 检查3)
# ==========================================
print("推断品评团场次...")
def tie_pair_counts(z_df):
    cnt = {}
    for c in z_df.columns:
        # 将z值保留9位小数进行分组
        for idx in pd.Series(np.arange(len(z_df))).groupby(z_df[c].round(9).values).groups.values():
            idx = sorted(idx)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    cnt[(idx[a], idx[b])] = cnt.get((idx[a], idx[b]), 0) + 1
    return cnt

def get_components(cnt, k, n):
    # 找出在至少 k 个维度上完全并列的啤酒对
    e = [p for p, v in cnt.items() if v >= k]
    if not e:
        return np.arange(n) # 如果没有并列，每瓶酒自成一个场次
    # 使用连通图算法将它们连起来
    A = coo_matrix((np.ones(len(e)), ([i for i, _ in e], [j for _, j in e])), shape=(n, n))
    return connected_components(A, directed=False)[1]

# 统计并列对，并设置阈值 k=3
cnt = tie_pair_counts(y[SENS_COLS])
SESSION = get_components(cnt, 3, len(y))

# 获取每个场次包含的啤酒数量
session_counts = pd.Series(SESSION).value_counts()
# 找到包含 2 款及以上啤酒的场次 ID
multi_sessions = session_counts[session_counts > 1].index

# ==========================================
# 3. 计算品评团零点 (对应工单 4.2)
# ==========================================
print("计算品评团零点...")
def session_zero(col):
    z = y[col].round(9)
    out = {}
    # 只在多款啤酒的场次中寻找零点
    for g in multi_sessions:
        v = z[SESSION == g]
        # 如果该场次的最小值至少出现两次，视为全员打0分（零点）
        if (v == v.min()).sum() >= 2:
            out[g] = v.min()
    return pd.Series(out, dtype=float)

Z0 = {c: session_zero(c) for c in SENS_COLS}

# ==========================================
# 4. 模型训练与交叉验证 (对应工单 3.3 & 5.1)
# ==========================================
print("开始训练模型 (包含单调约束与 5 次重复 CV)...")

# 队长工单 3.3 要求的单调约束: 目标维度 -> (特征名, 方向)
MONOTONIC_PAIRS = {
    "alcohol": ("ethanol..v.v.", 1),
    "bitternes": ("bitterness", 1),
    "acidity": ("pH", -1),
    "lactic": ("lactic_acid.mg.L.", 1),
    "acetic": ("acetic_acid.g.L.", 1),
    "X4vg": ("X4VG", 1)
}

style = y['tasting_category_fine']
# 按场次(SESSION)分组，按风格(style)分层
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# 存放 5 次重复预测结果的列表
oof_preds_all_runs = []

for run in range(5):
    print(f"  --> 正在进行第 {run+1}/5 次交叉验证...")
    oof_preds_run = pd.DataFrame(index=y.index, columns=SENS_COLS, dtype=float)
    
    for col in SENS_COLS:
        y_target = y[col].values
        
        # 准备单调约束数组
        mono_cst = None
        if col in MONOTONIC_PAIRS:
            feature_name, direction = MONOTONIC_PAIRS[col]
            if feature_name in X_proc.columns:
                # 初始化全为0的数组，然后把指定特征的位置设为方向
                mono_cst = np.zeros(X_proc.shape[1])
                mono_cst[X_proc.columns.get_loc(feature_name)] = direction

        # 5折交叉验证
        for train_idx, test_idx in sgkf.split(X_proc, style, groups=SESSION):
            X_train, X_test = X_proc.iloc[train_idx], X_proc.iloc[test_idx]
            y_train = y_target[train_idx]
            
            model = HistGradientBoostingRegressor(
                max_iter=100, 
                learning_rate=0.05, 
                max_depth=3, 
                min_samples_leaf=10,
                monotonic_cst=mono_cst, 
                random_state=42 + run
            )
            model.fit(X_train, y_train)
            
            # 记录测试折的预测值 (折外预测)
            oof_preds_run.iloc[test_idx, oof_preds_run.columns.get_loc(col)] = model.predict(X_test)
            
    oof_preds_all_runs.append(oof_preds_run)

# 将 5 次运行的结果取平均
oof_preds_final = pd.concat(oof_preds_all_runs).groupby(level=0).mean()

# 保存结果
oof_preds_final.insert(0, "beer", y["beer"])
oof_preds_final.to_csv(RESULTS / "q1_oof_z_predictions.csv", index=False)

print(f"\n✅ 第一阶段完成！")
print(f"预测的标准化评分 (z值) 已保存至: {RESULTS / 'q1_oof_z_predictions.csv'}")
print("请将终端输出截图反馈，接下来我们将进行 z 到 0-6 分的转换！")