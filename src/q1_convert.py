"""
Q1.2: 将预测的 z 值转换为 0-6 分评分
包含：m 模型、C-mid 标度、置信区间计算
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 路径与数据加载
# ==========================================
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results" / "q1"

print("正在加载数据...")
y = pd.read_csv(RAW / "4.csv")
X = pd.read_csv(RAW / "1.csv")
# 读取我们刚跑出来的 z 预测值
z_preds = pd.read_csv(RESULTS / "q1_oof_z_predictions.csv")

SENS_COLS = list(y.columns[3:])

# 队长 4.2 节确定的无锚维度 (11个)
NO_ANCHOR_COLS = [
    "A_malt_all", "A_malt_grain", "A_esters_all", "F_hops_all", 
    "F_esters_all", "alcohol", "aftertaste", "body", "co2", "overall", 
    "F_malt_all" # F_malt_all 在 4.2 节被重新归类为无锚
]

# ==========================================
# 1. 重新推断品评团和零点 (为了获取 \tilde z_{0,d})
# ==========================================
print("重新计算品评团零点...")
def tie_pair_counts(z_df):
    cnt = {}
    for c in z_df.columns:
        for idx in pd.Series(np.arange(len(z_df))).groupby(z_df[c].round(9).values).groups.values():
            idx = sorted(idx)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    cnt[(idx[a], idx[b])] = cnt.get((idx[a], idx[b]), 0) + 1
    return cnt

def get_components(cnt, k, n):
    e = [p for p, v in cnt.items() if v >= k]
    if not e: return np.arange(n)
    A = coo_matrix((np.ones(len(e)), ([i for i, _ in e], [j for _, j in e])), shape=(n, n))
    return connected_components(A, directed=False)[1]

cnt = tie_pair_counts(y[SENS_COLS])
SESSION = get_components(cnt, 3, len(y))
session_counts = pd.Series(SESSION).value_counts()
multi_sessions = session_counts[session_counts > 1].index

def session_zero(col):
    z = y[col].round(9)
    out = {}
    for g in multi_sessions:
        v = z[SESSION == g]
        if (v == v.min()).sum() >= 2:
            out[g] = v.min()
    return pd.Series(out, dtype=float)

Z0 = {c: session_zero(c) for c in SENS_COLS}

# ==========================================
# 2. z -> 0-6 分转换 (对应队长 4.1 & 4.4)
# ==========================================
print("正在将 z 值转换为 0-6 分...")
m_param = 3  # 队长 4.3 节主值取 3
FINAL_SCORES = pd.DataFrame(index=y.index, columns=SENS_COLS, dtype=int)
LOWER_CI = pd.DataFrame(index=y.index, columns=SENS_COLS, dtype=float) # 90% 下限
UPPER_CI = pd.DataFrame(index=y.index, columns=SENS_COLS, dtype=float) # 90% 上限

# 我们用一个简单的残差 bootstrap 来模拟置信区间 (队长 4.5 节)
np.random.seed(42)
N_BOOTSTRAPS = 200

for col in SENS_COLS:
    z_pred = z_preds[col].values
    z_true = y[col].values
    
    # 计算折外残差 (用于估计预测的不确定性)
    residuals = z_true - z_pred
    
    if col in NO_ANCHOR_COLS or len(Z0[col]) == 0:
        # 使用 C-mid 标度: 3 + 3 * z / max|z|
        # 队长 4.4 节: M_d 是训练折中 |z| 的最大值，这里用全量观测最大值代替
        M_d = np.max(np.abs(z_true)) 
        
        raw_score = 3 + 3 * z_pred / (M_d + 1e-9)
        
        # 模拟置信区间
        boot_scores = []
        for _ in range(N_BOOTSTRAPS):
            z_boot = z_pred + np.random.choice(residuals, size=len(z_pred), replace=True)
            boot_scores.append(np.clip(3 + 3 * z_boot / (M_d + 1e-9), 0, 6))
            
    else:
        # 使用 m 模型 (有锚维度)
        z0_tilde = Z0[col].median() # 训练折中位零点 (我们这里简化为整体中位数)
        r_d = -z0_tilde
        if abs(r_d) < 1e-5: r_d = 1e-5
            
        raw_score = m_param * (r_d / (1 + r_d**2)) * (z_pred - z0_tilde)
        
        # 模拟置信区间
        boot_scores = []
        for _ in range(N_BOOTSTRAPS):
            z_boot = z_pred + np.random.choice(residuals, size=len(z_pred), replace=True)
            boot_scores.append(np.clip(m_param * (r_d / (1 + r_d**2)) * (z_boot - z0_tilde), 0, 6))

    # 截断并保存最终整数评分
    raw_score = np.clip(raw_score, 0, 6)
    FINAL_SCORES[col] = np.round(raw_score).astype(int)
    
    # 计算 5% 和 95% 分位数作为 90% 置信区间
    boot_scores = np.array(boot_scores)
    LOWER_CI[col] = np.percentile(boot_scores, 5, axis=0)
    UPPER_CI[col] = np.percentile(boot_scores, 95, axis=0)

# ==========================================
# 3. 保存结果 (对应队长 4.5 节交付要求)
# ==========================================
print("正在保存最终结果...")
# 保存 0-6 整数评分
FINAL_SCORES.insert(0, "beer", y["beer"])
FINAL_SCORES.to_csv(RESULTS / "q1_final_scores_0_to_6.csv", index=False)

# 保存置信区间
LOWER_CI.insert(0, "beer", y["beer"])
UPPER_CI.insert(0, "beer", y["beer"])
LOWER_CI.to_csv(RESULTS / "q1_ci_lower_90.csv", index=False)
UPPER_CI.to_csv(RESULTS / "q1_ci_upper_90.csv", index=False)

print(f"\n🎉 大功告成！")
print(f"1. 0-6 分整数评分已保存至: {RESULTS / 'q1_final_scores_0_to_6.csv'}")
print(f"2. 90% 置信区间已保存至: {RESULTS / 'q1_ci_lower_90.csv'} 和 'q1_ci_upper_90.csv'")