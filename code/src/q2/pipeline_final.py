# code/src/q2/pipeline_final.py
"""Q2 最终整合：OAV 阈值计算与 Top-1 热力图。"""
from __future__ import annotations

import sys
from pathlib import Path

# 路径补丁
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from data import load_q2_dataset

# 设置绘图风格和中文显示
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def run_oav_analysis():
    print("="*50)
    print("1. 正在计算 OAV (浓度/阈值)...")
    print("="*50)
    
    try:
        df_thresh = pd.read_excel(C.THRESHOLD_PATH, sheet_name="Sensory thresholds")
    except Exception as e:
        print(f"⚠️ 读取阈值失败，跳过 OAV 计算。错误: {e}")
        return

    name_col = None
    val_col = None
    for c in df_thresh.columns:
        if 'name' in str(c).lower() or 'compound' in str(c).lower():
            name_col = c
            break
    if name_col is None:
        name_col = df_thresh.columns[1] if len(df_thresh.columns) > 1 else df_thresh.columns[0]
        
    for c in df_thresh.columns:
        if c != name_col and pd.api.types.is_numeric_dtype(df_thresh[c]):
            val_col = c
            break

    thresh_dict = {}
    for _, row in df_thresh.iterrows():
        name = str(row[name_col]).strip()
        try:
            val = float(row[val_col])
            if val > 0:
                thresh_dict[name] = val
        except:
            pass

    ds = load_q2_dataset()
    chem_df = ds.chem_raw
    
    final_file = C.Q2_DIR / "final_top3_with_rationality.csv"
    if not final_file.exists():
        print("❌ 找不到 rationality 结果文件，跳过 OAV。")
        return
    df_final = pd.read_csv(final_file)

    oav_results = []
    for idx, row in df_final.iterrows():
        comp = row["compound"]
        if comp in thresh_dict:
            threshold = thresh_dict[comp]
            conc_series = chem_df[comp]
            
            if conc_series.median() < 0 or (conc_series < 0).any():
                real_conc = 10 ** conc_series
            else:
                real_conc = conc_series
                
            oav = real_conc / threshold
            ratio_gt_1 = (oav >= 1).mean()
            oav_results.append({
                "threshold": threshold,
                "oav_ge_1_ratio": ratio_gt_1
            })
        else:
            oav_results.append({
                "threshold": np.nan,
                "oav_ge_1_ratio": np.nan
            })

    df_oav = pd.DataFrame(oav_results)
    df_final["threshold"] = df_oav["threshold"]
    df_final["oav_ge_1_ratio"] = df_oav["oav_ge_1_ratio"]
    
    out_file = C.Q2_DIR / "final_top3_with_oav.csv"
    df_final.to_csv(out_file, index=False)
    print(f"✅ OAV 计算完毕！成功匹配 {df_final['threshold'].notna().sum()} 个阈值，已保存至 final_top3_with_oav.csv")


def plot_heatmap():
    print("\n" + "="*50)
    print("2. 正在绘制 Top-1 热力图...")
    print("="*50)
    
    df_final = pd.read_csv(C.Q2_DIR / "final_top3_with_stability.csv")
    
    # 【修复】：如果 score 全是 0，去 top3_results_with_sig.csv 借真实数据
    if "score" not in df_final.columns or (df_final["score"] == 0).all():
        sig_file = C.Q2_DIR / "top3_results_with_sig.csv"
        if sig_file.exists():
            df_sig = pd.read_csv(sig_file)
            df_final = df_final.drop(columns=["score"], errors="ignore").merge(
                df_sig[["dimension", "compound", "score"]], on=["dimension", "compound"], how="left"
            )
    
    df_top1 = df_final[df_final["rank"] == 1]
    
    top_compounds = df_top1["compound"].value_counts().head(30).index.tolist()
    
    dims = df_top1["dimension"].unique()
    matrix = pd.DataFrame(0.0, index=dims, columns=top_compounds)
    
    for _, row in df_top1.iterrows():
        if row["compound"] in top_compounds:
            matrix.loc[row["dimension"], row["compound"]] = row["score"]

    # 剔除全 0 的行列
    matrix = matrix.loc[(matrix != 0).any(axis=1), (matrix != 0).any(axis=0)]
    
    # 【防呆设计】：如果矩阵为空，直接跳过
    if matrix.empty or len(matrix) == 0 or len(matrix.columns) == 0:
        print("⚠️ 矩阵为空或全为 0，跳过绘制热力图。")
        return

    plt.figure(figsize=(14, 10))
    sns.heatmap(matrix, cmap="RdBu_r", center=0, annot=False, cbar_kws={'label': 'Spearman r'})
    plt.title("Top-1 Key Compounds across Sensory Dimensions", fontsize=16)
    plt.xlabel("Compounds", fontsize=12)
    plt.ylabel("Sensory Dimensions", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    fig_path = C.FIGURES / "top1_heatmap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ 热力图已保存至: {fig_path}")
    plt.close()


if __name__ == "__main__":
    run_oav_analysis()
    plot_heatmap()
    print("\n🎉 数据和图表处理完毕！已生成热力图和最终 OAV 表格。")