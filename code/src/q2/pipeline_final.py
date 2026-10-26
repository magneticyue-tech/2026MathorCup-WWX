# code/src/q2/pipeline_final.py
"""绘制热力图"""
from __future__ import annotations
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as C

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_heatmap():
    print("绘制 Top-1 热力图...")
    df_final = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    df_top1 = df_final[(df_final["rank"] == 1) & (df_final["method"] == "main")]
    
    sig_file = C.Q2_DIR / "top3_results_with_sig.csv"
    if sig_file.exists():
        df_sig = pd.read_csv(sig_file)
        df_top1 = df_top1.drop(columns=["score"], errors="ignore").merge(
            df_sig[["dimension", "compound", "score"]], on=["dimension", "compound"], how="left"
        )
    else:
        df_top1["score"] = 0.0
    df_top1["score"] = df_top1["score"].fillna(0.0)

    top_compounds = df_top1["compound"].value_counts().head(30).index.tolist()
    matrix = pd.DataFrame(0.0, index=df_top1["dimension"].unique(), columns=top_compounds)
    for _, row in df_top1.iterrows():
        if row["compound"] in top_compounds:
            matrix.loc[row["dimension"], row["compound"]] = row["score"]

    matrix = matrix.loc[(matrix != 0).any(axis=1), (matrix != 0).any(axis=0)]
    if matrix.empty: return

    plt.figure(figsize=(14, 10))
    sns.heatmap(matrix, cmap="RdBu_r", center=0, annot=False, cbar_kws={'label': 'Spearman r'})
    plt.title("Top-1 Key Compounds across Sensory Dimensions", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(C.FIGURES / "top1_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✅ 热力图已保存至: {C.FIGURES / 'top1_heatmap.png'}")
    plt.close()

if __name__ == "__main__":
    plot_heatmap()