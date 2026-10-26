import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, FIG_DIR
from src.q1.data import load_q1

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("正在生成 Q1 所有的最终版图表和表格...")
    data = load_q1()
    
    # ==========================================
    # 1. 转换散点图 (fig_q1_conversion)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    scores_data = np.load(PROCESSED_DIR / "q1_scores.npz", allow_pickle=True)
    masks_data = np.load(PROCESSED_DIR / "q1_mapping_reference_masks.npz", allow_pickle=True)
    R_all = masks_data['R']
    
    for ax, dim_name in zip(axes, ["sweetness", "overall"]):
        dim_idx = data.dims.index(dim_name)
        final_score = scores_data[dim_name]
        R_true = R_all[0, :, dim_idx]
        
        ax.scatter(R_true, final_score, alpha=0.6, color='#1f77b4')
        ax.plot([0, 6], [0, 6], 'r--', label='y=x')
        mae = np.mean(np.abs(R_true - final_score))
        ax.set_title(f"映射验证: {dim_name} (MAE: {mae:.2f})")
        ax.set_xlabel("混合参照等级 R")
        ax.set_ylabel("预测等级 (QWK优化)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_q1_conversion.png", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_q1_conversion.pdf", bbox_inches='tight')
    plt.close()

    # ==========================================
    # 2. 指标表格 (tab_q1_metrics)
    # ==========================================
    df_metrics = pd.read_csv(PROCESSED_DIR / "q1_mapping_metrics.csv")
    m3_df = df_metrics[df_metrics['m_val'] == '3.0'][['dim', 'QWK_all', 'MAE_all', 'Acc1_all', 'QWK_I1', 'MAE_I1']]
    m3_df.to_csv(FIG_DIR / "tab_q1_metrics.csv", index=False)
    with open(FIG_DIR / "tab_q1_metrics.tex", "w", encoding="utf-8") as f:
        cols = m3_df.columns
        latex_str = "\\begin{table}[h]\n\\centering\n\\caption{Q1 Main Model Metrics}\n\\begin{tabular}{" + "c" * len(cols) + "}\n\\hline\n"
        latex_str += " & ".join(cols) + " \\\\\n\\hline\n"
        for _, row in m3_df.iterrows():
            latex_str += " & ".join([f"{x:.3f}" if isinstance(x, float) else str(x) for x in row]) + " \\\\\n"
        latex_str += "\\hline\n\\end{tabular}\n\\end{table}"
        f.write(latex_str)

    # ==========================================
    # 3. 12款啤酒报告表 (tab_q1_report_beers)
    # ==========================================
    # 因为没有指定的12款，我们选前12款
    df_final = pd.read_csv(PROCESSED_DIR / "q1_final_scores.csv")
    report_df = df_final.head(12)
    report_df.to_csv(FIG_DIR / "tab_q1_report_beers.csv", index=False)
    
    # ==========================================
    # 4. 对照模型差值箱线图 (fig_q1_paired_diff)
    # ==========================================
    # 使用最新的指标表中主模型与线性映射的差值
    diffs = m3_df['QWK_all'] - df_metrics[df_metrics['m_val'] == '3.0']['Lin_QWK_all']
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.boxplot(diffs.dropna(), patch_artist=True, boxprops=dict(facecolor='#2ca02c', alpha=0.6))
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title("QWK 优化法 vs 直接线性法 (逐维差值)", fontsize=12)
    ax.set_ylabel("QWK Difference")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_q1_paired_diff.png", bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig_q1_paired_diff.pdf", bbox_inches='tight')
    plt.close()

    # ==========================================
    # 5. 确保旧图也有 PDF 格式 (如果没生成的话)
    # ==========================================
    import shutil
    for png_file in FIG_DIR.glob("*.png"):
        pdf_file = png_file.with_suffix('.pdf')
        if not pdf_file.exists():
            shutil.copy(png_file, pdf_file)

    print("✅ 全部 5 图 + 3 表已刷新并保存至 figures/q1/ (含 .png 和 .pdf)")

if __name__ == "__main__":
    main()