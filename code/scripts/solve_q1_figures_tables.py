import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, FIG_DIR
from src.q1.data import load_q1

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("正在生成 T7 全部图表与表格...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_q1()
    q1_data = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    G_obs = q1_data['G']
    
    scores = np.load(PROCESSED_DIR / "q1_scores.npz", allow_pickle=True)
    G_hat = scores['G_hat']
    
    metrics = pd.read_json(PROCESSED_DIR / "q1_metrics.json")
    
    # 1. 混淆矩阵 (fig_q1_confusion)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(G_obs.flatten(), G_hat.flatten(), labels=range(7))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(7), yticklabels=range(7))
    plt.title("主模型等级混淆矩阵 (行归一化)")
    plt.xlabel("预测等级")
    plt.ylabel("真实等级")
    plt.savefig(FIG_DIR / "fig_q1_confusion.pdf", bbox_inches='tight')
    plt.close()
    
    # 2. 擂台对比图 (fig_q1_paired_diff)
    controls = np.load(PROCESSED_DIR / "q1_controls.npz", allow_pickle=True)
    models = ["B0", "B1", "EN", "RF", "K2", "C1", "C2"]
    
    maes = [metrics["controls"][m]["mae"] for m in models]
    main_mae = metrics["main"]["mae"]
    
    plt.figure(figsize=(10, 5))
    plt.bar(models, maes, color='gray', alpha=0.6, label='Controls')
    plt.axhline(main_mae, color='red', linestyle='--', linewidth=2, label='Main Model')
    plt.ylabel("MAE (越低越好)")
    plt.title("Q1 主模型与各对照组 MAE 对比")
    plt.legend()
    plt.savefig(FIG_DIR / "fig_q1_paired_diff.pdf", bbox_inches='tight')
    plt.close()
    
    # 3. 输出 12 款啤酒表格
    df_scores = pd.DataFrame(G_hat, columns=data.dims)
    df_scores.insert(0, "beer", data.beer)
    report_df = df_scores.head(12)
    report_df.to_csv(FIG_DIR / "tab_q1_report_beers.csv", index=False)
    
    # 4. 输出主指标表
    metric_rows = [{"Model": "Main", "MAE": main_mae, "QWK": metrics["main"]["qwk"]}]
    for m in models:
        metric_rows.append({"Model": m, "MAE": metrics["controls"][m]["mae"], "QWK": metrics["controls"][m]["qwk"]})
    pd.DataFrame(metric_rows).to_csv(FIG_DIR / "tab_q1_metrics.csv", index=False)
    
    # 5. 输出长表
    long_rows = []
    for i in range(250):
        for j, dim in enumerate(data.dims):
            long_rows.append({
                "beer_row": i+1,
                "beer": data.beer.iloc[i],
                "dim": dim,
                "G_hat": G_hat[i, j],
                "G_obs": G_obs[i, j],
                "xbar_obs": q1_data['xbar'][i, j]
            })
    pd.DataFrame(long_rows).to_csv(PROCESSED_DIR / "q1_scores_long.csv", index=False)
    
    print("✅ T7 全部图表与表格生成完毕，已存至 figures/q1/")

if __name__ == "__main__":
    main()