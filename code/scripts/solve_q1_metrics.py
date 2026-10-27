import sys
import numpy as np
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.metrics import grade_mae, exact_match, within1, qwk_by_dim

def main():
    print("正在执行 T5 指标计算与擂台对比...")
    data = load_q1()
    q1_data = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    G_obs = q1_data['G']
    
    scores_data = np.load(PROCESSED_DIR / "q1_scores.npz", allow_pickle=True)
    G_hat_main = scores_data['G_hat']
    
    controls = np.load(PROCESSED_DIR / "q1_controls.npz", allow_pickle=True)
    models = ["B0", "B1", "EN", "RF", "K2", "C1", "C2"]
    
    metrics = {"main": {}, "controls": {}}
    
    # 主模型指标 (全量 12500 个单元)
    metrics["main"]["mae"] = float(grade_mae(G_obs, G_hat_main))
    metrics["main"]["exact"] = float(exact_match(G_obs, G_hat_main))
    metrics["main"]["within1"] = float(within1(G_obs, G_hat_main))
    metrics["main"]["qwk"] = float(np.nanmean(qwk_by_dim(G_obs, G_hat_main)))
    
    # 对照组指标 (为了速度，这里取第0次重复的结果进行对比)
    for m in models:
        G_m = controls[f"G_{m}"][0]
        metrics["controls"][m] = {
            "mae": float(grade_mae(G_obs, G_m)),
            "qwk": float(np.nanmean(qwk_by_dim(G_obs, G_m)))
        }
        
    # 验收判定
    metrics["success"] = bool(metrics["main"]["mae"] < metrics["controls"]["B1"]["mae"])
    metrics["linear_overturn"] = bool(metrics["controls"]["EN"]["mae"] <= metrics["main"]["mae"])
    metrics["control_beats_main"] = any(metrics["controls"][m]["mae"] < metrics["main"]["mae"] for m in models)

    print("\n" + "="*50)
    print("🏆 Q1 最终模型擂台报告")
    print("="*50)
    print(f"主模型   MAE: {metrics['main']['mae']:.4f} | QWK: {metrics['main']['qwk']:.4f}")
    for m in models:
        print(f"{m:<6}   MAE: {metrics['controls'][m]['mae']:.4f} | QWK: {metrics['controls'][m]['qwk']:.4f}")
    print("-" * 50)
    print(f"成功标准 (主模型MAE < B1): {'✅ 达成' if metrics['success'] else '❌ 失败'}")
    print(f"线性模型是否推翻主模型: {'⚠️ 是' if metrics['linear_overturn'] else '否'}")
    print(f"是否有任何对照打败主模型: {'⚠️ 是' if metrics['control_beats_main'] else '否'}")
    print("="*50)
    
    save_path = PROCESSED_DIR / "q1_metrics.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ 指标已保存至: {save_path}")

if __name__ == "__main__":
    main()