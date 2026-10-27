import sys
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1

def main():
    print("正在生成 T1 最终数据产物 q1_data.npz...")
    data = load_q1()
    
    meta_info = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__
    }
    
    save_path = PROCESSED_DIR / "q1_data.npz"
    np.savez_compressed(
        save_path,
        xbar=data.targets.xbar,
        G=data.targets.G,
        n_ratings=data.targets.n_ratings,
        date_group=data.targets.date_group,
        dims=np.array(data.dims),
        seed=SEED,
        meta=json.dumps(meta_info)
    )
    
    print(f"✅ 成功生成！保存至: {save_path}")
    
    # 执行工单要求的数值核对
    counts = np.bincount(data.targets.date_group)
    print("\n================ T1 终极验收报告 ================")
    print(f"G 的等级计数: {np.bincount(data.targets.G.flatten(), minlength=7).tolist()}")
    print(f"日期组数量: {len(counts)} (工单要求: 25)")
    print(f"日期组大小最小: {counts.min()} (工单要求: 5)")
    print(f"日期组大小最大: {counts.max()} (工单要求: 24)")
    print("==================================================")

if __name__ == "__main__":
    main()