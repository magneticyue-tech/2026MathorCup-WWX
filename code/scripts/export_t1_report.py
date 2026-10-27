import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR
from src.q1.data import load_q1

def main():
    data = load_q1()
    q1_data = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    
    # 导出信度与重建报告
    df = pd.DataFrame({
        "dim": data.dims,
        "reliability": q1_data['reliability'],
        "half_grade_exact": q1_data['half_grade_exact'],
        "half_grade_qwk": q1_data['half_grade_qwk'],
        "attendance_share": q1_data['attendance_share'],
        "reconstruct_ok_ratio": np.mean(q1_data['reconstruct_ok'], axis=0)
    })
    df.to_csv(PROCESSED_DIR / "q1_t1_validation_report.csv", index=False)
    print("✅ T1 验证报告已生成！")

if __name__ == "__main__":
    main()