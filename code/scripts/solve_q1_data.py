import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sklearn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1

def main():
    print("正在重新生成 T1 真实数据产物 (含信度、重建校验)...")
    data = load_q1()
    save_path = PROCESSED_DIR / "q1_data.npz"
    
    np.savez_compressed(
        save_path,
        xbar=data.targets.xbar, G=data.targets.G, n_ratings=data.targets.n_ratings,
        date_group=data.targets.date_group, reliability=data.targets.reliability,
        reconstruct_ok=data.targets.reconstruct_ok, half_grade_exact=data.targets.half_grade_exact,
        half_grade_qwk=data.targets.half_grade_qwk, attendance_share=data.targets.attendance_share,
        dims=np.array(data.dims), seed=SEED, meta=json.dumps({"python": sys.version, "sklearn": sklearn.__version__})
    )
    print(f"✅ T1 产物已更新，真实信度已写入。")

if __name__ == "__main__":
    main()