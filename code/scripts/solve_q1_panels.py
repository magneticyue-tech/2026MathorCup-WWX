import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import Bunch # 用来存 meta 方便点
import sklearn

# 把 src 加入路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1
from src.q1.panels import infer_panels, panel_zeros, at_zero_mask

print("正在加载数据并推断品评团 (T1)...")
data = load_q1()

# 1. 推断品评团 (k=2 到 k=5 的敏感性测试)
panels_k = {}
for k in range(2, 6):
    panels_k[k] = infer_panels(data.Z, k=k, decimals=9)

panels_main = panels_k[3] # 主设定 k=3

# 2. 计算品评团零点
zeros = panel_zeros(data.Z, panels_main)
zero_panel_ids = zeros.index.values

# 3. 计算是否处于零点
at_zero = at_zero_mask(data.Z, panels_main, zeros)

# 4. 打包并保存为 npz
meta_info = {
    "python": sys.version,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__
}

save_path = PROCESSED_DIR / "q1_panels.npz"
np.savez_compressed(
    save_path,
    panels=panels_main,
    panels_k2=panels_k[2],
    panels_k3=panels_k[3],
    panels_k4=panels_k[4],
    panels_k5=panels_k[5],
    zeros=zeros.values,
    zero_panel_ids=zero_panel_ids,
    at_zero=at_zero,
    dims=np.array(data.dims),
    seed=SEED,
    meta=json.dumps(meta_info)
)

print(f"✅ T1 运行完毕！结果已保存至: {save_path}")
print(f"共找到 {len(np.unique(panels_main))} 个品评团（包括单款）。")