# code/src/q2/inspect_q1.py
import numpy as np
import pandas as pd
from pathlib import Path

# 定位到根目录的 code/data/processed
root = Path(__file__).resolve().parents[3]
q1_dir = root / "code" / "data" / "processed"

print("="*50)
print("正在解析 q1_panels.npz ...")
try:
    p = np.load(q1_dir / "q1_panels.npz", allow_pickle=True)
    print("✅ panels 里的键名:", p.files)
    # 顺便看看第一个键长什么样
    for key in p.files:
        arr = p[key]
        print(f"   -> 键 '{key}' 的形状: {arr.shape}, 类型: {arr.dtype}")
except Exception as e:
    print("❌ 读取 npz 失败:", e)

print("\n" + "="*50)
print("正在解析 q1_metrics.json ...")
try:
    m = pd.read_json(q1_dir / "q1_metrics.json", typ='series')
    print("✅ metrics 里的键名(前20个):", m.index.tolist()[:20])
except Exception as e:
    print("❌ 读取 json 失败，尝试当作普通文本读取...")
    try:
        import json
        with open(q1_dir / "q1_metrics.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("✅ 顶层键名:", list(data.keys())[:20])
    except Exception as e2:
        print("❌ 完全无法读取:", e2)