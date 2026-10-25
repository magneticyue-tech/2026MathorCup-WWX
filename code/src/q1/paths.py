from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 1. 原始数据现在被移到了 code/data/raw/
RAW_DIR = ROOT / "code" / "data" / "raw" 

# 2. 处理后的数据被加上了 q1/ 子目录
PROCESSED_DIR = ROOT / "code" / "data" / "processed" / "q1"

# 3. 图片被加上了 q1/ 子目录
FIG_DIR = ROOT / "figures" / "q1"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0