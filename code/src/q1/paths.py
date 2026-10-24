from pathlib import Path

# 自动定位项目根目录
ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "code" / "data" / "processed"
FIG_DIR = ROOT / "figures"

# 确保输出目录存在
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0