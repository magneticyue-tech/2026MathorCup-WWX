from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
# 工单强调的路径更改
RAW_DIR = ROOT / "data" / "raw" 
RAW_SENSORY = RAW_DIR / "raw_sensory_data_anonymized.csv"
PROCESSED_DIR = ROOT / "code" / "data" / "processed" / "q1"
FIG_DIR = ROOT / "figures" / "q1"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0