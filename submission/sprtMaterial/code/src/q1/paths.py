from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
RAW_SENSORY = RAW_DIR / "raw_sensory_data_anonymized.csv"

PROCESSED_DIR = ROOT / "code" / "data" / "processed" / "q1"
CHUNK_DIR = PROCESSED_DIR / "_chunks"
SMOKE_DIR = PROCESSED_DIR / "_smoke"
FIG_DIR = ROOT / "figures" / "q1"

SEED = 0


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, CHUNK_DIR, SMOKE_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
