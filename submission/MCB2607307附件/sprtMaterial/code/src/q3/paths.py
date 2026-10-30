from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "code" / "data" / "processed" / "q3"
SMOKE_DIR = PROCESSED_DIR / "_smoke"
CHUNK_DIR = PROCESSED_DIR / "_chunks"
SENS_DIR = PROCESSED_DIR / "sens"
FIG_DIR = ROOT / "figures" / "q3"
Q1_DIR = ROOT / "code" / "data" / "processed" / "q1"
Q2_DIR = ROOT / "code" / "data" / "processed" / "q2"
Q2_NODES = Q2_DIR / "graph_nodes.csv"
Q2_TOP3 = Q2_DIR / "final_top3_with_rationality.csv"
Q2_CONTROLS = Q2_DIR / "full_controls_results.csv"
Q2_REPLACEMENTS = Q2_DIR / "sensitivity" / "attachment8_replacements.csv"
CHEM_DICT = ROOT / "experimentalCode" / "results" / "data_audit" / "chemical_dictionary.csv"

SEED = 0


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
