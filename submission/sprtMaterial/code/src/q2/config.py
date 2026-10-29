# code/src/q2/config.py
"""Q2 全局配置。"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "code" / "data" / "processed"
Q1_DIR = PROCESSED_DIR / "q1"
Q2_DIR = PROCESSED_DIR / "q2"
SENS_DIR = Q2_DIR / "sensitivity"
SMOKE_DIR = Q2_DIR / "_smoke"
FIGURES = PROJECT_ROOT / "figures" / "q2"

for _d in (Q2_DIR, SENS_DIR, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

CHEM_PATH = DATA_RAW / "1.csv"
CORR_PATH = DATA_RAW / "2.csv"
SENSORY_PATH = DATA_RAW / "4.csv"
ATT6_PATH = DATA_RAW / "6.csv"
ATT8_PATH = DATA_RAW / "8.xlsx"
THRESHOLD_SHEET = "Sensory thresholds"
CALIBRATED_SHEET = "Calibrated chemical dataset"

Q1_DATA_NPZ = Q1_DIR / "q1_data.npz"
Q1_MAIN_OOF_NPZ = Q1_DIR / "q1_main_oof.npz"

AUDIT_DIR = PROJECT_ROOT / "experimentalCode" / "results" / "data_audit"
CHEMICAL_DICT_PATH = AUDIT_DIR / "chemical_dictionary.csv"
DISCORDANCE_PATH = AUDIT_DIR / "calibrated_discordances.csv"

SEED = 0
TAU_MAIN = 0.6
TAU_GRID = (0.5, 0.6, 0.7)
N_BOOTSTRAP = 1000
N_PERMUTATION = 1000
N_SIMULATION = 1000
SIM_WEIGHTS = ((1.0, 0.75, 0.5), (1.0, 1.0, 1.0))
SIM_TARGET_R = (0.2, 0.3, 0.45)
M3_N_SPLITS = 5
M3_N_REPEATS_SPLIT = 5
M3_PERM_REPEATS = 5
ENET_L1_RATIO = 0.5
ENET_N_SPLITS = 5
ZERO_SHARE_MIN = 0.3

N_DATE_GROUPS_EXPECTED = 25
N_STYLES_EXPECTED = 22

STABLE_THRESHOLD = 0.6
MEDIUM_THRESHOLD = 1.0 / 3.0
GRADE_A_PI = 0.6
BREADTH_FLAG = 5

COMPOSITE_COLS = ("esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum", "kcalperc")
INTERNAL_STANDARD_COL = "X3.methyl.2.butenal"
ANNOTATION_ONLY_COLS = COMPOSITE_COLS + (INTERNAL_STANDARD_COL,)

SIGNAL_FLOOR = -1.81781854

# 候选类型标注（§2.1）；其余候选为单一化合物。
PHYSCHEM_COLS = ("color.EBC.", "pH", "bitterness", "ethanol..v.v.", "CO2.PSI.")
CLASS_COLS = ("betaglucan.mg.L.", "protein.g.L.", "unfermentables", "GFS")

SENSORY_EXCLUDE = ("barnyard",)
SENSORY_DELIVERY_EXPECTED = 49
STALE_HOPS_NOTE = "同 barnyard"
