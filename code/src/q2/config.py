# code/src/q2/config.py
"""Q2 全局配置（双路径精准锁定版）。"""
from __future__ import annotations
from pathlib import Path

# 动态获取项目根目录 (向上找 3 层: q2 -> src -> code -> 项目根目录 MathorCupWWX)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# =====================================================================
# 1. 根目录的 data 文件夹（存放原始数据 1~8，以及 Q2 的输出图/表）
# =====================================================================
ROOT_DATA = PROJECT_ROOT / "data"
DATA_RAW = ROOT_DATA / "raw"
Q2_DIR = ROOT_DATA / "processed" / "q2"

# =====================================================================
# 2. code 目录下的 data 文件夹（存放 Q1 跑出来的处理结果）
# =====================================================================
CODE_DATA = PROJECT_ROOT / "code" / "data"
Q1_PROCESSED = CODE_DATA / "processed"

FIGURES = PROJECT_ROOT / "figures" / "q2"

# 确保 Q2 输出文件夹存在
Q2_DIR.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# 【原始数据路径】(在根目录 data/raw)
CHEM_PATH = DATA_RAW / "1.csv"
CORR_PATH = DATA_RAW / "2.csv"
SENSORY_PATH = DATA_RAW / "4.csv"
THRESHOLD_PATH = DATA_RAW / "8.xlsx"

# 【Q1 接口文件】(在 code/data/processed)
PANEL_NPZ_PATH = Q1_PROCESSED / "q1_panels.npz"

# =====================================================================
# 运行前固定参数
# =====================================================================
SEED = 0
TAU_MAIN = 0.6
TAU_GRID = (0.5, 0.6, 0.7)
N_BOOTSTRAP = 1000
N_PERMUTATION = 1000

STABLE_THRESHOLD = 0.6
MEDIUM_THRESHOLD = 1.0 / 3.0

COMPOSITE_COLS = (
    "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum", "kcalperc",
)
INTERNAL_STANDARD_COL = "X3.methyl.2.butenal"
ANNOTATION_ONLY_COLS = COMPOSITE_COLS + (INTERNAL_STANDARD_COL,)

SIGNAL_FLOOR = -1.81781854

BARNYARD_ALIAS = {"barnyard": "stale_hops"}
NO_PRIOR_DIMS = {"aftertaste", "overall"}