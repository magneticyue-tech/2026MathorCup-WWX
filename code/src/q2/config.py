# code/src/q2/config.py
"""Q2 全局配置。"""
from __future__ import annotations
from pathlib import Path

# 动态获取项目根目录 (向上找 3 层: q2 -> src -> code -> 项目根目录)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 1. 原始数据路径（当前工单：原始附件位于 data/raw）
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# 2. 处理后的数据路径（统一在 code/data/processed 下）
PROCESSED_DIR = PROJECT_ROOT / "code" / "data" / "processed"
Q1_PROCESSED = PROCESSED_DIR
Q2_DIR = PROCESSED_DIR / "q2"

# 3. 图表输出路径
FIGURES = PROJECT_ROOT / "figures" / "q2"

Q2_DIR.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# 4. 具体文件路径
CHEM_PATH = DATA_RAW / "1.csv"
CORR_PATH = DATA_RAW / "2.csv"
SENSORY_PATH = DATA_RAW / "4.csv"
THRESHOLD_PATH = DATA_RAW / "8.xlsx"
PANEL_NPZ_PATH = Q1_PROCESSED / "q1_panels.npz"

# 审计字典路径
CHEMICAL_DICT_PATH = (
    PROJECT_ROOT / "experimentalCode" / "results" / "data_audit" / "chemical_dictionary.csv"
)

# 5. 运行参数
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

# 仅用于交付层注释：在 stale_hops 的 interpretation 写“同 barnyard”。
# 严禁用于 rename / 合并计算；barnyard 不作为独立维度参与任何统计。
BARNYARD_ALIAS = {"barnyard": "stale_hops"}

# 交付维度排除项（按规范化名称比较）。barnyard 保留在输入列，但不进入 Y_rank。
SENSORY_EXCLUDE = ("barnyard",)
SENSORY_DELIVERY_EXPECTED = 49