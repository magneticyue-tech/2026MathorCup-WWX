# code/src/q2/config.py
"""Q2 全局配置。"""
from __future__ import annotations
from pathlib import Path

# 动态获取项目根目录 (向上找 3 层: q2 -> src -> code -> 项目根目录 MathorCupWWX)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 路径配置：根据你的截图，data 和 code 是平级的，都在根目录下
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
Q1_DIR = DATA_PROCESSED / "q1"
Q2_DIR = DATA_PROCESSED / "q2"
FIGURES = PROJECT_ROOT / "figures" / "q2"

# 确保输出文件夹存在，如果没有会自动创建
Q2_DIR.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# 按照你的截图，数据文件直接以数字命名
CHEM_PATH = DATA_RAW / "1.csv"          # 附件1：250 x 化学列
CORR_PATH = DATA_RAW / "2.csv"          # 附件2：231 x 231 相关矩阵
SENSORY_PATH = DATA_RAW / "4.csv"       # 附件4：250 x 50 感官维度 z 值
THRESHOLD_PATH = DATA_RAW / "8.xlsx"    # 附件8：阈值/来源/内标等

# Q1 接口文件（后续用到，现在如果没生成也没关系）
PANEL_CLUSTER_PATH = Q1_DIR / "panel_clusters.csv"
AT_ZERO_PATH = Q1_DIR / "at_zero_mask.csv"

# ---------------------------------------------------------------------------
# 运行前固定参数
# ---------------------------------------------------------------------------
SEED = 0
TAU_MAIN = 0.6
TAU_GRID = (0.5, 0.6, 0.7)
N_BOOTSTRAP = 1000
N_PERMUTATION = 1000

STABLE_THRESHOLD = 0.6
MEDIUM_THRESHOLD = 1.0 / 3.0

# 候选集合：231 列 - 5 个综合指标 - 1 个内标 = 225
COMPOSITE_COLS = (
    "esters_sum",
    "aroma_hops_sum",
    "acids_sum",
    "sulfur_sum",
    "kcalperc",
)
INTERNAL_STANDARD_COL = "X3.methyl.2.butenal"
ANNOTATION_ONLY_COLS = COMPOSITE_COLS + (INTERNAL_STANDARD_COL,)

# 化学 NA 下限
SIGNAL_FLOOR = -1.81781854

# 业务约定
BARNYARD_ALIAS = {"barnyard": "stale_hops"}
NO_PRIOR_DIMS = {"aftertaste", "overall"}