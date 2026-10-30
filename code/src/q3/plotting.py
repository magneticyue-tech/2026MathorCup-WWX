import json
from pathlib import Path

from ..q1.plotstyle import PALETTE, apply, save
from .paths import FIG_DIR, PROCESSED_DIR

GEN_LABELS = {
    "main": "以风格为条件的潜在高斯模型",
    "pure_fa": "纯因子模型",
    "G0": "无风格条件模型",
    "G1": "复制真实样本",
    "G2": "逐列独立生成",
    "G3": "风格内插值",
}


def load_json(name, src=PROCESSED_DIR):
    p = Path(src) / name
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


__all__ = ["PALETTE", "apply", "save", "GEN_LABELS", "load_json", "FIG_DIR", "PROCESSED_DIR"]
