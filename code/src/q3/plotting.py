import json
from pathlib import Path

from ..q1.plotstyle import PALETTE, apply, save
from .paths import FIG_DIR, PROCESSED_DIR

GEN_LABELS = {
    "main": "主模型",
    "pure_fa": "纯因子",
    "G0": "G0 无风格",
    "G1": "G1 复制",
    "G2": "G2 独立边际",
    "G3": "G3 风格内插值",
}


def load_json(name, src=PROCESSED_DIR):
    p = Path(src) / name
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


__all__ = ["PALETTE", "apply", "save", "GEN_LABELS", "load_json", "FIG_DIR", "PROCESSED_DIR"]
