import json

import numpy as np

from .composite import CompositeDeriver
from .data import ChemData, apply_attachment8, load_chem
from .generator import StyleGen
from .modules import modules_from_q2

PRERUN_CONFIG = {"kind": "famod", "k": 15, "kappa": 2.0}
VARIANTS = ("base", "mean_rank", "na_floor", "att8")


def load_final_config(path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在；正式运行须先运行 solve_q3_select.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["final_config"]


def variant_data(variant="base", dic=None):
    chem = load_chem(dic=dic)
    extra = {}
    if variant == "att8":
        chem, rep = apply_attachment8(chem)
        extra["replacements"] = rep
    return chem, extra


def gen_kwargs(variant):
    return {
        "tie_scheme": "mean_rank" if variant == "mean_rank" else "cond_mean",
        "na_fill": "min" if variant == "na_floor" else "mean",
    }


def fit_full(chem: ChemData, cfg: dict, variant="base", style_scale=False):
    """全数据生成器（Q2 模块表）与综合指标派生器。"""
    kw = gen_kwargs(variant)
    gen = StyleGen(kind=cfg["kind"], k=cfg["k"], kappa=cfg["kappa"], style_scale=style_scale, **kw)
    gen.fit(chem.XG, chem.style, chem.styles, module_labels=modules_from_q2(chem.gen_cols))
    comp = CompositeDeriver(kw["tie_scheme"]).fit(
        gen.U, chem.X[chem.composite_cols].to_numpy(float)
    )
    return gen, comp


def x_filled(chem: ChemData, gen: StyleGen) -> np.ndarray:
    XG = chem.XG
    return np.where(np.isnan(XG), gen.colmean, XG)
