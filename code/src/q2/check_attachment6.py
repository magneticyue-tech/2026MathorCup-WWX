# code/src/q2/check_attachment6.py
"""Q2 §7.1 复算验收：附件 2 与附件 1 逐对 Spearman；逐对删除边际相关复现附件 6 的 44 行。"""
import re
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd

import config as C
import core
from data import load_q2_dataset

TOL = 1e-12

SENSORY_MAP = {
    "Malt aroma - overall": "A_malt_all", "Malt aroma - grain": "A_malt_grain",
    "Malt aroma - bread": "A_malt_bread", "Malt aroma - caramel": "A_malt_cara",
    "Malt aroma - smoked": "A_malt_burn", "Hops aroma - overall": "A_hops_all",
    "Hops aroma - citrus": "A_hops_citrus", "Hops aroma - tropical fruit": "A_hops_tropical",
    "Hops aroma - grassy": "A_hops_noble", "Hops aroma - woody": "A_hops_woody",
    "Esters aroma - overall": "A_esters_all", "Esters aroma - nail polish": "A_esters_ethac",
    "Esters aroma - banana": "A_esters_isoaa", "Esters aroma - floral": "A_esters_flower",
    "Esters aroma - fruity": "A_esters_fruity", "Malt taste - overall": "F_malt_all",
    "Malt taste - grain": "F_malt_grain", "Malt taste - bread": "F_malt_bread",
    "Malt taste - caramel": "F_malt_cara", "Malt taste - smoked": "F_malt_burn",
    "Hops taste - overall": "F_hops_all", "Hops taste - citrus": "F_hops_citrus",
    "Hops taste - tropical fruit": "F_hops_tropical", "Hops taste - grassy": "F_hops_noble",
    "Hops taste - woody": "F_hops_woody", "Esters taste - overall": "F_esters_all",
    "Esters taste - nail polish": "F_esters_ethac", "Esters taste - banana": "F_esters_isoaa",
    "Esters taste - floral": "F_esters_flower", "Esters taste - fruity": "F_esters_fruity",
    "Acidity": "acidity", "Bitterness": "bitternes", "Sweetness": "sweetness",
    "4-vinyl guaiacol": "X4vg", "Diacetyl": "diacetyl", "Dimethyl sulfide": "dms",
    "Metallic": "metallic", "Stale hops": "stale_hops", "Trans-2-nonenal": "t2n",
    "Orange": "orange", "Coriander": "coriander", "Clove": "clove",
    "Lactic acid": "lactic", "Acetic acid": "acetic", "Barnyard": "barnyard",
    "Alcohol": "alcohol", "Carbonation": "co2", "Aftertaste": "aftertaste",
    "Body": "body", "Overall appreciation (Trained panel)": "overall",
}


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def pairwise_marginal(chem: pd.DataFrame, sens: pd.DataFrame):
    """每一对 (化学列, 感官维度) 只用两列都非缺失的啤酒计算 Spearman；返回系数与有效样本数。"""
    X = chem.to_numpy(float)
    Y = sens.to_numpy(float)
    R = np.full((X.shape[1], Y.shape[1]), np.nan)
    N = np.zeros_like(R, dtype=int)
    for j in range(X.shape[1]):
        for d in range(Y.shape[1]):
            rows = ~np.isnan(X[:, j]) & ~np.isnan(Y[:, d])
            Zx, okx = core.zrank(X[rows][:, [j]])
            Zy, oky = core.zrank(Y[rows][:, [d]])
            N[j, d] = rows.sum()
            if okx[0] and oky[0]:
                R[j, d] = float(Zx[:, 0] @ Zy[:, 0] / rows.sum())
    return (pd.DataFrame(R, index=chem.columns, columns=sens.columns),
            pd.DataFrame(N, index=chem.columns, columns=sens.columns))


def match_attachment6(ds):
    a6 = pd.read_csv(C.ATT6_PATH, index_col=0)
    dic = pd.read_csv(C.CHEMICAL_DICT_PATH)
    name_to_field = dict(zip(dic["chemical_name"], dic["csv1_field"]))
    chem = []
    for r in a6.index:
        f = name_to_field.get(r)
        if f is None or f not in ds.chem_cols:
            hits = [c for c in ds.chem_cols if _norm(c) == _norm(r)]
            f = hits[0] if len(hits) == 1 else None
        if f is None:
            raise KeyError(f"附件 6 行无法匹配：{r}")
        chem.append(f)
    sens = []
    for c in a6.columns:
        key = re.sub(r"\s+", " ", str(c)).strip()
        if key not in SENSORY_MAP:
            raise KeyError(f"附件 6 列无法匹配：{c!r}")
        sens.append(SENSORY_MAP[key])
    if len(chem) != 44 or len(set(chem)) != 44 or len(sens) != 50:
        raise AssertionError(f"附件 6 匹配应为 44 × 50，实际 {len(set(chem))} × {len(sens)}")
    return a6, chem, sens


def check_att6(verbose=True):
    ds = load_q2_dataset()
    R2 = core.pairwise_spearman(ds.chem_raw.to_numpy(float))
    err2 = float(np.nanmax(np.abs(R2 - ds.corr.to_numpy())))
    a6, chem, sens = match_attachment6(ds)
    R6, N6 = pairwise_marginal(ds.chem_raw[chem], ds.sensory_raw[sens])
    err6 = float(np.nanmax(np.abs(R6.to_numpy() - a6.to_numpy(float))))
    if verbose:
        print(f"附件 2 复算最大误差 {err2:.2e}；附件 6 44×50 复算最大误差 {err6:.2e}（阈值 {TOL:g}）")
    if not (err2 <= TOL and err6 <= TOL):
        raise AssertionError(f"复算误差超限：附件 2 {err2:.2e}，附件 6 {err6:.2e}")
    return {"attachment2_max_error": err2, "attachment6_max_error": err6, "attachment6_rows": 44}


if __name__ == "__main__":
    check_att6()
