from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    COMPOSITE_COLS,
    DEFECT_CHEM,
    INTERNAL_STD_COL,
    NAMED_STYLES,
    SMALL_STYLES,
    SRC_STYLE,
)
from .paths import CHEM_DICT, Q2_CONTROLS, Q2_REPLACEMENTS, Q2_TOP3, RAW_DIR


@dataclass
class ChemData:
    X: pd.DataFrame
    style: np.ndarray
    beer: np.ndarray
    chem_cols: list
    gen_cols: list
    composite_cols: list
    internal_std_col: str

    @property
    def XG(self) -> np.ndarray:
        return self.X[self.gen_cols].to_numpy(float)

    @property
    def styles(self) -> list:
        return sorted(set(self.style))


def load_dictionary(path=CHEM_DICT) -> pd.DataFrame:
    dic = pd.read_csv(path, encoding="utf-8-sig").set_index("csv1_field")
    istd = dic.index[dic["origin_paper"] == "Internal standard solution"].tolist()
    assert istd == [INTERNAL_STD_COL], istd
    return dic


def load_chem(path=None, dic=None) -> ChemData:
    c1 = pd.read_csv(RAW_DIR / "1.csv" if path is None else path)
    assert len(c1) == 250, len(c1)
    chem_cols = list(c1.columns[3:])
    assert len(chem_cols) == 231
    X = c1[chem_cols].astype(float)
    gen_cols = [c for c in chem_cols if c not in COMPOSITE_COLS and c != INTERNAL_STD_COL]
    assert len(gen_cols) == 225
    style = c1["tasting_category_fine"].astype(str).to_numpy()
    cnt = pd.Series(style).value_counts()
    assert len(cnt) == 22 and cnt.min() == 3 and cnt.max() == 31, cnt
    for s in (*SMALL_STYLES, SRC_STYLE, *NAMED_STYLES):
        assert s in cnt.index, s
    dic = load_dictionary() if dic is None else dic
    off = set(dic.index[dic["off_flavors"].notna()])
    assert set(DEFECT_CHEM[:7]) == off, off
    assert DEFECT_CHEM[7:] == ("X4VG", "X.E..2.nonenal")
    return ChemData(
        X=X,
        style=style,
        beer=c1["beer"].to_numpy(),
        chem_cols=chem_cols,
        gen_cols=gen_cols,
        composite_cols=list(COMPOSITE_COLS),
        internal_std_col=INTERNAL_STD_COL,
    )


def is_log(col, dic) -> bool:
    return str(dic.loc[col, "transform_interpretation"]).startswith("log10")


def is_linear(col, dic) -> bool:
    return str(dic.loc[col, "transform_interpretation"]).startswith("线性")


def threshold(col, dic, kind="threshold_mean") -> float:
    th = dic.loc[col, kind]
    if pd.isna(th):
        return np.nan
    return float(np.log10(th)) if is_log(col, dic) else float(th)


def concentration(x, col, dic):
    return 10 ** np.asarray(x, float) if is_log(col, dic) else np.asarray(x, float)


def defect_table(chem: ChemData, dic) -> pd.DataFrame:
    rows = []
    s = chem.style
    groups = [("all", None), ("pils", SRC_STYLE), ("tripel", "Tripel")]
    groups += [("kriek", "Kriek"), ("lambic", "Lambic")]
    for c in DEFECT_CHEM:
        x = chem.X[c].to_numpy(float)
        r = {
            "field": c,
            "chemical_name": dic.loc[c, "chemical_name"],
            "scale": "log10" if is_log(c, dic) else "linear",
            "unit": dic.loc[c, "unit"],
            "threshold_mean": dic.loc[c, "threshold_mean"],
            "threshold_median": dic.loc[c, "threshold_median"],
            "n_na": int(np.isnan(x).sum()),
            "n_na_pils": int(np.isnan(x[s == SRC_STYLE]).sum()),
        }
        conc = concentration(x, c, dic)
        for kind in ("threshold_mean", "threshold_median"):
            th = dic.loc[c, kind]
            for key, st in groups:
                m = np.ones(len(s), bool) if st is None else s == st
                if pd.isna(th):
                    r[f"oav_ge1_{kind[10:]}_{key}"] = np.nan
                else:
                    r[f"oav_ge1_{kind[10:]}_{key}"] = float(np.mean(conc[m] / th >= 1))
        rows.append(r)
    return pd.DataFrame(rows)


def apply_floor_na(XG: np.ndarray) -> np.ndarray:
    """G-S6：单化合物 NA 改为该列非缺失观测最小值。"""
    out = XG.copy()
    na = np.isnan(out)
    mins = np.nanmin(out, 0)
    out[na] = np.broadcast_to(mins, out.shape)[na]
    return out


def apply_attachment8(chem: ChemData, path=Q2_REPLACEMENTS) -> tuple:
    """G-S7：按 Q2 保存的替换表把附件 1 的 80 个单元格替换为附件 8 取值。"""
    rep = pd.read_csv(path)
    assert len(rep) == 80 and rep["beer"].nunique() == 24, (len(rep), rep["beer"].nunique())
    X = chem.X.copy()
    row_of = {b: i for i, b in enumerate(chem.beer)}
    for _, r in rep.iterrows():
        i = row_of[r["beer"]]
        v = X.at[i, r["field"]]
        assert np.isnan(v) or np.isclose(v, r["attachment1_value"]), (r["beer"], r["field"])
        X.at[i, r["field"]] = r["attachment8_value"]
    new = ChemData(
        X=X,
        style=chem.style,
        beer=chem.beer,
        chem_cols=chem.chem_cols,
        gen_cols=chem.gen_cols,
        composite_cols=chem.composite_cols,
        internal_std_col=chem.internal_std_col,
    )
    return new, rep


def load_q2_top3(dims, top3_path=Q2_TOP3, controls_path=Q2_CONTROLS) -> dict:
    """Q2 冻结产物：合并 Top-3（method == main，含符号与共变标记）与风格内 Top-3。

    返回 {dim: {"merged": [(compound, sign, covariation), ...], "within": [compound, ...]}}；
    Q2 中没有的维度取 None，原因写在 "reason"。"""
    t = pd.read_csv(top3_path)
    t = t[t["method"] == "main"]
    c = pd.read_csv(controls_path).set_index("dimension")
    out = {}
    for d in dims:
        g = t[t["dimension"] == d].sort_values("rank")
        if g.empty or d not in c.index:
            out[d] = {"merged": None, "within": None, "reason": "Q2 冻结产物中没有该维度"}
            continue
        merged = list(zip(g["compound"], g["sign"], g["covariation"].astype(bool)))
        w = c.loc[d, "Within_top3"]
        within = [] if pd.isna(w) else str(w).split("|")
        out[d] = {"merged": merged, "within": within, "reason": ""}
    return out
