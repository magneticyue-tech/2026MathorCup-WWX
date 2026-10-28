import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import grade

EXPECTED_MD5 = "52645e38d90aafda272532e19e2fea1c"
ATT = (
    "Malt",
    "Sweetness",
    "Hops",
    "Bitterness",
    "Esters",
    "Fullness (body)",
    "Alcohol (warming)",
    "Overall appreciation",
)
ATT_TO_DIMS = {
    "Malt": ["A_malt_all", "F_malt_all"],
    "Sweetness": ["sweetness"],
    "Hops": ["A_hops_all", "F_hops_all"],
    "Bitterness": ["bitternes"],
    "Esters": ["A_esters_all", "F_esters_all"],
    "Fullness (body)": ["body"],
    "Alcohol (warming)": ["alcohol"],
    "Overall appreciation": ["overall"],
}
CHEM_MAP = {
    "Ethanol (v/v)": "ethanol..v.v.",
    "Ethyl acetate (mg/l)": "ethyl_acetate",
    "Ethyl hexanoate (mg/l)": "ethyl_hexanoate",
    "Isoamyl acetate (mg/l)": "isopentyl_acetate",
    "Phenylethyl acetate (mg/l)": "phenethyl_acetate",
    "Glycerol (g/l)": "glycerol",
    "Lactic acid (g/l)": "lactic_acid.mg.L.",
}
BASE_STYLES = ("blond", "NABLAB")
EXPECTED_BASE_CODES = {"blond": 64, "NABLAB": 223}
EXPECTED_CONTROL_N = {"blond": 33, "NABLAB": 13}
WORKBOOK_NAME = "Supplemental Files and Figure source files.xlsx"
DOWNLOAD_URL = "https://doi.org/10.5281/zenodo.10653704"


def resolve_workbook(env=os.environ) -> Path:
    if env.get("ZENODO_XLSX"):
        path = Path(env["ZENODO_XLSX"])
    else:
        path = Path(env.get("TEMP", ".")) / "zenodo_10653704" / WORKBOOK_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 Zenodo 工作簿 {path}。请从 {DOWNLOAD_URL} 下载「{WORKBOOK_NAME}」，"
            "并把环境变量 ZENODO_XLSX 设为其路径，或放到 $TEMP/zenodo_10653704/ 下。"
        )
    return path


def check_md5(path) -> str:
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()
    if md5 != EXPECTED_MD5:
        raise RuntimeError(f"工作簿 md5 为 {md5}，应为 {EXPECTED_MD5}")
    return md5


def load_control_ratings(path):
    parts = []
    for k in (1, 2, 3):
        d = pd.read_excel(path, sheet_name=f"Validation Results {k}")
        d["date"] = pd.to_datetime(d["Time"]).dt.date
        parts.append(d)
    r = pd.concat(parts, ignore_index=True)
    n_all = len(r)
    r = r.drop_duplicates(subset=["date", "Name", "Beer", *ATT])
    n_dedup = len(r)
    det = pd.read_excel(path, sheet_name="Validation Details")
    det = det[det["SetName"].notna()].copy()
    det["date"] = pd.to_datetime(det["Date"]).dt.date
    r = r.merge(
        det[["date", "Code", "BeerStyle", "SampleName"]],
        left_on=["date", "Beer"],
        right_on=["date", "Code"],
        how="left",
    )
    assert r["SampleName"].notna().all(), "有评分未能匹配到 Validation Details"
    return r[r["SampleName"] == "Control"].copy(), {"n_all": n_all, "n_dedup": n_dedup}


def match_base_beers(path, cal) -> dict:
    conc = pd.read_excel(path, sheet_name="Validation Concentrations")
    conc = conc.drop_duplicates("BeerStyle").set_index("BeerStyle")
    L = np.log10(cal[list(CHEM_MAP.values())].to_numpy(float) + 1e-3)
    sd = np.nanstd(L, 0)
    lower = {str(s).lower(): s for s in conc.index}
    out = {}
    for style in BASE_STYLES:
        key = lower[style.lower()]
        v = np.log10(conc.loc[key, list(CHEM_MAP)].to_numpy(float) + 1e-3)
        dist = np.sqrt(np.nansum(((L - v) / sd) ** 2, axis=1))
        order = np.argsort(dist, kind="stable")
        out[style] = {
            "code": int(cal["beer"].iloc[order[0]]),
            "dist": float(dist[order[0]]),
            "dist2": float(dist[order[1]]),
        }
    return out


def control_means(ctrl, style_name) -> pd.DataFrame:
    sub = ctrl[ctrl["BeerStyle"].astype(str).str.lower() == style_name.lower()]
    rows = []
    for a in ATT:
        v = sub[a].astype(float)
        n = int(v.notna().sum())
        mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))
        for d in ATT_TO_DIMS[a]:
            rows.append({"dim": d, "attr": a, "n": n, "mean": mean, "se": se})
    return pd.DataFrame(rows, columns=["dim", "attr", "n", "mean", "se"])


def compare_rows(code, rows_of_beer, table, G_hat, G_hat_Q, G_obs, xbar, p_final, dims):
    i = int(np.atleast_1d(rows_of_beer)[0])
    dims = list(dims)
    out = []
    for rec in table.itertuples(index=False):
        j = dims.index(rec.dim)
        cg = int(grade(np.array([rec.mean]))[0])
        out.append(
            {
                "beer": int(code),
                "row": i,
                "dim": rec.dim,
                "attr": rec.attr,
                "G_hat": int(G_hat[i, j]),
                "G_hat_Q": int(G_hat_Q[i, j]),
                "G_obs": int(G_obs[i, j]),
                "xbar": float(xbar[i, j]),
                "ctrl_mean": float(rec.mean),
                "ctrl_se": float(rec.se),
                "ctrl_grade": cg,
                "p_hat_at_ctrl": float(p_final[i, j, cg]),
                "err_hat": int(G_hat[i, j]) - cg,
                "err_obs": int(G_obs[i, j]) - cg,
            }
        )
    return pd.DataFrame(out)


def _summ(df):
    out = {"n_cells": int(len(df))}
    for name, col in (("hat", "err_hat"), ("obs", "err_obs")):
        e = df[col].to_numpy(float)
        out[f"{name}_mae"] = float(np.abs(e).mean())
        out[f"{name}_exact"] = int((e == 0).sum())
        out[f"{name}_within1"] = int((np.abs(e) <= 1).sum())
        out[f"{name}_bias"] = float(e.mean())
    return out


def summarize(df) -> dict:
    res = {str(b): _summ(g) for b, g in df.groupby("beer")}
    res["all"] = _summ(df)
    return res
