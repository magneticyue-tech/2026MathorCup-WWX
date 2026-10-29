"""外部加标实验：同一品评员同日加标杯与对照杯的配对差与 t 值 -> q2_spike.csv。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.external import ATT, EXPECTED_MD5, check_md5, resolve_workbook  # noqa: E402
from src.q2 import config as C  # noqa: E402

SETS = (
    ("Blond", "Best predictors"),
    ("Blond", "Best predictors - no ethanol"),
    ("NABLAB", "Best predictors - no ethanol"),
)


def load_ratings(path):
    parts = []
    for k in (1, 2, 3):
        d = pd.read_excel(path, sheet_name=f"Validation Results {k}")
        d["date"] = pd.to_datetime(d["Time"]).dt.date
        parts.append(d)
    r = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["date", "Name", "Beer", *ATT])
    det = pd.read_excel(path, sheet_name="Validation Details")
    det = det[det["SetName"].notna()].copy()
    det["date"] = pd.to_datetime(det["Date"]).dt.date
    r = r.merge(
        det[["date", "Code", "BeerStyle", "SampleName", "SetName"]],
        left_on=["date", "Beer"],
        right_on=["date", "Code"],
        how="left",
    )
    if r["SampleName"].isna().any():
        raise ValueError("有评分未能匹配到 Validation Details")
    return r


def paired(r, style, set_name):
    s = r[(r["BeerStyle"] == style) & (r["SetName"] == set_name)]
    w = s.pivot_table(
        index=["date", "Name"], columns="SampleName", values=list(ATT), aggfunc="first"
    )
    rows = []
    for a in ATT:
        d = (w[(a, "Spiked")] - w[(a, "Control")]).dropna().to_numpy(float)
        n = len(d)
        sd = d.std(ddof=1)
        rows.append(
            {
                "style": style,
                "set": set_name,
                "attribute": a,
                "n": n,
                "mean_diff": d.mean(),
                "sd_diff": sd,
                "t": d.mean() / (sd / np.sqrt(n)) if sd > 0 else (0.0 if d.mean() == 0 else np.inf),
            }
        )
    return rows


def main():
    path = resolve_workbook()
    md5 = check_md5(path)
    if md5 != EXPECTED_MD5:
        raise ValueError(f"工作簿 md5 {md5} 与预期 {EXPECTED_MD5} 不符")
    r = load_ratings(path)
    df = pd.DataFrame([row for st, sn in SETS for row in paired(r, st, sn)])
    out = C.Q2_DIR / "q2_spike.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.groupby(["style", "set"])["n"].first().to_string())
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
