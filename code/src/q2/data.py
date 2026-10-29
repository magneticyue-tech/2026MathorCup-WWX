# code/src/q2/data.py
"""Q2 数据层：候选集合、样本对齐、品评团、NA 与秩。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parents[1]
for _p in (CURRENT_DIR, CODE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

import config as C
from src.q1.panels import at_zero_mask, infer_panels, panel_zeros

ID_COLS = ("beer", "beer_id", "tasting_category_fine")


@dataclass(frozen=True)
class Q2Dataset:
    """Q2 主流程所需的数据包。

    - chem_raw：附件 1 的 231 个化学列原值（含 NA），行按附件 1/4 的 beer 对齐。
    - sensory_raw：附件 4 的 50 个感官列；Y_rank 只含 49 个交付维度（无 barnyard）。
    - panels：问题 1 §2.4 的 k=3 推断品评团（63 个，含单例）。
    """

    chem_raw: pd.DataFrame
    corr: pd.DataFrame
    sensory_raw: pd.DataFrame
    sample_ids: pd.Index
    chem_cols: list[str]
    candidate_cols: list[str]
    annotation_cols: list[str]
    sensory_cols: list[str]
    delivery_sensory_cols: list[str]
    excluded_sensory_cols: list[str]
    style: np.ndarray
    panels: np.ndarray
    date_group: np.ndarray
    X_rank: pd.DataFrame
    Y_rank: pd.DataFrame


def fill_chem(df: pd.DataFrame, scheme: str = "mean") -> pd.DataFrame:
    """化学 NA 处理。mean：本样本列均值占位；lowest：NA 排在该列最低并列。"""
    out = df.astype(float)
    if scheme == "mean":
        return out.fillna(out.mean(axis=0))
    if scheme == "lowest":
        return out.fillna(-np.inf)
    raise ValueError(f"未知 NA 方案：{scheme}")


def rank_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """按列平均秩后标准化（均值 0、方差 1，ddof=0）。常数列报错。"""
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"秩变换前仍有 NA：{bad[:10]}")
    ranks = df.rank(axis=0, method="average")
    sd = ranks.std(axis=0, ddof=0)
    if (sd == 0).any():
        raise ValueError(f"存在常数列：{sd[sd == 0].index.tolist()[:10]}")
    return (ranks - ranks.mean(axis=0)) / sd


def load_q2_dataset(verbose: bool = False) -> Q2Dataset:
    chem = pd.read_csv(C.CHEM_PATH)
    sens = pd.read_csv(C.SENSORY_PATH)
    if not (chem["beer"].values == sens["beer"].values).all():
        raise ValueError("附件 1 与附件 4 的 beer 顺序不一致")
    if not (chem["tasting_category_fine"].values == sens["tasting_category_fine"].values).all():
        raise ValueError("附件 1 与附件 4 的风格不一致")
    ids = pd.Index(chem["beer"].astype(str), name="sample_id")
    if not ids.is_unique or len(ids) != 250:
        raise ValueError(f"样本 ID 应为 250 个且唯一，实际 {len(ids)}")

    chem_cols = [c for c in chem.columns if c not in ID_COLS]
    sens_cols = [c for c in sens.columns if c not in ID_COLS]
    if len(chem_cols) != 231 or len(sens_cols) != 50:
        raise ValueError(f"化学列 {len(chem_cols)}（应 231），感官列 {len(sens_cols)}（应 50）")
    chem_raw = chem[chem_cols].astype(float).set_axis(ids, axis=0)
    sens_raw = sens[sens_cols].astype(float).set_axis(ids, axis=0)

    corr = pd.read_csv(C.CORR_PATH, index_col=0)
    if corr.shape != (231, 231):
        raise ValueError(f"附件 2 形状应为 (231, 231)，实际 {corr.shape}")
    # 附件 2 行列为全称，顺序与附件 1 化学列一致；按位置对齐，复算误差由 check_attachment6 验收。
    corr = pd.DataFrame(corr.values.astype(float), index=chem_cols, columns=chem_cols)

    annotation = [c for c in C.ANNOTATION_ONLY_COLS if c in chem_cols]
    if len(annotation) != 6:
        raise ValueError(f"注释列应为 6 个，实际 {annotation}")
    candidates = [c for c in chem_cols if c not in annotation]

    excluded = [c for c in sens_cols if c in C.SENSORY_EXCLUDE]
    delivery = [c for c in sens_cols if c not in excluded]
    if len(delivery) != C.SENSORY_DELIVERY_EXPECTED or "stale_hops" not in delivery:
        raise ValueError(f"交付维度应为 49 且含 stale_hops，实际 {len(delivery)}")

    panels = infer_panels(sens_raw.reset_index(drop=True), k=3)
    n_panels = len(np.unique(panels))
    if n_panels != C.N_PANELS_EXPECTED:
        raise ValueError(f"品评团应为 {C.N_PANELS_EXPECTED}，实际 {n_panels}")

    style = chem["tasting_category_fine"].astype(str).values
    if len(np.unique(style)) != C.N_STYLES_EXPECTED:
        raise ValueError(f"风格应为 {C.N_STYLES_EXPECTED} 个")

    q1 = np.load(C.Q1_DATA_NPZ, allow_pickle=False)
    if not (q1["beer"] == chem["beer"].values).all():
        raise ValueError("Q1 q1_data.npz 的 beer 顺序与附件 1 不一致")
    date_group = q1["date_group"].astype(int)

    X_rank = rank_standardize(fill_chem(chem_raw[candidates], "mean"))
    Y_rank = rank_standardize(sens_raw[delivery])
    if verbose:
        print(f"样本 {len(ids)}；候选 {len(candidates)}；注释 {len(annotation)}；"
              f"交付维度 {len(delivery)}；品评团 {n_panels}；风格 {len(np.unique(style))}")
    return Q2Dataset(
        chem_raw=chem_raw,
        corr=corr,
        sensory_raw=sens_raw,
        sample_ids=ids,
        chem_cols=chem_cols,
        candidate_cols=candidates,
        annotation_cols=annotation,
        sensory_cols=sens_cols,
        delivery_sensory_cols=delivery,
        excluded_sensory_cols=excluded,
        style=style,
        panels=panels,
        date_group=date_group,
        X_rank=X_rank,
        Y_rank=Y_rank,
    )


def panel_zero_table(ds: Q2Dataset) -> tuple[pd.DataFrame, np.ndarray]:
    """问题 1 的名单零点（团 × 维度）与 at_zero 掩码（250 × 50，列序同附件 4）。"""
    Z = ds.sensory_raw.reset_index(drop=True)
    zeros = panel_zeros(Z, ds.panels)
    mask = at_zero_mask(Z, ds.panels, zeros)
    return zeros, mask


def anchored_reference_target(ds: Q2Dataset) -> tuple[pd.DataFrame, list[str]]:
    """有锚维度改用 z 减去本团零点；团内无零点时用该维度各团零点的中位数。"""
    zeros, _ = panel_zero_table(ds)
    anchored = [d for d in ds.delivery_sensory_cols if zeros[d].notna().any()]
    if len(anchored) != C.N_ANCHORED_EXPECTED:
        raise ValueError(f"有锚交付维度应为 {C.N_ANCHORED_EXPECTED}，实际 {len(anchored)}")
    Y = ds.sensory_raw[ds.delivery_sensory_cols].copy()
    for d in anchored:
        z0 = pd.Series(ds.panels).map(zeros[d]).fillna(zeros[d].median()).values
        Y[d] = Y[d].values - z0
    return Y, anchored


def attachment8_conflict_chem(ds: Q2Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把附件 1 与附件 8 冲突的单元格替换为附件 8 的数值；返回新化学表与替换清单。"""
    disc = pd.read_csv(C.DISCORDANCE_PATH)
    cal = pd.read_excel(C.ATT8_PATH, sheet_name=C.CALIBRATED_SHEET)
    cal = cal.set_index(cal["beer"].astype(str))
    if (disc["expected_mode"] != "identity").any():
        raise ValueError("冲突清单含非 identity 映射，不能直接替换")
    chem = ds.chem_raw.copy()
    rows = []
    for b, field, lit, cal_val in disc[["beer", "field", "csv1_literal", "calibrated_8"]].itertuples(
        index=False
    ):
        key = str(b)
        orig = chem.at[key, field]
        att8 = float(cal.at[key, field])
        if not np.isclose(orig, lit, rtol=0, atol=1e-8):
            raise ValueError(f"附件 1 原值与清单不符：beer {key} {field}")
        if not np.isclose(att8, cal_val, rtol=0, atol=1e-8):
            raise ValueError(f"附件 8 数值与清单不符：beer {key} {field}")
        chem.at[key, field] = att8
        rows.append({"beer": key, "field": field, "attachment1_value": orig, "attachment8_value": att8})
    log = pd.DataFrame(rows)
    if len(log) != 80 or log["beer"].nunique() != 24:
        raise ValueError(f"替换应为 24 款啤酒 80 个单元格，实际 {log['beer'].nunique()} / {len(log)}")
    return chem, log


if __name__ == "__main__":
    ds = load_q2_dataset(verbose=True)
    assert ds.X_rank.shape == (250, 225) and ds.Y_rank.shape == (250, 49)
    assert "barnyard" not in ds.Y_rank.columns
    assert abs(ds.X_rank.mean().abs().max()) < 1e-12
    print("数据层验收通过")
