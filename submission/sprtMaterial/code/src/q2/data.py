# code/src/q2/data.py
"""Q2 数据层：候选集合、样本对齐、感官目标 x̄、日期组、NA 与秩。"""

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

ID_COLS = ("beer", "beer_id", "tasting_category_fine")


@dataclass(frozen=True)
class Q2Dataset:
    """Q2 主流程所需的数据包。

    - chem_raw：附件 1 的 231 个化学列原值（含 NA），行序同附件 1/4。
    - xbar：问题 1 `q1_data.npz` 的原始评分均值（250 × 50），行号 beer_row 与附件 1/4 行序一致；
      Y_rank 为其中 49 个交付维度（无 barnyard）的标准化秩，是主感官目标（方法 §2.3）。
    - z：附件 4 的 50 个标准化感官列，只用于感官目标敏感性臂与附件 6 复算（方法 §7.1、§7.3）。
    - date_group：问题 1 §2.4 的 25 个日期组，bootstrap 与分组 CV 的单位（方法 §2.4）。
    """

    chem_raw: pd.DataFrame
    corr: pd.DataFrame
    xbar: pd.DataFrame
    z: pd.DataFrame
    sample_ids: pd.Index
    chem_cols: list[str]
    candidate_cols: list[str]
    annotation_cols: list[str]
    sensory_cols: list[str]
    delivery_sensory_cols: list[str]
    excluded_sensory_cols: list[str]
    style: np.ndarray
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


def _median_dim_spearman(A: np.ndarray, B: np.ndarray) -> float:
    ra = pd.DataFrame(A).rank().to_numpy()
    rb = pd.DataFrame(B).rank().to_numpy()
    return float(np.median([np.corrcoef(ra[:, d], rb[:, d])[0, 1] for d in range(A.shape[1])]))


def load_q1_inputs(chem: pd.DataFrame, z: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """读取问题 1 solve_q1_data.py 产物的 xbar 与 date_group，并执行输入口径的运行前断言。"""
    q1 = np.load(C.Q1_DATA_NPZ, allow_pickle=False)
    for k in ("beer_row", "beer", "xbar", "date_group", "dims"):
        if k not in q1.files:
            raise KeyError(f"q1_data.npz 缺少键 {k}")
    if not (q1["beer_row"] == np.arange(1, 251)).all():
        raise ValueError("q1_data.npz 的 beer_row 应恰为 1..250")
    if not (q1["beer"] == chem["beer"].values).all():
        raise ValueError("q1_data.npz 的 beer 与附件 1 行序不一致")
    if list(q1["dims"]) != list(z.columns) or q1["xbar"].shape != (250, 50):
        raise ValueError("q1_data.npz 的 xbar 维度或形状与附件 4 不一致")
    date_group = q1["date_group"].astype(int)
    if date_group.shape != (250,) or set(np.unique(date_group)) != set(range(C.N_DATE_GROUPS_EXPECTED)):
        raise ValueError(f"date_group 取值应恰为 0..{C.N_DATE_GROUPS_EXPECTED - 1}")
    xbar = pd.DataFrame(q1["xbar"].astype(float), index=z.index, columns=z.columns)
    if xbar.isna().any().any():
        raise ValueError("xbar 含 NA")
    rho_row = _median_dim_spearman(xbar.to_numpy(), z.to_numpy())
    order = np.argsort(chem["beer"].to_numpy())
    rho_beer = _median_dim_spearman(xbar.to_numpy(), z.to_numpy()[order])
    if rho_row < 0.95 or rho_beer > 0.1:
        raise ValueError(f"xbar 与 z 对齐检查失败：按行号 {rho_row:.3f}（应 ≥0.95），按 beer 列 {rho_beer:.3f}（应 ≤0.1）")
    return xbar, date_group


def input_checks() -> dict:
    """运行记录用：输入断言的数值与各日期组大小。"""
    chem = pd.read_csv(C.CHEM_PATH)
    sens = pd.read_csv(C.SENSORY_PATH)
    z = sens[[c for c in sens.columns if c not in ID_COLS]].astype(float)
    xbar, date_group = load_q1_inputs(chem, z)
    order = np.argsort(chem["beer"].to_numpy())
    return {
        "xbar_z_median_spearman_by_row": _median_dim_spearman(xbar.to_numpy(), z.to_numpy()),
        "xbar_z_median_spearman_by_beer": _median_dim_spearman(xbar.to_numpy(), z.to_numpy()[order]),
        "n_date_groups": int(len(np.unique(date_group))),
        "date_group_sizes": {int(g): int(n) for g, n in enumerate(np.bincount(date_group))},
    }


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
    z = sens[sens_cols].astype(float).set_axis(ids, axis=0)

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

    style = chem["tasting_category_fine"].astype(str).values
    if len(np.unique(style)) != C.N_STYLES_EXPECTED:
        raise ValueError(f"风格应为 {C.N_STYLES_EXPECTED} 个")

    xbar, date_group = load_q1_inputs(chem, z)

    X_rank = rank_standardize(fill_chem(chem_raw[candidates], "mean"))
    Y_rank = rank_standardize(xbar[delivery])
    if verbose:
        sizes = np.bincount(date_group)
        print(f"样本 {len(ids)}；候选 {len(candidates)}；注释 {len(annotation)}；交付维度 {len(delivery)}；"
              f"日期组 {len(sizes)}（每组 {sizes.min()}–{sizes.max()} 款）；风格 {len(np.unique(style))}")
    return Q2Dataset(
        chem_raw=chem_raw,
        corr=corr,
        xbar=xbar,
        z=z,
        sample_ids=ids,
        chem_cols=chem_cols,
        candidate_cols=candidates,
        annotation_cols=annotation,
        sensory_cols=sens_cols,
        delivery_sensory_cols=delivery,
        excluded_sensory_cols=excluded,
        style=style,
        date_group=date_group,
        X_rank=X_rank,
        Y_rank=Y_rank,
    )


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
