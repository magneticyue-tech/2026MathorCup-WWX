"""Compare three 0-6 mappings against an assumed linear reading of Table 4.

The reference labels are constructed from Table 4, not observed raw panel scores.
The available OOF file averages five repeats, so this is an exploratory comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from exp_q1_qwk_cutpoints import grades, optimize_cutpoints, qwk
from src.q1.convert import to_score
from src.q1.data import load_q1
from src.q1.paths import PROCESSED_DIR, ROOT, SEED
from src.q1.splits import outer_splits


OUT = ROOT / "results" / "q1_linear_reference_comparison"
METHODS = ("linear", "qwk_optimized", "group_zero_known", "group_zero_fallback")


def linear_score(z: np.ndarray, low: float, high: float) -> np.ndarray:
    if not high > low:
        raise ValueError("The training scores have no range")
    return np.clip(6 * (np.asarray(z) - low) / (high - low), 0, 6)


def own_group_zeros(
    groups: np.ndarray, zero_table: np.ndarray, group_rows: dict[int, int], dim_idx: int
) -> np.ndarray:
    return np.array([
        zero_table[group_rows[int(g)], dim_idx] if int(g) in group_rows else np.nan
        for g in groups
    ])


def training_zero(
    train_groups: np.ndarray, zero_table: np.ndarray,
    group_rows: dict[int, int], dim_idx: int,
) -> float:
    rows = [group_rows[int(g)] for g in np.unique(train_groups) if int(g) in group_rows]
    available = zero_table[rows, dim_idx]
    available = available[np.isfinite(available)]
    return float(np.median(available)) if len(available) else np.nan


def candidates(
    pred: np.ndarray, observed: np.ndarray, groups: np.ndarray,
    train: np.ndarray, test: np.ndarray, dim: str, dim_idx: int,
    zero_table: np.ndarray, group_rows: dict[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    low, high = float(observed[train].min()), float(observed[train].max())
    y_train = np.rint(linear_score(observed[train], low, high)).astype(np.int8)
    y_test = np.rint(linear_score(observed[test], low, high)).astype(np.int8)
    x_train = linear_score(pred[train], low, high)
    x_test = linear_score(pred[test], low, high)
    cuts, _ = optimize_cutpoints(x_train, y_train)

    z0 = training_zero(groups[train], zero_table, group_rows, dim_idx)
    m_scale = float(np.max(np.abs(observed[train])))
    own_zero = own_group_zeros(groups[test], zero_table, group_rows, dim_idx)
    known = to_score(pred[test], dim, 3.0, z0, m_scale, zero=own_zero)
    fallback = to_score(pred[test], dim, 3.0, z0, m_scale)
    predictions = {
        "linear": np.rint(x_test).astype(np.int8),
        "qwk_optimized": grades(x_test, cuts),
        "group_zero_known": np.rint(known).astype(np.int8),
        "group_zero_fallback": np.rint(fallback).astype(np.int8),
    }
    return y_test, predictions


def compare(data, groups, zero_table, group_rows, oof) -> pd.DataFrame:
    records = []
    for repeat, folds in enumerate(outer_splits(groups, data.style, seed=SEED)):
        for dim_idx, dim in enumerate(data.dims):
            observed = data.Z[dim].to_numpy(dtype=float)
            pred = oof[dim]
            reference = np.full(len(groups), -1, dtype=np.int8)
            pooled = {method: np.full(len(groups), -1, dtype=np.int8) for method in METHODS}
            for train, test in folds:
                y_test, predictions = candidates(
                    pred, observed, groups, train, test, dim, dim_idx,
                    zero_table, group_rows,
                )
                reference[test] = y_test
                for method, values in predictions.items():
                    pooled[method][test] = values
            if (reference < 0).any() or any((p < 0).any() for p in pooled.values()):
                raise RuntimeError("An outer fold did not receive predictions")
            for method, values in pooled.items():
                records.append({
                    "repeat": repeat,
                    "dimension": dim,
                    "method": method,
                    "qwk": qwk(reference, values),
                    "mae": float(np.mean(np.abs(reference.astype(int) - values.astype(int)))),
                    "within_1": float(np.mean(np.abs(reference.astype(int) - values.astype(int)) <= 1)),
                    "n_grades": int(np.unique(values).size),
                })
    return pd.DataFrame(records)


def final_candidates(data, groups, zero_table, group_rows, oof) -> dict[str, pd.DataFrame]:
    frames = {method: {} for method in METHODS}
    for dim_idx, dim in enumerate(data.dims):
        observed = data.Z[dim].to_numpy(dtype=float)
        pred = oof[dim]
        low, high = float(observed.min()), float(observed.max())
        x_pred = linear_score(pred, low, high)
        y_ref = np.rint(linear_score(observed, low, high)).astype(np.int8)
        cuts, _ = optimize_cutpoints(x_pred, y_ref)
        z0 = training_zero(groups, zero_table, group_rows, dim_idx)
        m_scale = float(np.max(np.abs(observed)))
        own_zero = own_group_zeros(groups, zero_table, group_rows, dim_idx)
        frames["linear"][dim] = np.rint(x_pred).astype(np.int8)
        frames["qwk_optimized"][dim] = grades(x_pred, cuts)
        frames["group_zero_known"][dim] = np.rint(
            to_score(pred, dim, 3.0, z0, m_scale, zero=own_zero)
        ).astype(np.int8)
        frames["group_zero_fallback"][dim] = np.rint(
            to_score(pred, dim, 3.0, z0, m_scale)
        ).astype(np.int8)
    return {name: pd.DataFrame(frame) for name, frame in frames.items()}


def main() -> None:
    data = load_q1()
    panel_file = np.load(PROCESSED_DIR / "q1_panels.npz", allow_pickle=True)
    groups = panel_file["panels"]
    zero_table = panel_file["zeros"]
    group_rows = {int(g): i for i, g in enumerate(panel_file["zero_panel_ids"])}
    if list(panel_file["dims"]) != data.dims:
        raise ValueError("Panel zeros do not match Table 4 dimensions")
    oof_file = np.load(PROCESSED_DIR / "q1_main_oof_demo.npz")
    oof = {dim: np.asarray(oof_file[dim], dtype=float) for dim in data.dims}
    if any(len(v) != len(groups) or not np.isfinite(v).all() for v in oof.values()):
        raise ValueError("Invalid OOF prediction")

    detail = compare(data, groups, zero_table, group_rows, oof)
    summary = detail.groupby("method", sort=False)[["qwk", "mae", "within_1", "n_grades"]].mean()
    eligible = ("linear", "qwk_optimized", "group_zero_fallback")
    winner = summary.loc[list(eligible), "qwk"].idxmax()
    summary["eligible_for_chemical_prediction"] = summary.index.isin(eligible)
    final = final_candidates(data, groups, zero_table, group_rows, oof)
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "repeat_dimension_metrics.csv", index=False)
    summary.to_csv(OUT / "summary.csv")
    scores = final[winner].copy()
    scores.insert(0, "beer", data.beer)
    scores.to_csv(OUT / "selected_scores_exploratory.csv", index=False)

    known_cells = int(np.isfinite(zero_table).sum())
    counts = np.array([(groups == g).sum() for g in panel_file["zero_panel_ids"]])
    covered_scores = int((np.isfinite(zero_table) * counts[:, None]).sum())
    suspect_dim_idx = data.dims.index("F_malt_all")
    excluded_zeros = int(np.isfinite(zero_table[:, suspect_dim_idx]).sum())
    excluded_scores = int((np.isfinite(zero_table[:, suspect_dim_idx]) * counts).sum())
    stats = {
        "multi_groups": len(group_rows),
        "single_groups": len(np.unique(groups)) - len(group_rows),
        "dimensions": len(data.dims),
        "known_group_dimension_zeros": known_cells,
        "missing_multi_group_dimension_zeros": int(zero_table.size - known_cells),
        "covered_beer_dimension_scores": covered_scores,
        "usable_group_dimension_zeros": known_cells - excluded_zeros,
        "usable_beer_dimension_scores": covered_scores - excluded_scores,
        "all_beer_dimension_scores": len(groups) * len(data.dims),
        "dimensions_without_any_zero": [
            data.dims[i] for i in np.where(~np.isfinite(zero_table).any(axis=0))[0]
        ],
        "excluded_suspect_zero": "F_malt_all",
        "selected_for_chemical_prediction_by_mean_qwk": winner,
        "selected_scores_are_exploratory": True,
    }
    (OUT / "selection.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.round(4).to_string())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
