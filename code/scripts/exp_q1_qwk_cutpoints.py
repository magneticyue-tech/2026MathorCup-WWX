"""探索 Q1 六切点 QWK 后处理；不修改正式评分。

使用已有的演示版折外 z 预测。校准标签是方法论定义的参考等级，
不是原始逐人 0–6 打分。连续映射参数固定为现有交付 CSV 所用的
全样本版本；只在训练折学习切点。现有折外预测跨重复共享训练样本，
故本实验仅用于探索，不能充当最终无泄漏的外部验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "code" / "data" / "processed"
OUT = ROOT / "results" / "q1_qwk_cutpoints_experiment"
NO_ANCHOR = {
    "A_malt_all", "A_malt_grain", "A_esters_all", "F_hops_all",
    "F_esters_all", "alcohol", "aftertaste", "body", "co2",
    "overall", "F_malt_all",
}
ANCHOR_CUTPOINTS = np.arange(6, dtype=float) + 0.5
QUANTILE_GRID = np.linspace(0.0, 1.0, 41)


def grades(value: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    return np.searchsorted(cutpoints, value, side="right").astype(np.int8)


def qwk(y: np.ndarray, pred: np.ndarray) -> float:
    """Quadratic weighted kappa; the common /36 factor cancels."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    observed = np.mean((y - pred) ** 2)
    expected = np.mean(y**2) + np.mean(pred**2) - 2 * np.mean(y) * np.mean(pred)
    if expected <= 1e-12:
        return 1.0 if observed <= 1e-12 else 0.0
    return float(1 - observed / expected)


def optimize_cutpoints(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Anchored start; coordinate ascent over 41 quantiles, current, +inf."""
    cuts = ANCHOR_CUTPOINTS.copy()
    best = qwk(y, grades(x, cuts))
    candidates = np.unique(np.quantile(x, QUANTILE_GRID))
    for _ in range(12):
        changed = False
        for e in range(6):
            lower = cuts[e - 1] if e else -np.inf
            upper = cuts[e + 1] if e < 5 else np.inf
            trial = np.unique(np.r_[candidates, cuts[e], np.inf])
            trial = trial[(trial >= lower) & (trial <= upper)]
            chosen = cuts[e]
            for threshold in trial:
                candidate_cuts = cuts.copy()
                candidate_cuts[e] = threshold
                score = qwk(y, grades(x, candidate_cuts))
                if score > best + 1e-12:
                    best = score
                    chosen = threshold
            if chosen != cuts[e]:
                cuts[e] = chosen
                changed = True
        if not changed:
            break
    assert np.all(np.diff(cuts) >= 0)
    return cuts, best


def matching_cutpoints(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    proportions = np.array([(y < e).mean() for e in range(1, 7)])
    cuts = np.quantile(x, proportions)
    cuts[proportions >= 1] = np.inf
    return cuts


def continuous_scores(
    dim: str, pred_z: np.ndarray, observed_z: np.ndarray,
    group: np.ndarray, zero_by_group: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed full-sample scale; observed reference uses its own group zero."""
    if dim in NO_ANCHOR:
        scale = np.max(np.abs(observed_z))
        assert scale > 0
        return (
            np.clip(3 + 3 * pred_z / scale, 0, 6),
            np.clip(3 + 3 * observed_z / scale, 0, 6),
        )
    z0 = float(np.median(list(zero_by_group.values())))
    r = -z0
    factor = 3 * r / (1 + r**2)
    own_zero = np.array([zero_by_group.get(int(g), z0) for g in group])
    return (
        np.clip(factor * (pred_z - z0), 0, 6),
        np.clip(factor * (observed_z - own_zero), 0, 6),
    )


def main() -> None:
    z = pd.read_csv(RAW / "4.csv")
    final = pd.read_csv(PROCESSED / "q1_final_scores.csv")
    dims = list(final.columns[1:])
    assert list(z["beer"]) == list(final["beer"])
    oof = np.load(PROCESSED / "q1_main_oof_demo.npz")
    panel_file = np.load(PROCESSED / "q1_panels.npz", allow_pickle=True)
    groups = panel_file["panels"]
    zeros = panel_file["zeros"]
    zero_groups = panel_file["zero_panel_ids"]
    assert list(panel_file["dims"]) == dims

    records = []
    pooled_records = []
    for repeat in range(5):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=repeat)
        folds = list(cv.split(np.zeros(len(z)), z["tasting_category_fine"], groups))
        for j, dim in enumerate(dims):
            pred_z = oof[dim]
            observed_z = z[dim].to_numpy()
            pooled_ref = np.full(len(z), -1, dtype=np.int8)
            pooled_pred = {
                method: np.full(len(z), -1, dtype=np.int8)
                for method in ("anchored", "distribution_matching", "qwk_optimized")
            }
            train_qwks = []
            zero_by_group = {
                int(g): float(v) for g, v in zip(zero_groups, zeros[:, j])
                if np.isfinite(v)
            }
            continuous, reference_cont = continuous_scores(
                dim, pred_z, observed_z, groups, zero_by_group
            )
            assert np.isfinite(continuous).all() and np.isfinite(reference_cont).all()
            for fold, (train, test) in enumerate(folds):
                y_train = np.rint(reference_cont[train]).astype(np.int8)
                y_test = np.rint(reference_cont[test]).astype(np.int8)
                assert y_train.min() >= 0 and y_train.max() <= 6
                assert y_test.min() >= 0 and y_test.max() <= 6
                x_train, x_test = continuous[train], continuous[test]
                cut_anchored = ANCHOR_CUTPOINTS
                cut_matching = matching_cutpoints(x_train, y_train)
                cut_qwk, train_qwk = optimize_cutpoints(x_train, y_train)
                pooled_ref[test] = y_test
                train_qwks.append(train_qwk)
                for method, cuts in (
                    ("anchored", cut_anchored),
                    ("distribution_matching", cut_matching),
                    ("qwk_optimized", cut_qwk),
                ):
                    predicted = grades(x_test, cuts)
                    pooled_pred[method][test] = predicted
                    records.append({
                        "repeat": repeat, "fold": fold, "dimension": dim,
                        "method": method, "n": len(test),
                        "qwk": qwk(y_test, predicted),
                        "mae": float(np.mean(np.abs(y_test - predicted))),
                        "within_1": float(np.mean(np.abs(y_test - predicted) <= 1)),
                        "n_grades": int(len(np.unique(predicted))),
                        "dominant_share": float(np.max(np.bincount(predicted, minlength=7)) / len(test)),
                        "changed_from_anchored": float(np.mean(predicted != grades(x_test, cut_anchored))),
                        "train_qwk_optimized": train_qwk if method == "qwk_optimized" else np.nan,
                        "cuts": json.dumps([float(v) for v in cuts]),
                    })
            assert (pooled_ref >= 0).all()
            anchored_pred = pooled_pred["anchored"]
            for method, predicted in pooled_pred.items():
                assert (predicted >= 0).all()
                pooled_records.append({
                    "repeat": repeat, "dimension": dim, "method": method,
                    "qwk": qwk(pooled_ref, predicted),
                    "mae": float(np.mean(np.abs(pooled_ref - predicted))),
                    "within_1": float(np.mean(np.abs(pooled_ref - predicted) <= 1)),
                    "n_grades": int(len(np.unique(predicted))),
                    "dominant_share": float(np.max(np.bincount(predicted, minlength=7)) / len(z)),
                    "changed_from_anchored": float(np.mean(predicted != anchored_pred)),
                    "train_qwk_optimized": float(np.mean(train_qwks)) if method == "qwk_optimized" else np.nan,
                })

    OUT.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(records)
    detail.to_csv(OUT / "fold_metrics.csv", index=False)
    pooled = pd.DataFrame(pooled_records)
    pooled.to_csv(OUT / "repeat_metrics.csv", index=False)
    by_dim = pooled.groupby(["dimension", "method"], sort=False).agg(
        qwk=("qwk", "mean"), mae=("mae", "mean"),
        within_1=("within_1", "mean"),
        n_grades=("n_grades", "mean"),
        dominant_share=("dominant_share", "mean"),
        changed_from_anchored=("changed_from_anchored", "mean"),
        train_qwk_optimized=("train_qwk_optimized", "mean"),
    ).reset_index()
    by_dim.to_csv(OUT / "per_dimension.csv", index=False)
    summary = by_dim.groupby("method", sort=False).mean(numeric_only=True).round(5)
    summary.to_csv(OUT / "summary.csv")
    wide = by_dim.pivot(index="dimension", columns="method", values="qwk")
    delta = wide["qwk_optimized"] - wide["anchored"]
    extra = {
        "dimensions_improved": int((delta > 0).sum()),
        "dimensions_worsened": int((delta < 0).sum()),
        "dimensions_unchanged": int((delta == 0).sum()),
        "mean_delta_qwk": float(delta.mean()),
        "median_delta_qwk": float(delta.median()),
        "largest_gains": delta.sort_values(ascending=False).head(8).round(4).to_dict(),
        "largest_losses": delta.sort_values().head(8).round(4).to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(extra, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string())
    print(json.dumps(extra, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
