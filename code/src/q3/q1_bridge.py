import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from ..q1.data import DIM_GROUPS, DIMS, ChemNAEncoder
from ..q1.discretize import apply_cuts, dist_median, predictive_dist, quantile_cuts
from ..q1.model import _fit_dim
from .paths import Q1_DIR


@dataclass
class Q1Bundle:
    models: list
    encoder: object
    final_resid: np.ndarray
    final_qcuts: np.ndarray
    dims: list
    chem_cols: list
    feature_names: list


def load_q1_bundle(model_dir=None, chem_cols=None) -> Q1Bundle:
    model_dir = Q1_DIR / "q1_models" if model_dir is None else model_dir
    enc_path = model_dir / "encoder.joblib"
    if not enc_path.exists():
        raise FileNotFoundError(enc_path)
    encoder = joblib.load(enc_path)
    models = []
    for d in DIMS:
        p = model_dir / f"{d}.joblib"
        if not p.exists():
            raise FileNotFoundError(p)
        models.append(joblib.load(p))
    z = np.load(Q1_DIR / "q1_scores.npz", allow_pickle=False)
    feats = list(encoder.fill_.index)
    return Q1Bundle(
        models=models,
        encoder=encoder,
        final_resid=z["final_resid"],
        final_qcuts=z["final_qcuts"],
        dims=list(DIMS),
        chem_cols=feats if chem_cols is None else list(chem_cols),
        feature_names=feats,
    )


def _as_frame(bundle, X231):
    if isinstance(X231, pd.DataFrame):
        return X231[bundle.feature_names]
    X = np.atleast_2d(np.asarray(X231, float))
    return pd.DataFrame(X, columns=bundle.chem_cols)[bundle.feature_names]


def encode(bundle, X231) -> np.ndarray:
    return bundle.encoder.transform(_as_frame(bundle, X231)).to_numpy(np.float64)


def predict_mu(bundle, X231, dims_idx=None) -> np.ndarray:
    Xe = encode(bundle, X231)
    dims_idx = range(len(bundle.models)) if dims_idx is None else dims_idx
    return np.column_stack([bundle.models[j].predict(Xe) for j in dims_idx])


def grade_dist(bundle, mu, dims_idx=None) -> np.ndarray:
    dims_idx = list(range(mu.shape[1])) if dims_idx is None else list(dims_idx)
    return np.stack(
        [predictive_dist(mu[:, i], bundle.final_resid[j]) for i, j in enumerate(dims_idx)], axis=1
    )


def grade_median(bundle, mu, dims_idx=None) -> np.ndarray:
    return dist_median(grade_dist(bundle, mu, dims_idx))


def grade_quant(bundle, mu, dims_idx=None) -> np.ndarray:
    dims_idx = list(range(mu.shape[1])) if dims_idx is None else list(dims_idx)
    return np.column_stack(
        [apply_cuts(mu[:, i], bundle.final_qcuts[j]) for i, j in enumerate(dims_idx)]
    )


def load_main_oof():
    z = np.load(Q1_DIR / "q1_main_oof.npz", allow_pickle=False)
    return {
        "oof_mu": z["oof_mu"],
        "inner_mu": z["inner_mu"],
        "fold_id": z["fold_id"],
        "chosen_params": json.loads(str(z["chosen_params"])),
        "final_params": json.loads(str(z["final_params"])),
    }


def fold_train_test(fold_id, rep, fold):
    f = fold_id[rep]
    return np.where(f != fold)[0], np.where(f == fold)[0]


def fit_fold_bundle(X231: pd.DataFrame, xbar, G, train_idx, outer_rep, outer_fold, oof) -> Q1Bundle:
    tr_ref, _ = fold_train_test(oof["fold_id"], outer_rep, outer_fold)
    assert np.array_equal(np.sort(train_idx), tr_ref), "train_idx 与 Q1 fold_id 不一致"
    tr = tr_ref
    enc = ChemNAEncoder("col_mean").fit(X231.iloc[tr])
    Xe = enc.transform(X231.iloc[tr])
    feats = list(Xe.columns)
    Xe = Xe.to_numpy(np.float64)
    models = [None] * len(DIMS)
    for group, gdims in DIM_GROUPS.items():
        best = oof["chosen_params"][f"{outer_rep}_{outer_fold}_{group}"]
        for d in gdims:
            j = DIMS.index(d)
            models[j] = _fit_dim(Xe, xbar[tr, j], feats, d, best)
    inner = oof["inner_mu"][outer_rep, outer_fold][tr]
    assert np.isfinite(inner).all()
    resid = np.stack([xbar[tr, j] - inner[:, j] for j in range(len(DIMS))])
    qcuts = np.stack([quantile_cuts(inner[:, j], G[tr, j]) for j in range(len(DIMS))])
    return Q1Bundle(models, enc, resid, qcuts, list(DIMS), feats, feats)
