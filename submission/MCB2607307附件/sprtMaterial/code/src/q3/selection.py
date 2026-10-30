import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import GRID_K, GRID_KAPPA, N_FOLDS, N_REPEATS, SMALL_STYLES
from .generator import fit_grid
from .paths import SEED

_FEW = "least populated class"

CONFIGS = [(kind, k, float(kap)) for kind in ("fa", "famod") for k in GRID_K for kap in GRID_KAPPA]
CONFIGS += [(kind, None, float(kap)) for kind in ("diag", "lw") for kap in GRID_KAPPA]
SELECTABLE = CONFIGS[: 2 * len(GRID_K) * len(GRID_KAPPA)]


def _skf(n_splits, seed, y):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{_FEW}.*", category=UserWarning)
        return list(cv.split(np.zeros(len(y)), y))


def outer_splits(style, n_repeats=N_REPEATS, seed=SEED):
    style = np.asarray(style)
    out = []
    for rep in range(n_repeats):
        for fold, (tr, te) in enumerate(_skf(N_FOLDS, seed + rep, style)):
            out.append((rep, fold, tr, te))
    return out


def inner_splits(train_idx, style, rep, seed=SEED):
    train_idx = np.asarray(train_idx)
    y = np.asarray(style)[train_idx]
    return [(train_idx[a], train_idx[b]) for a, b in _skf(N_FOLDS, seed + 100 + rep, y)]


def inner_scores(X, style, train_idx, rep, module_labels=None, styles=None, configs=CONFIGS):
    style = np.asarray(style)
    styles = sorted(set(style)) if styles is None else styles
    tot = np.zeros(len(configs))
    tot_small = np.zeros(len(configs))
    n_used, n_small, n_excl = 0, 0, 0
    for tr, va in inner_splits(train_idx, style, rep):
        gens = fit_grid(X[tr], style[tr], styles, configs, module_labels)
        present = np.isin(style[va], np.unique(style[tr]))
        n_excl += int((~present).sum())
        va = va[present]
        small = np.isin(style[va], SMALL_STYLES)
        n_used += len(va)
        n_small += int(small.sum())
        for i, cfg in enumerate(configs):
            sc = gens[cfg].score(X[va], style[va])
            assert np.isfinite(sc).all()
            tot[i] += sc.sum()
            tot_small[i] += sc[small].sum()
    small_scores = tot_small / n_small if n_small else np.full(len(configs), np.nan)
    return tot / n_used, small_scores, n_excl


def select_config(scores, configs=CONFIGS) -> dict:
    best, best_key = None, None
    for i, cfg in enumerate(configs):
        if cfg not in SELECTABLE:
            continue
        kind, k, kap = cfg
        key = (-scores[i], k, kap, 0 if kind == "fa" else 1)
        if best_key is None or key < best_key:
            best, best_key = cfg, key
    kind, k, kap = best
    return {"kind": kind, "k": int(k), "kappa": float(kap)}
