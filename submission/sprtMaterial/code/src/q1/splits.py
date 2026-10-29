import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .paths import SEED

_FEW_MEMBERS = "The least populated class in y has only"


def _sgkf_split(n_splits, seed, y, groups):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_FEW_MEMBERS, category=UserWarning)
        return list(cv.split(np.zeros(len(y)), y, groups))


def outer_splits(groups, style, n_splits=5, n_repeats=5, seed=SEED):
    y = pd.Series(style).to_numpy()
    return [_sgkf_split(n_splits, seed + r, y, np.asarray(groups)) for r in range(n_repeats)]


def inner_splits(train_idx, groups, style, seed, n_splits=4):
    train_idx = np.asarray(train_idx)
    y = pd.Series(style).to_numpy()[train_idx]
    g = np.asarray(groups)[train_idx]
    return [(train_idx[a], train_idx[b]) for a, b in _sgkf_split(n_splits, seed, y, g)]


def leave_one_style_splits(style):
    s = pd.Series(style).to_numpy()
    return [(np.where(s != name)[0], np.where(s == name)[0], name) for name in sorted(set(s))]


def paper_split(style, seed=SEED):
    idx = np.arange(len(style))
    return tuple(train_test_split(idx, test_size=0.3, random_state=seed, stratify=style))


def split_diagnostics(splits, groups, style) -> dict:
    groups = np.asarray(groups)
    n = len(style)
    sizes, n_groups = [], []
    for r, rep in enumerate(splits):
        counts = np.zeros(n, int)
        for k, (tr, te) in enumerate(rep):
            counts[te] += 1
            if set(groups[tr]) & set(groups[te]):
                raise AssertionError(f"重复 {r} 折 {k}：日期组跨训练集与测试集")
            sizes.append(int(len(te)))
            n_groups.append(int(len(np.unique(groups[te]))))
        if not (counts == 1).all():
            raise AssertionError(f"重复 {r}：有啤酒不是恰好测试一次")
    return {"test_sizes": sizes, "test_n_groups": n_groups}
