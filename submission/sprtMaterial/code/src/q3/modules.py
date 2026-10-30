import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .config import TAU
from .paths import Q2_NODES


def modules_from_corr(abs_corr, tau=TAU) -> np.ndarray:
    D = np.clip(1 - np.asarray(abs_corr, float), 0, None)
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    return fcluster(linkage(squareform(D, checks=False), method="average"), 1 - tau, "distance")


def spearman_abs(X_filled) -> np.ndarray:
    return pd.DataFrame(X_filled).corr("spearman").abs().fillna(0).values


def modules_from_data(X_filled, tau=TAU) -> np.ndarray:
    return modules_from_corr(spearman_abs(X_filled), tau)


def relabel(labels) -> np.ndarray:
    labels = np.asarray(labels)
    _, first = np.unique(labels, return_index=True)
    order = labels[np.sort(first)]
    mapping = {v: i for i, v in enumerate(order)}
    return np.array([mapping[v] for v in labels], dtype=int)


def modules_from_q2(gen_cols, path=Q2_NODES) -> np.ndarray:
    nodes = pd.read_csv(path, usecols=["node", "module_id"]).set_index("node")["module_id"]
    missing = [c for c in gen_cols if c not in nodes.index]
    assert not missing, f"Q2 模块表缺少列: {missing}"
    lab = nodes.loc[list(gen_cols)].to_numpy(float)
    assert not np.isnan(lab).any(), "Q2 模块号含缺失"
    return relabel(lab.astype(int))


def n_multi(labels) -> int:
    _, cnt = np.unique(labels, return_counts=True)
    return int((cnt >= 2).sum())
