"""Shared loaders for the Q1 raw-target experiments (model/q1_methodology.md).

Inputs: data/raw/{1.csv,4.csv,raw_sensory_data_anonymized.csv}. Outputs of the experiment scripts go to
experimentalCode/results/q1_raw_target/. Project root can be overridden with Q1_ROOT.
"""
import os
from pathlib import Path

import numpy as np, pandas as pd

PROJ = Path(os.environ.get("Q1_ROOT", Path(__file__).resolve().parents[2]))
ROOT = str(PROJ / "data" / "raw") + os.sep
OUT = str(PROJ / "experimentalCode" / "results" / "q1_raw_target")
os.makedirs(OUT, exist_ok=True)

def load():
    r = pd.read_csv(ROOT + "raw_sensory_data_anonymized.csv", encoding="cp1252")
    r = r.rename(columns={"A_ester_fruity": "A_esters_fruity", "F_ester_fruity": "F_esters_fruity"})
    a4 = pd.read_csv(ROOT + "4.csv"); a4.index = np.arange(1, 251)
    c1 = pd.read_csv(ROOT + "1.csv"); c1.index = np.arange(1, 251)
    dims = [c for c in a4.columns if c not in ("beer", "beer_id", "tasting_category_fine")]
    r[dims] = r[dims].astype(float)
    return r, a4, c1, dims

def additive(r, dims, n_iter=300):
    """x_pbd = alpha_bd + beta_pd + e; beta centred (rating-weighted). Returns alpha (beer x dim), beta (taster x dim)."""
    b = r.beer_name.values; p = r.taster.values
    bi = pd.factorize(b)[0]; pi = pd.factorize(p)[0]
    nb, np_ = bi.max() + 1, pi.max() + 1
    cb = np.bincount(bi, minlength=nb); cp = np.bincount(pi, minlength=np_)
    X = r[dims].values
    alpha = np.zeros((nb, len(dims))); beta = np.zeros((np_, len(dims)))
    for _ in range(n_iter):
        for j in range(len(dims)):
            alpha[:, j] = np.bincount(bi, X[:, j] - beta[pi, j], nb) / cb
            beta[:, j] = np.bincount(pi, X[:, j] - alpha[bi, j], np_) / cp
        m = (beta * cp[:, None]).sum(0) / cp.sum(); beta -= m; alpha += m
    return (pd.DataFrame(alpha, index=pd.unique(b), columns=dims),
            pd.DataFrame(beta, index=pd.unique(p), columns=dims))

def session_groups(r):
    """Union dates linked by beers tasted on several dates; group id per beer."""
    parent = {d: d for d in r.date.unique()}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for _, ds in r.groupby("beer_name").date.unique().items():
        for d in ds[1:]: parent[find(d)] = find(ds[0])
    g = r.groupby("beer_name").date.first().map(lambda d: find(d))
    return pd.Series(pd.factorize(g)[0], index=g.index)
