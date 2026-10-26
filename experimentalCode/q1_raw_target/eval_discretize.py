"""Evaluate grade rules against round(raw panel mean)."""

import os
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
t = np.load(os.path.join(OUT, "exp_targets_oof.npz"), allow_pickle=True)
d = np.load(os.path.join(OUT, "exp_discretize_oof.npz"))
xbar = t["xbar"]
G = np.floor(np.clip(xbar, 0, 6) + 0.5).astype(int)
nd = G.shape[1]
print(f"{'rule':10s}{'QWK':>8s}{'exact':>8s}{'pm1':>8s}{'gradeMAE':>9s}  pooled grade freq 0..6")
print(f"{'truth':10s}{'':33s}  {np.round(np.bincount(G.ravel(), minlength=7) / G.size, 3)}")
for k in ["round", "round_w", "qwk_cut", "quant"]:
    g = d[k]
    q = np.mean([np.mean([cohen_kappa_score(G[:, j], g[r, :, j], weights="quadratic", labels=range(7))
                          for j in range(nd)]) for r in range(g.shape[0])])
    ex = (g == G[None]).mean(); pm = (np.abs(g - G[None]) <= 1).mean(); mae = np.abs(g - G[None]).mean()
    print(f"{k:10s}{q:8.4f}{ex:8.4f}{pm:8.4f}{mae:9.4f}  {np.round(np.bincount(g.ravel(), minlength=7) / g.size, 3)}")
