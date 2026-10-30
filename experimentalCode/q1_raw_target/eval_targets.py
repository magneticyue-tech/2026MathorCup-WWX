"""Evaluate exp_targets_oof.npz on the raw 0-6 scale. Bootstrap seed 0, 2000 cluster resamples."""

import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
f = np.load(os.path.join(OUT, "exp_targets_oof.npz"), allow_pickle=True)
dims, xbar, grp = list(f["dims"]), f["xbar"], f["grp"]
methods = ["b0", "b1", "raw", "add", "z_rep", "z_ols"]
grade = lambda v: np.floor(np.clip(v, 0, 6) + 0.5).astype(int)
G = grade(xbar)


def metrics(p):
    out = {k: [] for k in ["MAE", "RMSE/SD", "R2", "rho", "QWK", "exact", "pm1", "bias"]}
    for rep in range(p.shape[0]):
        acc = {k: [] for k in out}
        for j in range(len(dims)):
            y, q = xbar[:, j], p[rep, :, j]
            e = q - y
            acc["MAE"].append(np.abs(e).mean())
            acc["RMSE/SD"].append(np.sqrt((e**2).mean()) / y.std())
            acc["R2"].append(1 - (e**2).sum() / ((y - y.mean()) ** 2).sum())
            acc["rho"].append(spearmanr(q, y)[0] if q.std() > 0 else np.nan)
            g = grade(q)
            acc["QWK"].append(cohen_kappa_score(G[:, j], g, weights="quadratic", labels=range(7)))
            acc["exact"].append((g == G[:, j]).mean())
            acc["pm1"].append((np.abs(g - G[:, j]) <= 1).mean())
            acc["bias"].append(e.mean())
        for k in out:
            out[k].append(np.nanmean(acc[k]))
    return {k: np.mean(v) for k, v in out.items()}


print(f"{'method':8s}" + "".join(f"{k:>9s}" for k in ["MAE", "RMSE/SD", "R2", "rho", "QWK", "exact", "pm1", "bias"]))
for m in methods:
    s = metrics(f[f"pred_{m}"])
    print(f"{m:8s}" + "".join(f"{v:9.4f}" for v in s.values()))

# paired cluster bootstrap on per-beer MAE averaged over dims and reps
rng = np.random.default_rng(0)
ug = np.unique(grp)
loss = {m: np.abs(f[f"pred_{m}"] - xbar[None]).mean(axis=(0, 2)) for m in methods}
for a, b in [("add", "raw"), ("z_rep", "raw"), ("z_ols", "raw"), ("raw", "b1"), ("add", "b1")]:
    d = loss[a] - loss[b]
    bs = []
    for _ in range(2000):
        gs = rng.choice(ug, len(ug))
        idx = np.concatenate([np.where(grp == g)[0] for g in gs])
        bs.append(d[idx].mean())
    print(f"MAE[{a}] - MAE[{b}] = {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")

# per-dim winners
mae = {m: np.abs(f[f"pred_{m}"] - xbar[None]).mean(axis=(0, 1)) for m in methods}
print("dims where add beats raw:", int((mae["add"] < mae["raw"]).sum()), "/ 50;  z_rep beats raw:", int((mae["z_rep"] < mae["raw"]).sum()))
print("dims where raw beats b1:", int((mae["raw"] < mae["b1"]).sum()))
print("grade distribution of round(xbar) pooled:", np.bincount(G.ravel(), minlength=7) / G.size)
