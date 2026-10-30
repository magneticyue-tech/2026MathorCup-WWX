"""How close are attachment-only constructed references to the true raw means? Split-half seed 0."""

import os
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import additive, load

r, a4, c1, dims = load()
beers = a4.index.values
xbar = r.groupby("beer_name")[dims].mean().loc[beers]
alpha, _ = additive(r, dims)
alpha = alpha.loc[beers]
grade = lambda v: np.floor(np.clip(v, 0, 6) + 0.5).astype(int)
qwk = lambda a, b: cohen_kappa_score(a, b, weights="quadratic", labels=range(7))


def summarise(name, est, truth):
    G, g = grade(truth), grade(est)
    mae = np.mean([np.abs(est[:, j] - truth[:, j]).mean() for j in range(len(dims))])
    k = np.mean([qwk(G[:, j], g[:, j]) for j in range(len(dims))])
    ex = (G == g).mean()
    print(f"{name:52s} MAE {mae:.3f}  QWK {k:.3f}  exact {ex:.3f}")


Z = a4[dims].values
lin = 6 * (Z - Z.min(0)) / (Z.max(0) - Z.min(0))
summarise("linear endpoint ref L on observed z  vs round(xbar)", lin, xbar.values)
B = np.array([np.polyfit(Z[:, j], xbar.values[:, j], 1) for j in range(len(dims))])
summarise("OLS z->xbar (in-sample, best linear)  vs round(xbar)", Z * B[:, 0] + B[:, 1], xbar.values)
summarise("additive alpha  vs round(xbar)", alpha.values, xbar.values)

# split-half reliability of the panel grade: noise ceiling
rng = np.random.default_rng(0)
ks, exs, rel = [], [], []
for _ in range(20):
    h = np.zeros(len(r), bool)
    for _, idx in r.groupby("beer_name").groups.items():
        idx = rng.permutation(np.asarray(idx)); h[idx[: len(idx) // 2]] = True
    m1 = r[h].groupby("beer_name")[dims].mean().loc[beers].values
    m2 = r[~h].groupby("beer_name")[dims].mean().loc[beers].values
    g1, g2 = grade(m1), grade(m2)
    ks.append(np.mean([qwk(g1[:, j], g2[:, j]) for j in range(len(dims))]))
    exs.append((g1 == g2).mean())
    rh = np.array([np.corrcoef(m1[:, j], m2[:, j])[0, 1] for j in range(len(dims))])
    rel.append(2 * rh / (1 + rh))
rel = np.mean(rel, 0)
print(f"split-half panels: grade QWK {np.mean(ks):.3f}, exact {np.mean(exs):.3f} (half-size panels, lower bound)")
print(f"Spearman-Brown reliability of full-panel mean: median {np.median(rel):.3f}, range {rel.min():.3f}-{rel.max():.3f}")
