"""Data checks cited in model/q1_methodology.md (raw ratings vs attachments). Reads data/raw only; writes nothing.

Run: python experimentalCode/q1_raw_target/q1_raw_data_checks.py   (split-half seed 0, 20 splits)

  A  ID mapping: raw beer_name = row position of attachments 1/4, not the `beer` column
  B  Attachment 4 reconstructed from raw ratings (per taster x dim standardisation, ddof=1)
  C  Cells not reproduced: zero-variance taster x dim combinations
  D  barnyard vs stale_hops in the raw ratings
  E  Date groups used for CV; nesting of z-tie panels inside date groups
  F  Attendance (taster-offset) share of between-beer variance of the raw mean
  G  Reliability of the panel mean and of the panel grade; grade frequencies
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import additive, load, session_groups  # noqa: E402

r, a4, c1, dims = load()
beers = a4.index.values
grade = lambda v: np.floor(np.clip(v, 0, 6) + 0.5).astype(int)  # noqa: E731
xbar = r.groupby("beer_name")[dims].mean()

print("== A  ID mapping")
by_col = xbar.reindex(a4["beer"].values)
rho_col = [spearmanr(by_col[d].values, a4[d].values)[0] for d in dims]
rho_row = [spearmanr(xbar.loc[beers, d].values, a4[d].values)[0] for d in dims]
print(f"median Spearman raw mean vs A4: by `beer` column {np.median(rho_col):+.3f}; by row {np.median(rho_row):+.3f}")
A, B = a4[dims].values, xbar.loc[beers, dims].values
Ac, Bc = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
C = (Bc / np.linalg.norm(Bc, axis=1, keepdims=True)) @ (Ac / np.linalg.norm(Ac, axis=1, keepdims=True)).T
_, ci = linear_sum_assignment(-C)
print(f"profile assignment equals row order for {(ci == np.arange(250)).sum()}/250 beers")
ct = pd.crosstab(r.groupby("beer_name").beer_category.first().loc[beers].values, a4.tasting_category_fine.values)
print(f"raw style label vs A4 style: each raw label maps to {ct.gt(0).sum(1).max()} A4 labels at most; "
      f"A4 labels split across raw labels: {(ct.gt(0).sum(0) > 1).sum()}")

print("\n== B  reconstruction of attachment 4")
g = r.groupby("taster")[dims]
zr = (r[dims] - g.transform("mean")) / g.transform("std")
zr["b"] = r.beer_name
zb = zr.groupby("b")[dims].mean().loc[beers]
D = (zb - a4[dims]).abs().values
ok = D < 1e-6
print(f"cells within 1e-6: {ok.mean():.4f} ({ok.sum()}/{ok.size}); median |diff| {np.nanmedian(D):.1e}")

print("\n== C  cells not reproduced")
sd = r.groupby("taster")[dims].std()
zero_var = [(t, d) for t in sd.index for d in dims if sd.loc[t, d] == 0]
bad_dims = sorted({d for _, d in zero_var})
print(f"zero-variance taster x dim: {len(zero_var)} in {len(bad_dims)} dims; non-reproduced cells {(~ok).sum()}; "
      f"all inside those dims: {set(np.array(dims)[(~ok).any(0)]) <= set(bad_dims)}")
for d in bad_dims:
    nan = zr[d].isna()
    v = [(a4.loc[b, d] * len(i) - zr[d][list(i)].sum(skipna=True)) / nan[list(i)].sum()
         for b, i in r.groupby("beer_name").groups.items() if nan[list(i)].any()]
    print(f"  {d:16s} implied constant {np.mean(v):+.6f} (sd {np.std(v):.0e}, beers {len(v)})")

print("\n== D  barnyard vs stale_hops")
print(f"identical rows {(r.barnyard == r.stale_hops).mean():.4f} (differ {(r.barnyard != r.stale_hops).sum()}); "
      f"beer-mean Spearman {spearmanr(xbar.barnyard, xbar.stale_hops)[0]:.3f}")

print("\n== E  date groups")
grp = session_groups(r).loc[beers]
purity = [a4.tasting_category_fine[grp[grp == k].index].value_counts(normalize=True).iloc[0] for k in grp.unique()]
n_rat = r.groupby("beer_name").size()
print(f"dates {r.date.nunique()} -> groups {grp.nunique()}; sizes {sorted(grp.value_counts().tolist())}; "
      f"style purity median {np.median(purity):.2f}")
print(f"beers rated on >1 date {(r.groupby('beer_name').date.nunique() > 1).sum()}; ratings per beer median "
      f"{n_rat.median():.0f} (min {n_rat.min()}, max {n_rat.max()}); tasters {r.taster.nunique()}")
ZR = np.round(a4[dims].values, 9)
parent = np.arange(250)


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for i in range(250):
    for j in np.where((ZR[i + 1:] == ZR[i]).sum(1) >= 3)[0] + i + 1:
        parent[find(j)] = find(i)
panel = np.array([find(i) for i in range(250)])
multi = [p for p in np.unique(panel) if (panel == p).sum() > 1]
print(f"z-tie panels (k=3): {len(multi)} multi-beer covering {sum((panel == p).sum() for p in multi)} beers; "
      f"spanning >1 date group: {sum(len(np.unique(grp.values[panel == p])) > 1 for p in multi)}")

print("\n== F  attendance share")
alpha, _ = additive(r, dims)
att = (xbar.loc[beers] - alpha.loc[beers]).var() / xbar.loc[beers].var()
print(f"median {att.median():.3f}, IQR {att.quantile(.25):.3f}-{att.quantile(.75):.3f}; largest: "
      + ", ".join(f"{d} {v:.3f}" for d, v in att.sort_values(ascending=False).head(3).items()))

print("\n== G  reliability and grade frequencies")
rng = np.random.default_rng(0)
rel, kap, ex = [], [], []
for _ in range(20):
    h = np.zeros(len(r), bool)
    for _, idx in r.groupby("beer_name").groups.items():
        idx = rng.permutation(np.asarray(idx))
        h[idx[: len(idx) // 2]] = True
    m1 = r[h].groupby("beer_name")[dims].mean().loc[beers].values
    m2 = r[~h].groupby("beer_name")[dims].mean().loc[beers].values
    rh = np.array([np.corrcoef(m1[:, j], m2[:, j])[0, 1] for j in range(len(dims))])
    rel.append(2 * rh / (1 + rh))
    g1, g2 = grade(m1), grade(m2)
    kap.append(np.mean([cohen_kappa_score(g1[:, j], g2[:, j], weights="quadratic", labels=range(7))
                        for j in range(len(dims))]))
    ex.append((g1 == g2).mean())
rel = pd.Series(np.mean(rel, 0), index=dims)
print(f"Spearman-Brown reliability of the full-panel mean: median {rel.median():.2f}, range {rel.min():.2f}-{rel.max():.2f}")
print("lowest:", ", ".join(f"{d} {v:.2f}" for d, v in rel.sort_values().head(4).items()))
print(f"half-panel grade agreement: exact {np.mean(ex):.3f}, QWK {np.mean(kap):.3f}")
G = grade(xbar.loc[beers].values)
print("grade counts 0..6:", np.bincount(G.ravel(), minlength=7).tolist(),
      f"; dims with >=50% grade 0: {int(((G == 0).mean(0) >= 0.5).sum())}")
