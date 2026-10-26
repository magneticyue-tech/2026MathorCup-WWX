"""Score exp_twostage_oof.npz against round(raw panel mean). Bootstrap seed 0, 2000 resamples of date groups."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
from exp_twostage import grade, qwk

t = np.load(os.path.join(OUT, "exp_targets_oof.npz"), allow_pickle=True)
f = np.load(os.path.join(OUT, "exp_twostage_oof.npz"))
xbar, grp, dims = t["xbar"], t["grp"], list(t["dims"])
G = grade(xbar)
NR, NB, ND = f["1S_round"].shape
LAB = {
    "2SA": "two-stage, attachment-only map (methodology)",
    "2SR": "two-stage, z->0-6 OLS calibrated on raw",
    "1S": "one-stage, chem -> raw mean",
}
RULES = ["round", "quant", "qwk"]


def mean_qwk(g, ref, mask=None):
    out = []
    for r in range(NR):
        v = []
        for j in range(ND):
            m = slice(None) if mask is None else mask[r, :, j]
            v.append(qwk(ref[r, m, j] if ref.ndim == 3 else ref[m, j], g[r, m, j]))
        out.append(np.nanmean(v))
    return np.mean(out)


print("== vs truth round(xbar)")
print(f"{'variant':12s}{'QWK':>8s}{'exact':>8s}{'pm1':>8s}{'gMAE':>8s}{'5-6 %':>8s}{'QWK vs R':>10s}")
for s in LAB:
    for rule in RULES:
        g = f[f"{s}_{rule}"]
        print(f"{s + '_' + rule:12s}{mean_qwk(g, G):8.4f}{(g == G).mean():8.4f}{(abs(g - G) <= 1).mean():8.4f}"
              f"{abs(g - G).mean():8.4f}{100 * (g >= 5).mean():8.2f}{mean_qwk(g, f['R_te']):10.4f}")
print(f"truth 5-6 %: {100 * (G >= 5).mean():.2f}")
cm = {s: abs(f[f'{s}_cont'] - xbar[None]).mean() for s in ("2SR", "1S")}
print("continuous MAE on raw scale:", {k: round(v, 4) for k, v in cm.items()})
R = f["R_te"]
print("constructed reference R vs truth: QWK %.4f exact %.4f" % (mean_qwk(R, G), (R == G[None]).mean()))

# what the methodology's own selection rule would pick (QWK vs R, round vs qwk-cut on 2SA)
pick = max(["2SA_round", "2SA_qwk"], key=lambda k: mean_qwk(f[k], R))
print("methodology rule picks:", pick)

# split by local-reference availability
use = f["use_te"].astype(bool)
print("\n== by cell type (exact / gMAE vs truth); local share %.3f" % use.mean())
for k in [pick, "2SR_quant", "1S_round", "1S_quant"]:
    g = f[k]
    print(f"{k:12s} local {(g == G)[use].mean():.4f}/{abs(g - G)[use].mean():.4f}   "
          f"other {(g == G)[~use].mean():.4f}/{abs(g - G)[~use].mean():.4f}")

# paired cluster bootstrap: QWK difference recomputed on resampled beers
rng = np.random.default_rng(0)
ug = np.unique(grp)
members = [np.where(grp == g)[0] for g in ug]


def boot(a, b, n=2000):
    ga, gb = f[a], f[b]
    point = mean_qwk(ga, G) - mean_qwk(gb, G)
    pe = (ga == G).mean() - (gb == G).mean()
    bs_q, bs_e = [], []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
        qa = np.nanmean([qwk(G[idx, j], ga[r, idx, j]) for r in range(NR) for j in range(ND)])
        qb = np.nanmean([qwk(G[idx, j], gb[r, idx, j]) for r in range(NR) for j in range(ND)])
        bs_q.append(qa - qb)
        bs_e.append((ga[:, idx] == G[idx]).mean() - (gb[:, idx] == G[idx]).mean())
    ci = lambda v: f"[{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]"
    print(f"{a:10s} - {b:10s}  dQWK {point:+.4f} {ci(bs_q)}   dExact {pe:+.4f} {ci(bs_e)}")


print("\n== paired differences (cluster bootstrap over 25 date groups)")
for a, b in [("1S_quant", pick), ("1S_round", pick), ("1S_quant", "2SA_quant"),
             ("1S_quant", "2SR_quant"), ("1S_round", "2SR_round")]:
    boot(a, b, n=1000)

# per-dim wins
q = lambda k: np.array([np.mean([qwk(G[:, j], f[k][r, :, j]) for r in range(NR)]) for j in range(ND)])
d1 = q("1S_quant") - q(pick)
d2 = q("1S_quant") - q("2SR_quant")
print(f"\ndims 1S_quant > {pick}: {(d1 > 0).sum()}/50;  1S_quant > 2SR_quant: {(d2 > 0).sum()}/50")
order = np.argsort(d2)
print("2SR_quant best relative dims:", [(dims[i], round(d2[i], 3)) for i in order[:5]])


# reference error of the attachment-only mixed reference R, split by cell type
print("\n== constructed reference R vs truth, by cell type")
Rt, Ut = f["R_te"], f["use_te"].astype(bool)
Gb = np.broadcast_to(G, Rt.shape)
for name, m in [("all", np.ones_like(Ut)), ("A (local zero)", Ut), ("L (linear)", ~Ut)]:
    d = Rt[m] - Gb[m]
    print(f"{name:16s} cells/rep {m.sum() // NR:5d} exact {np.mean(d == 0):.3f} pm1 {np.mean(abs(d) <= 1):.3f} "
          f"gMAE {np.mean(abs(d)):.3f} bias {d.mean():+.3f}")
