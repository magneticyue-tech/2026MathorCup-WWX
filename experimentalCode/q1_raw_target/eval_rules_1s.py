"""Compare one-stage discretisation rules against round(raw mean). Bootstrap seed 0, 1000 date-group resamples."""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
warnings.filterwarnings("ignore")
from exp_twostage import cut, fit_cuts, grade, qwk, quantile_cuts

t = np.load(os.path.join(OUT, "exp_targets_oof.npz"), allow_pickle=True)
XB, grp, dims = t["xbar"], t["grp"], list(t["dims"])
G = grade(XB)
NB, ND, NR, K_NN = XB.shape[0], XB.shape[1], 3, 50


def dist_rules(p_te, p_in, y_in, knn):
    """Grade distribution of p + residual; residuals global or from K_NN nearest inner predictions."""
    res = y_in - p_in
    med, mode, prob = [], [], []
    for p in p_te:
        e = res[np.argsort(np.abs(p_in - p))[:K_NN]] if knn else res
        pk = np.bincount(grade(p + e), minlength=7) / len(e)
        med.append(int(np.searchsorted(np.cumsum(pk), 0.5)))
        mode.append(int(pk.argmax()))
        prob.append(pk)
    return np.array(med), np.array(mode), np.array(prob)


RULES = ["const_mode", "round", "med_glob", "mode_glob", "med_knn", "mode_knn", "quant", "qwk_cut"]
P = {r: np.zeros((NR, NB, ND), int) for r in RULES}
PROB = np.zeros((NR, NB, ND, 7))
for rep in range(NR):
    for k in range(5):
        d = np.load(os.path.join(OUT, f"f1s_{rep}_{k}.npz"))
        tr, te, inner, outer = d["tr"], d["te"], d["inner"], d["outer"]
        for j in range(ND):
            y_in, g_in, p_in, p_te = XB[tr, j], G[tr, j], inner[:, j], outer[:, j]
            P["const_mode"][rep, te, j] = np.bincount(g_in, minlength=7).argmax()
            P["round"][rep, te, j] = grade(p_te)
            P["med_glob"][rep, te, j], P["mode_glob"][rep, te, j], _ = dist_rules(p_te, p_in, y_in, False)
            P["med_knn"][rep, te, j], P["mode_knn"][rep, te, j], PROB[rep, te, j] = dist_rules(p_te, p_in, y_in, True)
            P["quant"][rep, te, j] = cut(p_te, quantile_cuts(p_in, g_in))
            P["qwk_cut"][rep, te, j] = cut(p_te, fit_cuts(p_in, g_in))


def mqwk(g, idx=slice(None)):
    return np.nanmean([qwk(G[idx, j], g[r, idx, j]) for r in range(NR) for j in range(ND)])


print(f"{'rule':11s}{'gMAE':>8s}{'exact':>8s}{'pm1':>8s}{'QWK':>8s}{'5-6 %':>8s}  grade freq 0..6")
print(f"{'truth':11s}{'':32s}{100 * (G >= 5).mean():8.2f}  {np.round(np.bincount(G.ravel(), minlength=7) / G.size, 3)}")
for r in RULES:
    g = P[r]
    print(f"{r:11s}{abs(g - G).mean():8.4f}{(g == G).mean():8.4f}{(abs(g - G) <= 1).mean():8.4f}{mqwk(g):8.4f}"
          f"{100 * (g >= 5).mean():8.2f}  {np.round(np.bincount(g.ravel(), minlength=7) / g.size, 3)}")

# calibration of knn predictive distribution
Gr = np.broadcast_to(G, PROB.shape[:3])
pt = np.take_along_axis(PROB, Gr[..., None], -1)[..., 0]
cdf = np.cumsum(PROB, -1)
lo = (cdf < 0.05).sum(-1)
hi = (cdf < 0.95).sum(-1)
print(f"\nknn predictive dist: mean P(true grade) {pt.mean():.3f}; 90% grade interval coverage {((Gr >= lo) & (Gr <= hi)).mean():.3f}")

rng = np.random.default_rng(0)
ug = np.unique(grp)
mem = [np.where(grp == g)[0] for g in ug]


def boot(a, b, n=1000):
    ga, gb = P[a], P[b]
    pts = ((abs(ga - G) - abs(gb - G)).mean(), (ga == G).mean() - (gb == G).mean(), mqwk(ga) - mqwk(gb))
    bs = []
    for _ in range(n):
        idx = np.concatenate([mem[i] for i in rng.integers(0, len(ug), len(ug))])
        bs.append(((abs(ga[:, idx] - G[idx]) - abs(gb[:, idx] - G[idx])).mean(),
                   (ga[:, idx] == G[idx]).mean() - (gb[:, idx] == G[idx]).mean(),
                   mqwk(ga, idx) - mqwk(gb, idx)))
    lo_, hi_ = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f"{a:9s}-{b:9s} dMAE {pts[0]:+.4f} [{lo_[0]:+.4f},{hi_[0]:+.4f}]  dExact {pts[1]:+.4f} "
          f"[{lo_[1]:+.4f},{hi_[1]:+.4f}]  dQWK {pts[2]:+.4f} [{lo_[2]:+.4f},{hi_[2]:+.4f}]")


print()
for a, b in [("med_glob", "round"), ("med_knn", "med_glob"), ("mode_knn", "round"), ("quant", "round"),
             ("quant", "qwk_cut"), ("round", "const_mode")]:
    boot(a, b)

# where do rules differ: zero-inflated dims vs others
zi = (G == 0).mean(0) >= 0.5
for r in ["const_mode", "round", "med_knn", "quant"]:
    g = P[r]
    print(f"{r:11s} zero-heavy dims ({zi.sum()}) exact {(g == G)[:, :, zi].mean():.3f} gMAE {abs(g - G)[:, :, zi].mean():.3f} | "
          f"others ({(~zi).sum()}) exact {(g == G)[:, :, ~zi].mean():.3f} gMAE {abs(g - G)[:, :, ~zi].mean():.3f}")
np.savez(os.path.join(OUT, "rules_1s.npz"), prob=PROB, **P)
