"""Main model (regression + residual predictive distribution) vs K2 (ordinal binary decomposition).

Truth: G = round(raw panel mean). Same outer folds (GroupKFold shuffle random_state=rep, rep 0..2).
Log loss uses p' = (1 - EPS) p + EPS / 7. Bootstrap seed 0, 1000 resamples of the 25 date groups.
Writes OUT/eval_k2.npz (pooled confusion matrices, per-rule grades).
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
from rawlib import OUT  # noqa: E402
from exp_twostage import grade, qwk  # noqa: E402

EPS, NR = 0.01, 3
t = np.load(os.path.join(OUT, "exp_targets_oof.npz"), allow_pickle=True)
XB, grp = t["xbar"], t["grp"]
G = grade(XB)
NB, ND = G.shape
rules = np.load(os.path.join(OUT, "rules_1s.npz"))

prob = {"reg": np.zeros((NR, NB, ND, 7)), "k2": np.zeros((NR, NB, ND, 7)), "clim": np.zeros((NR, NB, ND, 7))}
for rep in range(NR):
    for k in range(5):
        d = np.load(os.path.join(OUT, f"f1s_{rep}_{k}.npz"))
        tr, te, inner, outer = d["tr"], d["te"], d["inner"], d["outer"]
        for j in range(ND):
            gk = grade(outer[:, j][:, None] + (XB[tr, j] - inner[:, j])[None, :])
            prob["reg"][rep, te, j] = np.stack([(gk == c).mean(1) for c in range(7)], 1)
            prob["clim"][rep, te, j] = np.bincount(G[tr, j], minlength=7) / len(tr)
        e = np.load(os.path.join(OUT, f"k2_{rep}_{k}.npz"))
        prob["k2"][rep, e["te"]] = e["prob"]


def median(p):
    return (np.cumsum(p, -1) < 0.5).sum(-1)


grades = {
    "const_mode": rules["const_mode"],
    "reg_round": rules["round"],
    "reg_med (main)": median(prob["reg"]),
    "reg_quant": rules["quant"],
    "K2_med": median(prob["k2"]),
    "K2_mode": prob["k2"].argmax(-1),
}


def confusion(g):
    return np.bincount((7 * np.broadcast_to(G, g.shape) + g).ravel(), minlength=49).reshape(7, 7) / NR


def bal_acc(cm):
    present = cm.sum(1) > 0
    return np.mean(np.diag(cm)[present] / cm.sum(1)[present])


def mqwk(g, idx=slice(None)):
    return np.nanmean([qwk(G[idx, j], g[r, idx, j]) for r in range(NR) for j in range(ND)])


def rps(p, idx=slice(None)):
    F = np.cumsum(p[:, idx], -1)[..., :6]
    O = (G[idx][None, ..., None] <= np.arange(6)).astype(float)
    return ((F - O) ** 2).sum(-1).mean()


def logloss(p, idx=slice(None)):
    pp = (1 - EPS) * p[:, idx] + EPS / 7
    return -np.log(np.take_along_axis(pp, np.broadcast_to(G[idx], pp.shape[:3])[..., None], -1)).mean()


print(f"{'rule':16s}{'gMAE':>8s}{'exact':>8s}{'pm1':>8s}{'QWK':>8s}{'balAcc':>8s}{'5-6 %':>8s}")
CM = {}
for name, g in grades.items():
    CM[name] = confusion(g)
    print(f"{name:16s}{np.abs(g - G).mean():8.4f}{(g == G).mean():8.4f}{(np.abs(g - G) <= 1).mean():8.4f}"
          f"{mqwk(g):8.4f}{bal_acc(CM[name]):8.4f}{100 * (g >= 5).mean():8.2f}")
print("\npredictive distributions:")
for name in ["clim", "reg", "k2"]:
    print(f"  {name:5s} RPS {rps(prob[name]):.4f}  logloss {logloss(prob[name]):.4f}")

np.set_printoptions(suppress=True, linewidth=150)
for name in ["reg_med (main)", "K2_med", "reg_round"]:
    cm = CM[name]
    print(f"\npooled confusion matrix, {name} (rows truth 0..6, cols predicted 0..6; mean over reps)")
    print(np.round(cm).astype(int))
    print("recall by class:", np.round(np.diag(cm) / np.maximum(cm.sum(1), 1e-9), 3))

rng = np.random.default_rng(0)
ug = np.unique(grp)
mem = [np.where(grp == g)[0] for g in ug]
a, b = grades["K2_med"], grades["reg_med (main)"]
pt = (np.abs(a - G).mean() - np.abs(b - G).mean(), mqwk(a) - mqwk(b), rps(prob["k2"]) - rps(prob["reg"]),
      logloss(prob["k2"]) - logloss(prob["reg"]))
bs = []
for _ in range(1000):
    idx = np.concatenate([mem[i] for i in rng.integers(0, len(ug), len(ug))])
    bs.append((np.abs(a[:, idx] - G[idx]).mean() - np.abs(b[:, idx] - G[idx]).mean(), mqwk(a, idx) - mqwk(b, idx),
               rps(prob["k2"], idx) - rps(prob["reg"], idx), logloss(prob["k2"], idx) - logloss(prob["reg"], idx)))
lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
print("\nK2_med - reg_med (main), 95% CI:")
for i, n in enumerate(["gMAE", "QWK", "RPS", "logloss"]):
    print(f"  d{n:8s} {pt[i]:+.4f} [{lo[i]:+.4f}, {hi[i]:+.4f}]")
mae_d = lambda g: np.abs(g - G[None]).mean(axis=(0, 1))
print("dims where K2_med has lower gMAE than main:", int((mae_d(a) < mae_d(b)).sum()), "/ 50; equal:",
      int((mae_d(a) == mae_d(b)).sum()))
np.savez(os.path.join(OUT, "eval_k2.npz"), names=np.array(list(CM)), cm=np.stack(list(CM.values())),
         **{f"grade_{k.split()[0]}": v for k, v in grades.items()})
