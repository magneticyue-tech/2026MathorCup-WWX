"""One-stage fold: save outer test predictions and inner OOF predictions on the training set.

Usage: run_fold_1s.py REP K. Seeds: outer GroupKFold shuffle random_state=REP; inner GroupKFold(4)
shuffle random_state=100+REP; HGB random_state=0.
"""

import os
import sys
import warnings

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
warnings.filterwarnings("ignore")
from exp_targets import X, dims, fit_pred, grp, xbar

XB, ND = xbar.values, len(dims)
rep, k = int(sys.argv[1]), int(sys.argv[2])
tr, te = list(GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=grp))[k]
inner = np.zeros((len(tr), ND))
for itr, ite in GroupKFold(4, shuffle=True, random_state=100 + rep).split(tr, groups=grp[tr]):
    for j in range(ND):
        inner[ite, j] = fit_pred(XB[:, j], tr[itr], tr[ite])
outer = np.column_stack([fit_pred(XB[:, j], tr, te) for j in range(ND)])
np.savez(os.path.join(OUT, f"f1s_{rep}_{k}.npz"), tr=tr, te=te, inner=inner, outer=outer)
print("done", rep, k, flush=True)
