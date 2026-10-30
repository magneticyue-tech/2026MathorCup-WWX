"""Run one outer fold of exp_twostage and save it. Usage: run_fold.py REP K."""

import os
import sys
import warnings

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
warnings.filterwarnings("ignore")
import exp_twostage as e

rep, k = int(sys.argv[1]), int(sys.argv[2])
tr, te = list(GroupKFold(5, shuffle=True, random_state=rep).split(e.X, groups=e.grp))[k]
_, te, out = e.one_fold(rep, tr, te)
np.savez(os.path.join(OUT, f"fold_{rep}_{k}.npz"), te=te, **out)
print("done", rep, k, flush=True)
