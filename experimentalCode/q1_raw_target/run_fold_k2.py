"""K2 control: ordinal binary decomposition. One outer fold per call: run_fold_k2.py REP K.

For k = 1..6 a HistGradientBoostingClassifier (log loss) estimates P(G >= k) = P(xbar >= k - 0.5);
cumulative probabilities are made non-increasing in k and differenced into p_k, k = 0..6.
Thresholds with fewer than MIN_POS positives or negatives in the training fold use the training share.
Same outer folds as run_fold_1s.py (GroupKFold shuffle random_state=REP); HGB random_state=0.
"""

import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
from rawlib import OUT  # noqa: E402
from exp_targets import X, dims, grp, xbar  # noqa: E402

MIN_POS = 2
G = np.floor(np.clip(xbar.values, 0, 6) + 0.5).astype(int)
rep, k = int(sys.argv[1]), int(sys.argv[2])
tr, te = list(GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=grp))[k]
prob = np.zeros((len(te), len(dims), 7))
for j in range(len(dims)):
    q = np.zeros((len(te), 7))
    q[:, 0] = 1.0
    for t in range(1, 7):
        y = (G[tr, j] >= t).astype(int)
        if min(y.sum(), len(y) - y.sum()) < MIN_POS:
            q[:, t] = y.mean()
            continue
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           min_samples_leaf=20, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y)
        q[:, t] = m.predict_proba(X[te])[:, 1]
    q = np.minimum.accumulate(q, axis=1)
    prob[:, j] = q - np.c_[q[:, 1:], np.zeros(len(te))]
np.savez(os.path.join(OUT, f"k2_{rep}_{k}.npz"), te=te, prob=prob)
print("done", rep, k, flush=True)
