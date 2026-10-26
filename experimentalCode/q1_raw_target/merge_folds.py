import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rawlib import OUT  # noqa: E402
from exp_twostage import N_REP, ND, PANEL, beers
P = {}
for rep in range(N_REP):
    for k in range(5):
        d = np.load(os.path.join(OUT, f"fold_{rep}_{k}.npz"))
        for key in d.files:
            if key == "te":
                continue
            P.setdefault(key, np.zeros((N_REP, len(beers), ND), dtype=d[key].dtype))[rep, d["te"]] = d[key]
np.savez(os.path.join(OUT, "exp_twostage_oof.npz"), panel=PANEL, **P)
print("merged", sorted(P))
