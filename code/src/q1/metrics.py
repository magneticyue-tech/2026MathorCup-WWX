import numpy as np
from sklearn.metrics import cohen_kappa_score

def grade_mae(G, Ghat):
    return np.mean(np.abs(G - Ghat))

def exact_match(G, Ghat):
    return np.mean(G == Ghat)

def within1(G, Ghat):
    return np.mean(np.abs(G - Ghat) <= 1)

def qwk_by_dim(G, Ghat):
    scores = []
    for d in range(G.shape[1]):
        try:
            scores.append(cohen_kappa_score(G[:, d], Ghat[:, d], weights="quadratic", labels=range(7)))
        except:
            scores.append(np.nan)
    return np.array(scores)