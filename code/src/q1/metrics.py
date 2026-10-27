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

def rps(p, G):
    F = np.cumsum(p, axis=-1)
    O = (np.arange(7) >= G[..., np.newaxis]).astype(float)
    return np.mean(np.sum((F - O)**2, axis=-1))

def confusion_pooled(G, Ghat):
    cm = np.zeros((7, 7), dtype=int)
    for i in range(len(G)):
        for j in range(G.shape[1]):
            cm[G[i, j], Ghat[i, j]] += 1
    return cm

def balanced_accuracy(cm):
    recalls = np.diag(cm) / (cm.sum(axis=1) + 1e-9)
    return np.mean(recalls)

def wins_by_dim(G, Ghat_a, Ghat_b):
    """逐维比较 MAE"""
    wins = 0
    for d in range(G.shape[1]):
        if grade_mae(G[:, d], Ghat_a[:, d]) < grade_mae(G[:, d], Ghat_b[:, d]):
            wins += 1
    return wins

def cluster_bootstrap(a, b, groups, n_boot=2000, seed=0):
    """工单要求的：按日期组整组重抽样算 CI"""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    diffs = []
    for _ in range(n_boot):
        boot_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in boot_groups])
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def log_loss_score(p, G, eps=1e-9):
    """工单要求的 Log Loss"""
    p_true = np.take_along_axis(p, G[..., np.newaxis], axis=2).squeeze()
    return float(-np.mean(np.log(np.clip(p_true, eps, 1-eps))))

def split_report(G, Ghat, zero_heavy_dims, dims):
    """工单要求的：分别报告 ZERO_HEAVY_DIMS 与其余维度的指标"""
    zero_heavy_idx = [i for i, d in enumerate(dims) if d in zero_heavy_dims]
    other_idx = [i for i, d in enumerate(dims) if d not in zero_heavy_dims]
    
    report = {}
    if zero_heavy_idx:
        report["zero_heavy_mae"] = float(grade_mae(G[:, zero_heavy_idx], Ghat[:, zero_heavy_idx]))
    if other_idx:
        report["other_mae"] = float(grade_mae(G[:, other_idx], Ghat[:, other_idx]))
    return report