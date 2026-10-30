import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from .gen_metrics import make_rf
from .paths import SEED

REACH, IMPROVED, KEPT, NOT_IMPROVED = "reach", "improved", "kept", "not_improved"
CATEGORIES = (REACH, IMPROVED, KEPT, NOT_IMPROVED)


class StyleDiscriminator:
    """交叉拟合的风格判别器（方法论 §6.4）。"""

    def __init__(self, x_filled, style, seed=SEED):
        self.style = np.asarray(style)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*least populated class.*")
            self.splits = list(cv.split(x_filled, self.style))
        self.fold = np.empty(len(self.style), int)
        self.models = []
        for f, (tr, te) in enumerate(self.splits):
            self.fold[te] = f
            self.models.append(make_rf().fit(x_filled[tr], self.style[tr]))
        self._x = x_filled

    def fold_of(self, b):
        return int(self.fold[b])

    def train_idx(self, b):
        return self.splits[self.fold[b]][0]

    def p_target(self, b, x225, style):
        rf = self.models[self.fold[b]]
        cls = list(rf.classes_)
        P = rf.predict_proba(np.atleast_2d(x225))
        return P[:, cls.index(style)] if style in cls else np.zeros(len(P))

    def p_real_oof(self, style):
        rows = np.where(self.style == style)[0]
        return np.array([self.p_target(b, self._x[b], style)[0] for b in rows])

    def ref_q25(self, style):
        return float(np.quantile(self.p_real_oof(style), 0.25))


def d2(gen, u, style):
    return gen.mahal2(u, style)


def retention(u_b, mu_s, u_new, mu_t):
    return float(spearmanr(u_b - mu_s, u_new - mu_t)[0])


def change_amount(u_b, u_new, Cinv):
    d = np.atleast_2d(u_new - u_b)
    return np.einsum("ij,jk,ik->i", d, Cinv, d)


def n_changed_cols(x_src, x_new, tol=1e-9):
    return (np.abs(np.atleast_2d(x_new) - x_src) > tol).sum(-1)


def category(p_new, p_src, alpha_star, ref_q25):
    if p_new >= ref_q25:
        return REACH
    if p_new - p_src >= 0.05:
        return IMPROVED
    if alpha_star == 0:
        return KEPT
    return NOT_IMPROVED
