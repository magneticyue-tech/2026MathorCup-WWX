import numpy as np
from sklearn.linear_model import RidgeCV

from .marg import Marg

ALPHAS = np.logspace(-1, 4, 30)


class CompositeDeriver:
    """由 225 个生成列潜变量岭回归派生 5 个综合指标的潜变量（方法论 §2.1、§5.3）。"""

    def __init__(self, tie_scheme="cond_mean"):
        self.tie_scheme = tie_scheme

    def fit(self, U_gen, X_comp):
        X_comp = np.asarray(X_comp, float)
        self.colmean = np.nanmean(X_comp, 0)
        self.mg = Marg(self.tie_scheme).fit(X_comp)
        V = self.observed_latent(X_comp)
        self.models = [RidgeCV(alphas=ALPHAS).fit(U_gen, V[:, i]) for i in range(V.shape[1])]
        return self

    def predict_latent(self, U_gen):
        U_gen = np.atleast_2d(U_gen)
        return np.column_stack([m.predict(U_gen) for m in self.models])

    def observed_latent(self, X_comp):
        X_comp = np.atleast_2d(np.asarray(X_comp, float))
        return self.mg.transform(np.where(np.isnan(X_comp), self.colmean, X_comp))

    def to_original(self, V):
        return self.mg.inverse(np.atleast_2d(V))
