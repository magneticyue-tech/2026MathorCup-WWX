import copy

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis

from .marg import Marg
from .modules import modules_from_data
from .paths import SEED

KINDS = ("famod", "fa", "lw", "diag")
STYLE_SCALE_NU = 10.0


def _within_resid(U, s):
    R = U.copy()
    for st in np.unique(s):
        m = s == st
        R[m] -= U[m].mean(0)
    return R


def _style_means(U, s, styles, kappa, global_only):
    g = U.mean(0)
    mu = {}
    for st in styles:
        m = s == st
        n = int(m.sum())
        if global_only:
            mu[st] = g
        elif n == 0:
            mu[st] = None
        else:
            mu[st] = (n * U[m].mean(0) + kappa * g) / (n + kappa)
    return g, mu


def _famod_cov(R, comps, labels):
    LL = comps.T @ comps
    Res = R.T @ R / len(R) - LL
    B = np.diag(np.diag(Res).clip(1e-3))
    for m in np.unique(labels):
        idx = np.where(labels == m)[0]
        if len(idx) > 1:
            B[np.ix_(idx, idx)] = Res[np.ix_(idx, idx)]
    w, V = np.linalg.eigh(B)
    return LL + (V * w.clip(1e-3)) @ V.T


def style_scale_factors(U, s, styles, nu=STYLE_SCALE_NU):
    """敏感性 G-S4：风格专属对角方差比，以 nu 个伪自由度向 1 收缩（方法论 §6.7）。"""
    R = _within_resid(U, s)
    S = len(np.unique(s))
    vW = (R**2).sum(0) / (len(U) - S)
    D = {}
    for st in styles:
        m = s == st
        n = int(m.sum())
        if n <= 1:
            D[st] = np.ones(U.shape[1])
            continue
        vs = U[m].var(0, ddof=1)
        d = ((n - 1) * vs + nu * vW) / ((n - 1 + nu) * np.where(vW > 0, vW, 1.0))
        D[st] = np.where(vW > 0, d, 1.0)
    return D


class _Core:
    """同一训练集上各候选设定共用的量：秩变换、潜变量、残差与模块。"""

    def __init__(self, X_tr, s_tr, tie_scheme, global_only, module_labels, na_fill):
        X_tr = np.asarray(X_tr, float)
        self.mg = Marg(tie_scheme).fit(X_tr)
        if na_fill == "mean":
            self.fill = np.nanmean(X_tr, 0)
        elif na_fill == "min":
            self.fill = np.nanmin(X_tr, 0)
        else:
            raise ValueError(na_fill)
        self.Xf = np.where(np.isnan(X_tr), self.fill, X_tr)
        self.U = self.mg.transform(self.Xf)
        self.s = np.asarray(s_tr)
        n = len(self.U)
        if global_only:
            R, dof = self.U - self.U.mean(0), n - 1
        else:
            R, dof = _within_resid(self.U, self.s), n - len(np.unique(self.s))
        self.R = R * np.sqrt(n / dof)
        self._module_labels = module_labels
        self._fa = {}

    @property
    def modules(self):
        if self._module_labels is None:
            self._module_labels = modules_from_data(self.Xf)
        return self._module_labels

    def fa(self, k, seed):
        if k not in self._fa:
            self._fa[k] = FactorAnalysis(n_components=k, random_state=seed).fit(self.R)
        return self._fa[k]

    def cov(self, kind, k, seed):
        if kind == "diag":
            return np.diag(self.R.var(0))
        if kind == "lw":
            return LedoitWolf(assume_centered=True).fit(self.R).covariance_
        fa = self.fa(k, seed)
        if kind == "fa":
            return fa.get_covariance()
        if kind == "famod":
            return _famod_cov(self.R, fa.components_, self.modules)
        raise ValueError(kind)


class StyleGen:
    """风格条件高斯 copula（方法论 §3.1）。"""

    def __init__(
        self,
        kind="famod",
        k=15,
        kappa=2.0,
        global_only=False,
        tie_scheme="cond_mean",
        seed=SEED,
        style_scale=False,
        na_fill="mean",
    ):
        if kind not in KINDS:
            raise ValueError(kind)
        if kind in ("fa", "famod") and k is None:
            raise ValueError("fa 与 famod 须给 k")
        self.kind, self.k, self.kappa = kind, k, float(kappa)
        self.global_only, self.tie_scheme, self.seed = global_only, tie_scheme, seed
        self.style_scale, self.na_fill = style_scale, na_fill

    def config(self) -> dict:
        return {"kind": self.kind, "k": self.k, "kappa": self.kappa}

    def _from_core(self, core, styles):
        self.styles = list(styles)
        self.mg, self.U, self.s, self.colmean = core.mg, core.U, core.s, core.fill
        self.module_labels = core.modules if self.kind == "famod" else core._module_labels
        self.C = core.cov(self.kind, self.k, self.seed)
        self.L = np.linalg.cholesky(self.C + 1e-9 * np.eye(len(self.C)))
        self.g, self.mu = _style_means(self.U, self.s, self.styles, self.kappa, self.global_only)
        self._finish()
        return self

    def _finish(self):
        self.Cs, self.Ls = {}, {}
        if self.style_scale:
            D = style_scale_factors(self.U, self.s, self.styles)
            for st, d in D.items():
                sd = np.sqrt(d)
                Cs = self.C * sd[:, None] * sd[None, :]
                self.Cs[st] = Cs
                self.Ls[st] = np.linalg.cholesky(Cs + 1e-9 * np.eye(len(Cs)))
        self._logdet = {}

    def fit(self, X_tr, s_tr, styles, module_labels=None):
        core = _Core(X_tr, s_tr, self.tie_scheme, self.global_only, module_labels, self.na_fill)
        return self._from_core(core, styles)

    def with_kappa(self, kappa):
        new = copy.copy(self)
        new.kappa = float(kappa)
        new.g, new.mu = _style_means(self.U, self.s, self.styles, new.kappa, self.global_only)
        return new

    def has_style(self, style) -> bool:
        return self.mu.get(style) is not None

    def chol(self, style):
        return self.Ls.get(style, self.L)

    def cov_for(self, style):
        return self.Cs.get(style, self.C)

    def latent(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return self.mg.transform(np.where(np.isnan(X), self.colmean, X))

    def log_density(self, u, style):
        u = np.atleast_2d(u)
        if not self.has_style(style):
            return np.full(len(u), np.nan)
        L = self.chol(style)
        Z = np.linalg.solve(L, (u - self.mu[style]).T)
        return (
            -0.5 * (Z**2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * L.shape[0] * np.log(2 * np.pi)
        )

    def score_latent(self, U, s):
        out = np.full(len(U), np.nan)
        s = np.asarray(s)
        for st in np.unique(s):
            m = s == st
            if self.has_style(st):
                out[m] = self.log_density(U[m], st)
        return out

    def score(self, X, s):
        return self.score_latent(self.latent(X), s)

    def sample_latent(self, style, n, rng):
        if not self.has_style(style):
            raise ValueError(f"训练集中没有风格 {style}")
        L = self.chol(style)
        return self.mu[style] + rng.standard_normal((n, L.shape[0])) @ L.T

    def sample(self, style, n, rng):
        return self.mg.inverse(self.sample_latent(style, n, rng))

    def mahal2(self, u, style):
        u = np.atleast_2d(u)
        Z = np.linalg.solve(self.chol(style), (u - self.mu[style]).T)
        return (Z**2).sum(0)


def fit_grid(
    X_tr,
    s_tr,
    styles,
    configs,
    module_labels=None,
    tie_scheme="cond_mean",
    seed=SEED,
    global_only=False,
) -> dict:
    core = _Core(X_tr, s_tr, tie_scheme, global_only, module_labels, "mean")
    out, base = {}, {}
    for kind, k, kappa in configs:
        key = (kind, k)
        if key not in base:
            g = StyleGen(kind, k, 0.0, global_only, tie_scheme, seed)
            base[key] = g._from_core(core, styles)
        out[(kind, k, kappa)] = base[key].with_kappa(kappa)
    return out
