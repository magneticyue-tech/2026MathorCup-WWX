from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SourceRepr:
    u: np.ndarray
    x_src: np.ndarray
    comp_obs_latent: np.ndarray
    comp_src: np.ndarray
    internal_std: float


class InputBuilder:
    """231 列 Q1 输入的组装（方法论 §5.2、§5.3）。"""

    def __init__(self, gen, comp, chem):
        self.gen, self.comp = gen, comp
        self.chem_cols = list(chem.chem_cols)
        self.gen_cols = list(chem.gen_cols)
        self.comp_cols = list(chem.composite_cols)
        self.istd_col = chem.internal_std_col
        self.Xc = chem.X[self.comp_cols].to_numpy(float)
        self.istd = chem.X[self.istd_col].to_numpy(float)
        self._pos_gen = [self.chem_cols.index(c) for c in self.gen_cols]
        self._pos_comp = [self.chem_cols.index(c) for c in self.comp_cols]
        self._pos_istd = self.chem_cols.index(self.istd_col)

    def build_source(self, b) -> SourceRepr:
        u = self.gen.U[b]
        v = self.comp.observed_latent(self.Xc[b])[0]
        return SourceRepr(
            u=u,
            x_src=self.gen.mg.inverse(u[None])[0],
            comp_obs_latent=v,
            comp_src=self.comp.to_original(v[None])[0],
            internal_std=float(self.istd[b]),
        )

    def _frame(self, X225, comp, istd):
        n = len(X225)
        out = np.empty((n, len(self.chem_cols)))
        out[:, self._pos_gen] = X225
        out[:, self._pos_comp] = comp
        out[:, self._pos_istd] = istd
        return pd.DataFrame(out, columns=self.chem_cols)

    def source_input231(self, sr: SourceRepr) -> pd.DataFrame:
        return self._frame(sr.x_src[None], sr.comp_src[None], sr.internal_std)

    def transfer_input231(self, sr: SourceRepr, U_new, keep_resid=True, X225=None):
        U_new = np.atleast_2d(U_new)
        X225 = self.gen.mg.inverse(U_new) if X225 is None else X225
        V = self.comp.predict_latent(U_new)
        if keep_resid:
            V = V + (sr.comp_obs_latent - self.comp.predict_latent(sr.u[None])[0])
        return self._frame(X225, self.comp.to_original(V), sr.internal_std)

    def raw_input231(self, sr: SourceRepr, X225):
        X225 = np.atleast_2d(X225)
        return self._frame(X225, np.repeat(sr.comp_src[None], len(X225), 0), sr.internal_std)

    def resid_comp(self, b):
        u = self.gen.U[b]
        return self.comp.observed_latent(self.Xc[b])[0] - self.comp.predict_latent(u[None])[0]
