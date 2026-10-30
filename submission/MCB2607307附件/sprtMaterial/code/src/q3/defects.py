from dataclasses import dataclass

import numpy as np

from .config import DEFECT_CHEM
from .data import threshold

GRID = np.linspace(-6, 6, 4801)


@dataclass
class DefectSpec:
    cols: list
    idx: np.ndarray
    theta: np.ndarray
    has_threshold: np.ndarray


def build_defect_spec(gen_cols, dic, kind="threshold_mean") -> DefectSpec:
    cols = list(DEFECT_CHEM)
    gi = {c: i for i, c in enumerate(gen_cols)}
    missing = [c for c in cols if c not in gi]
    assert not missing, missing
    theta = np.array([threshold(c, dic, kind) for c in cols])
    return DefectSpec(cols, np.array([gi[c] for c in cols]), theta, ~np.isnan(theta))


def chem_bounds(x_src_row, spec: DefectSpec, rule="threshold") -> np.ndarray:
    x = np.asarray(x_src_row, float)[spec.idx]
    if rule == "strict":
        return x.copy()
    if rule != "threshold":
        raise ValueError(rule)
    return np.where(spec.has_threshold, np.fmax(x, spec.theta), x)


class LatentBounder:
    def __init__(self, mg, spec: DefectSpec):
        self.spec = spec
        self.xgrid = [mg.inverse_column(j, GRID) for j in spec.idx]

    def beta(self, b, u_row) -> np.ndarray:
        out = np.empty(len(b))
        for k, xg in enumerate(self.xgrid):
            ok = xg <= b[k] + 1e-12
            if not ok.any():
                raise ValueError(f"缺陷列 {self.spec.cols[k]} 网格上无满足上界的点")
            out[k] = max(GRID[ok].max(), u_row[self.spec.idx[k]])
        return out


def violation(x_new, b, spec: DefectSpec, tol=1e-9) -> np.ndarray:
    x_new = np.asarray(x_new, float)
    return x_new[..., spec.idx] > b + tol
