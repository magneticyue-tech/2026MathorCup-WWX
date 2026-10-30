import numpy as np
from scipy.special import logsumexp

from ..q1.data import DIMS
from .config import ALPHA_GRID, DEFECT_SENS
from .defects import violation
from .q1_bridge import grade_median, grade_quant, predict_mu

DEFECT_SENS_IDX = [DIMS.index(d) for d in DEFECT_SENS]
assert len(DEFECT_SENS_IDX) == 6
RULES = ("grade", "mu_cont", "grade_quant", "none")


def sens_values(bundle, X231, rule):
    if rule == "none":
        return None
    mu = predict_mu(bundle, X231, DEFECT_SENS_IDX)
    if rule == "grade":
        return grade_median(bundle, mu, DEFECT_SENS_IDX)
    if rule == "grade_quant":
        return grade_quant(bundle, mu, DEFECT_SENS_IDX)
    if rule == "mu_cont":
        return mu
    raise ValueError(rule)


def sens_feasible(vals, base, rule):
    if rule == "none":
        return None
    if rule == "mu_cont":
        return np.all(vals <= base + 1e-9, axis=-1)
    return np.all(vals <= base, axis=-1)


def path_points(u_b, u_end, alphas=ALPHA_GRID):
    return u_b[None] + alphas[:, None] * (u_end - u_b)[None]


def path_search(
    builder, bundle, sr, b, spec, u_end, rule, base, keep_resid=True, alphas=ALPHA_GRID
):
    gen = builder.gen
    U = path_points(sr.u, u_end, alphas)
    X225 = gen.mg.inverse(U)
    ok_chem = ~violation(X225, b, spec).any(1)
    if rule == "none":
        ok_sens = np.ones(len(alphas), bool)
        vals = None
    else:
        X231 = builder.transfer_input231(sr, U, keep_resid, X225)
        vals = sens_values(bundle, X231, rule)
        ok_sens = sens_feasible(vals, base, rule)
    feas = ok_chem & ok_sens
    if not feas[0]:
        raise RuntimeError("α=0 不可行，源表示下应精确可行")
    a_idx = int(np.where(feas)[0].max())
    bad = np.where(~feas)[0]
    first = int(bad[0]) if len(bad) else -1
    viol_dims = []
    if first >= 0 and vals is not None:
        cmp = vals[first] > (base + 1e-9 if rule == "mu_cont" else base)
        viol_dims = [DEFECT_SENS[k] for k in np.where(cmp)[0]]
    return {
        "U": U,
        "X225": X225,
        "vals": vals,
        "ok_sens": ok_sens,
        "ok_chem_orig": ok_chem,
        "feasible": feas,
        "alpha_idx": a_idx,
        "alpha_star": float(alphas[a_idx]),
        "first_infeasible": first,
        "violated_dims": viol_dims,
        "first_chem_viol": bool(first >= 0 and not ok_chem[first]),
        "n_recheck_fail": int((~ok_chem).sum()),
        "discontinuous": bool(feas[: a_idx + 1].sum() < a_idx + 1),
    }


class StylePosterior:
    """生成器各风格高斯密度按贝叶斯公式给出的风格后验（共享协方差）。"""

    def __init__(self, gen, styles, log_prior):
        self.styles = list(styles)
        Linv = np.linalg.inv(gen.L)
        self.Linv = Linv
        self.M = np.array([Linv @ gen.mu[s] for s in self.styles])
        self.log_prior = np.asarray(log_prior)

    def log_post(self, U):
        Z = np.atleast_2d(U) @ self.Linv.T
        d2 = (Z**2).sum(1)[:, None] - 2 * Z @ self.M.T + (self.M**2).sum(1)[None, :]
        lp = -0.5 * d2 + self.log_prior[None, :]
        return lp - logsumexp(lp, axis=1, keepdims=True)


def alpha_fit(post, feasible, U_path, target, alphas=ALPHA_GRID):
    ti = post.styles.index(target)
    lp = post.log_post(U_path)[:, ti]
    lp = np.where(feasible, lp, -np.inf)
    i = int(np.flatnonzero(lp == lp.max())[0])
    return float(alphas[i]), i
