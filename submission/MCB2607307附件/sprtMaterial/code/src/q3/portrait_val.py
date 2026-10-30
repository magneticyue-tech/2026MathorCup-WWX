import numpy as np
from scipy.stats import rankdata

from .composite import CompositeDeriver
from .generator import StyleGen
from .portrait import assemble_input
from .q1_bridge import fit_fold_bundle, fold_train_test, predict_mu
from .selection import inner_scores, select_config


def fold_unit(chem, xbar, G, oof, rep, fold, n_synth, seed):
    """画像验证的一个外层折（方法论 §6.3）。"""
    S = chem.style
    tr, te = fold_train_test(oof["fold_id"], rep, fold)
    XG = chem.XG
    sc, _, n_excl = inner_scores(XG, S, tr, rep, styles=chem.styles)
    cfg = select_config(sc)
    gen = StyleGen(cfg["kind"], cfg["k"], cfg["kappa"]).fit(XG[tr], S[tr], chem.styles)
    Xc = chem.X[chem.composite_cols].to_numpy(float)
    comp = CompositeDeriver().fit(gen.U, Xc[tr])
    istd = float(chem.X[chem.internal_std_col].iloc[tr].median())
    bundle = fit_fold_bundle(chem.X, xbar, G, tr, rep, fold, oof)
    cols = (chem.chem_cols, chem.gen_cols, chem.composite_cols, chem.internal_std_col)
    real_pred = predict_mu(bundle, chem.X.iloc[te])
    recon_in = chem.X.iloc[te].copy()
    der = assemble_input(gen, comp, gen.latent(XG[te]), istd, *cols)
    for c in (*chem.composite_cols, chem.internal_std_col):
        recon_in[c] = der[c].to_numpy()
    recon_pred = predict_mu(bundle, recon_in)
    cells = {}
    styles_tr = set(S[tr])
    for si, st in enumerate(chem.styles):
        m = S[te] == st
        if not m.any() or st not in styles_tr:
            continue
        rng = np.random.default_rng([seed, rep, fold, si])
        u = gen.sample_latent(st, n_synth, rng)
        mu = predict_mu(bundle, assemble_input(gen, comp, u, istd, *cols))
        cells[st] = {
            "test_rows": te[m],
            "gen_med": np.median(mu, 0),
            "gen_q10": np.quantile(mu, 0.1, 0),
            "gen_q90": np.quantile(mu, 0.9, 0),
            "real_pred": real_pred[m],
            "recon_pred": recon_pred[m],
            "obs": xbar[te[m]],
        }
    missing = [st for st in chem.styles if st not in styles_tr]
    n_one = [st for st in chem.styles if (S[tr] == st).sum() == 1]
    return {
        "rep": rep,
        "fold": fold,
        "config": cfg,
        "inner_excluded": n_excl,
        "cells": cells,
        "styles_missing_train": missing,
        "styles_one_train": n_one,
    }


def _spear(a, b):
    ra, rb = rankdata(a, axis=-1), rankdata(b, axis=-1)
    ra = ra - ra.mean(-1, keepdims=True)
    rb = rb - rb.mean(-1, keepdims=True)
    den = np.sqrt((ra**2).sum(-1) * (rb**2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra * rb).sum(-1) / den


def cell_metrics(c, w=None):
    """w 为测试啤酒的重抽次数（bootstrap），None 表示原样本。"""
    rp, ob, rc = c["real_pred"], c["obs"], c["recon_pred"]
    if w is not None:
        keep = w > 0
        if not keep.any():
            return None
        rp, ob, rc = np.repeat(rp, w, 0), np.repeat(ob, w, 0), np.repeat(rc, w, 0)
    rmed, omed = np.median(rp, 0), np.median(ob, 0)
    g = c["gen_med"]
    cov = ((rp >= c["gen_q10"]) & (rp <= c["gen_q90"])).mean()
    return np.array(
        [
            _spear(g, rmed),
            _spear(g, omed),
            np.abs(g - rmed).mean(),
            np.abs(g - omed).mean(),
            _spear(rmed, omed),
            np.abs(rmed - omed).mean(),
            cov,
            np.abs(rc - rp).mean(),
        ]
    )


METRIC_NAMES = (
    "rho_gen_realpred",
    "rho_gen_obs",
    "mae_gen_realpred",
    "mae_gen_obs",
    "rho_realpred_obs",
    "mae_realpred_obs",
    "coverage_q10_q90",
    "recon_abs_change",
)
