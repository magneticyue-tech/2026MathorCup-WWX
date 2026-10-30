import numpy as np

from .controls import draw_g1, draw_g2, draw_g3
from .gen_metrics import (
    GEN_NAMES,
    auc_from_oof,
    dcr,
    dispersion_values,
    floor_metrics,
    make_rf,
    module_rho,
    paired_auc,
    rf_style_accuracy,
    spearman,
    within_style_corr_error,
)
from .generator import StyleGen, fit_grid
from .paths import SEED
from .selection import outer_splits


def _paired(draw, s_te, rng):
    out = None
    for st in np.unique(s_te):
        m = np.where(s_te == st)[0]
        G = draw(st, len(m), rng)
        if out is None:
            out = np.empty((len(s_te), G.shape[1]))
        out[m] = G
    return out


def fit_fold_generators(X_tr, s_tr, styles, cfg, opts, which):
    kind = opts.get("kind", cfg["kind"])
    k = opts.get("k", cfg["k"]) if kind in ("fa", "famod") else None
    kappa = opts.get("kappa", cfg["kappa"])
    tie = opts.get("tie_scheme", "cond_mean")
    na_fill = opts.get("na_fill", "mean")
    style_scale = opts.get("style_scale", False)
    gens = {}
    if style_scale or na_fill != "mean":
        gens["main"] = StyleGen(kind, k, kappa, False, tie, SEED, style_scale, na_fill).fit(
            X_tr, s_tr, styles
        )
    else:
        cfgs = [(kind, k, kappa)]
        if "pure_fa" in which:
            cfgs.append(("fa", k, kappa))
        g = fit_grid(X_tr, s_tr, styles, cfgs, tie_scheme=tie)
        gens["main"] = g[(kind, k, kappa)]
        if "pure_fa" in which:
            gens["pure_fa"] = g[("fa", k, kappa)]
    if "G0" in which:
        gens["G0"] = StyleGen(kind, k, kappa, True, tie, SEED).fit(X_tr, s_tr, styles)
    return gens


def gen_rep(X, S, styles, rep, cfg_by_fold, opts=None, which=GEN_NAMES, smoke_folds=None):
    """一次重复：5 个外层折的配对生成池、DCR、离散、TRTR/TRTS/TSTR（方法论 §6.2）。"""
    opts = {} if opts is None else opts
    S = np.asarray(S)
    n, p = X.shape
    G = len(which)
    pool = {
        "real_x": np.full((n, p), np.nan),
        "real_u": np.full((n, p), np.nan),
        "gen_x": np.full((G, n, p), np.nan),
        "gen_u": np.full((G, n, p), np.nan),
        "fold": np.full(n, -1),
        "d_real": np.full(n, np.nan),
        "d_gen": np.full((G, n), np.nan),
        "disp": np.full(n, np.nan),
        "trtr": np.zeros(n, bool),
        "trts": np.zeros((G, n), bool),
        "tstr": np.zeros((G, n), bool),
        "mu": [],
        "C": [],
        "pure_equals_main": [],
    }
    for r, fold, tr, te in outer_splits(S):
        if r != rep or (smoke_folds is not None and fold not in smoke_folds):
            continue
        cfg = cfg_by_fold[(rep, fold)]
        gens = fit_fold_generators(X[tr], S[tr], styles, cfg, opts, which)
        main = gens["main"]
        pool["pure_equals_main"].append(cfg["kind"] == "fa")
        Xtr_f = np.where(np.isnan(X[tr]), main.colmean, X[tr])
        Xte_f = np.where(np.isnan(X[te]), main.colmean, X[te])
        s_tr, s_te = S[tr], S[te]
        U_te = main.latent(X[te])
        pool["real_x"][te] = Xte_f
        pool["real_u"][te] = U_te
        pool["fold"][te] = fold
        pool["d_real"][te] = dcr(U_te, main.U)
        mu_te = {st: main.mu[st] for st in np.unique(s_te)}
        cov_te = {st: main.cov_for(st) for st in mu_te} if main.Cs else main.C
        pool["disp"][te] = dispersion_values(U_te, s_te, mu_te, cov_te)
        pool["mu"].append(
            np.array([main.mu[st] if main.has_style(st) else np.full(p, np.nan) for st in styles])
        )
        pool["C"].append(main.C)
        pool["trtr"][te] = rf_style_accuracy(Xtr_f, s_tr, Xte_f, s_te)
        rf = make_rf().fit(Xtr_f, s_tr)
        present = [st for st in styles if (s_tr == st).any()]
        n_by = {st: int((s_tr == st).sum()) for st in present}
        for gi, name in enumerate(which):
            rng = np.random.default_rng([SEED, rep, fold, gi])
            if name in ("main", "pure_fa", "G0"):
                g = gens[name]

                def draw(st, m, rr, g=g):
                    return g.sample(st, m, rr)

            elif name == "G1":

                def draw(st, m, rr):
                    return draw_g1(Xtr_f, s_tr, st, m, rr)

            elif name == "G2":

                def draw(st, m, rr):
                    return draw_g2(Xtr_f, s_tr, st, m, rr)

            else:

                def draw(st, m, rr):
                    return draw_g3(main, s_tr, st, m, rr)

            Gx = _paired(draw, s_te, rng)
            pool["gen_x"][gi, te] = Gx
            Gu = main.latent(Gx)
            pool["gen_u"][gi, te] = Gu
            pool["d_gen"][gi, te] = dcr(Gu, main.U)
            pool["trts"][gi, te] = rf.predict(Gx) == s_te
            Gs = np.vstack([draw(st, n_by[st], rng) for st in present])
            Ss = np.concatenate([[st] * n_by[st] for st in present])
            pool["tstr"][gi, te] = rf_style_accuracy(Gs, Ss, Xte_f, s_te)
    pool["mu"] = np.array(pool["mu"])
    pool["C"] = np.array(pool["C"])
    pool["pure_equals_main"] = np.array(pool["pure_equals_main"])
    return pool


def scalar_metrics(pool, idx, style, mins, inmod, oof, which=GEN_NAMES, point=None):
    """在行号 idx 上计算 §6.2 的标量指标。point 不为 None 时 AUC 取逐折平均（点估计）。"""
    n = pool["real_x"].shape[0]
    rx, ru = pool["real_x"][idx], pool["real_u"][idx]
    s = np.asarray(style)[idx]
    out = {}
    cr = spearman(rx)
    out["real"] = {
        "trtr": float(pool["trtr"][idx].mean()),
        "real_module_rho": module_rho(cr, inmod),
    }
    for key, var in (("rr_auc_marg", "rr_marg"), ("rr_auc_cond", "rr_cond")):
        if point is not None:
            out["real"][key] = point.get(var, np.nan)
        elif var in oof:
            o, y, ia, ib = oof[var]
            pos = {v: i for i, v in enumerate(ia)}
            pos_b = {v: i for i, v in enumerate(ib)}
            sa = [pos[i] for i in idx if i in pos]
            sb = [len(ia) + pos_b[i] for i in idx if i in pos_b]
            sel = np.r_[sa, sb].astype(int)
            out["real"][key] = np.nan if len(sel) == 0 else _auc(y[sel], o[sel])
        else:
            out["real"][key] = np.nan
    real_row = None
    for gi, name in enumerate(which):
        gx, gu = pool["gen_x"][gi, idx], pool["gen_u"][gi, idx]
        cg = spearman(gx)
        err, row, real_row = floor_metrics(rx, gx, mins)
        m = {
            "dcr_ratio": float(np.median(pool["d_gen"][gi, idx]) / np.median(pool["d_real"][idx])),
            "near_copy": float(
                np.mean(pool["d_gen"][gi, idx] < np.quantile(pool["d_real"][idx], 0.05))
            ),
            "trts": float(pool["trts"][gi, idx].mean()),
            "tstr": float(pool["tstr"][gi, idx].mean()),
            "floor_err": err,
            "floor_row": row,
            "corr_err": float(np.abs(_up(cr) - _up(cg)).mean()),
            "wcorr_err": within_style_corr_error(ru, s, gu, s),
            "module_rho": module_rho(cg, inmod),
        }
        for var in ("marg", "cond"):
            if point is not None:
                m[f"auc_{var}"] = point[(name, var)]
            elif (name, var) in oof:
                o, y = oof[(name, var)]
                m[f"auc_{var}"] = paired_auc(o, y, idx, n)
        out[name] = m
    out["real"]["real_floor_row"] = real_row
    return out


def _up(C):
    return C[np.triu_indices(len(C), 1)]


def _auc(y, p):
    return auc_from_oof(y, p)
