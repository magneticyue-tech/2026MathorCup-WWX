import numpy as np
import pandas as pd

from ..q1.discretize import dist_median
from .q1_bridge import grade_dist, predict_mu


def assemble_input(gen, comp, u, internal_std_value, chem_cols, gen_cols, comp_cols, istd_col):
    u = np.atleast_2d(u)
    out = pd.DataFrame(gen.mg.inverse(u), columns=gen_cols)
    xc = comp.to_original(comp.predict_latent(u))
    for i, c in enumerate(comp_cols):
        out[c] = xc[:, i]
    out[istd_col] = internal_std_value
    return out[list(chem_cols)]


def portrait_samples(gen, comp, bundle, style, n, rng, cols, istd_value):
    u = gen.sample_latent(style, n, rng)
    X = assemble_input(gen, comp, u, istd_value, *cols)
    mu = predict_mu(bundle, X)
    p = grade_dist(bundle, mu)
    G = dist_median(p)
    return {"u": u, "x": X.to_numpy(float), "mu": mu, "G": G, "p": p}


def summarize(mu, G, p) -> dict:
    pi = np.stack([(G == k).mean(0) for k in range(7)], axis=-1)
    return {
        "mu_q10": np.quantile(mu, 0.1, axis=0),
        "mu_q50": np.quantile(mu, 0.5, axis=0),
        "mu_q90": np.quantile(mu, 0.9, axis=0),
        "pi": pi,
        "mode": np.argmax(pi, axis=-1),
        "pi_tilde": p.mean(0),
    }


def chem_fingerprint(gen, style, n_boot=1000, seed=0) -> dict:
    sd = np.sqrt(np.diag(gen.C))
    delta = (gen.mu[style] - gen.g) / sd
    rows = np.where(gen.s == style)[0]
    rng = np.random.default_rng(seed)
    n = len(rows)
    boots = np.empty((n_boot, len(sd)))
    for b in range(n_boot):
        pick = rows[rng.integers(0, n, n)]
        mu_b = (n * gen.U[pick].mean(0) + gen.kappa * gen.g) / (n + gen.kappa)
        boots[b] = (mu_b - gen.g) / sd
    lo, hi = np.quantile(boots, [0.05, 0.95], axis=0)
    return {"delta": delta, "lo": lo, "hi": hi, "covers_zero": (lo <= 0) & (hi >= 0), "rows": rows}


def fingerprint_table(fp, style, gen_cols, modules, dic, x_filled, style_arr, top=10):
    delta = fp["delta"]
    order = np.argsort(-np.abs(delta))
    mins = x_filled.min(0)
    m_style = np.asarray(style_arr) == style
    seen, rows = set(), []
    for j in order:
        m = modules[j]
        if m in seen:
            continue
        seen.add(m)
        same = [gen_cols[i] for i in np.where(modules == m)[0] if i != j]
        c = gen_cols[j]
        rows.append(
            {
                "style": style,
                "rank": len(rows) + 1,
                "compound": c,
                "delta": delta[j],
                "delta_lo": fp["lo"][j],
                "delta_hi": fp["hi"][j],
                "ci_covers_zero": bool(fp["covers_zero"][j]),
                "q2_module": int(m),
                "same_module": "|".join(same),
                "origin_paper": dic.loc[c, "origin_paper"] if c in dic.index else np.nan,
                "off_flavors": dic.loc[c, "off_flavors"] if c in dic.index else np.nan,
                "aging": dic.loc[c, "aging"] if c in dic.index else np.nan,
                "floor_frac_style": float(np.isclose(x_filled[m_style, j], mins[j]).mean()),
            }
        )
        if len(rows) == top:
            break
    return pd.DataFrame(rows)


def sensory_fingerprint(styles, mu_by_style, dims, weak):
    allmu = np.concatenate([mu_by_style[s] for s in styles])
    med, sd = np.median(allmu, 0), allmu.std(0)
    dev = np.array(
        [(np.median(mu_by_style[s], 0) - med) / np.where(sd > 0, sd, np.nan) for s in styles]
    )
    rows = []
    for i, s in enumerate(styles):
        for j, d in enumerate(dims):
            rows.append(
                {
                    "style": s,
                    "dim": d,
                    "deviation_sd": dev[i, j],
                    "abs_rank": int((np.abs(dev[i]) > np.abs(dev[i, j])).sum() + 1),
                    "note": "同 stale_hops" if d == "barnyard" else "",
                    "weak_dim": bool(weak[j]),
                }
            )
    return dev, pd.DataFrame(rows)
