import numpy as np
import pandas as pd

from .data import ChemNAEncoder
from .features import monotone_cst
from .model import PARAM_KEYS, make_hgb
from .paths import ROOT
from .splits import inner_splits

OAV_CLASSES = {
    "esters": (
        "ethyl_2.methyl_butyrate",
        "ethyl_acetate",
        "ethyl_decanoate",
        "ethyl_hexanoate",
        "ethyl_isovalerate",
        "ethyl_octanoate",
        "ethyl_propionate",
        "isobutyl_acetate",
        "isopentyl_acetate",
        "octyl_acetate",
        "phenethyl_acetate",
        "ethyl.isobutyrate",
        "ethyl.butyrate",
    ),
    "sulfur": ("DES", "DMS", "H2S", "CS2"),
}
DICTIONARY_CSV = ROOT / "experimentalCode" / "results" / "data_audit" / "chemical_dictionary.csv"
N_THRESHOLDS = 43


def load_thresholds(path=DICTIONARY_CSV) -> dict:
    d = pd.read_csv(path, encoding="utf-8-sig")
    d = d[d["threshold_mean"].notna()]
    out = {
        str(r.csv1_field): (
            float(r.threshold_mean),
            str(r.transform_interpretation).strip().startswith("log10"),
        )
        for r in d.itertuples(index=False)
    }
    assert len(out) == N_THRESHOLDS, len(out)
    assert all(t > 0 for t, _ in out.values())
    return out


def threshold_units(path=DICTIONARY_CSV) -> dict:
    d = pd.read_csv(path, encoding="utf-8-sig")
    d = d[d["threshold_mean"].notna()]
    return dict(zip(d["csv1_field"].astype(str), d["unit"].astype(str)))


def concentration(x, is_log10) -> np.ndarray:
    x = np.asarray(x, float)
    return 10**x if is_log10 else x


def oav_class_sums(X_enc, thresholds, classes=OAV_CLASSES) -> pd.DataFrame:
    out = pd.DataFrame(index=X_enc.index)
    for name, cols in classes.items():
        missing = [c for c in cols if c not in X_enc.columns or c not in thresholds]
        if missing:
            raise KeyError(f"{name} 类缺少列或阈值: {missing}")
        sub = X_enc[list(cols)]
        if sub.isna().any().any():
            raise ValueError("编码后的化学表含 NaN")
        total = np.zeros(len(X_enc))
        for c in cols:
            thr, is_log = thresholds[c]
            total += concentration(sub[c].to_numpy(), is_log) / thr
        out[f"oav_{name}"] = total
    return out


def fit_fold_oav(
    X_raw,
    Y,
    dims,
    dim_groups,
    groups,
    style,
    train_idx,
    test_idx,
    inner_seed,
    chosen,
    thresholds,
    add_oav=True,
) -> dict:
    tr, te = np.sort(train_idx), np.sort(test_idx)
    dims = list(dims)
    group_of = {d: g for g, ds in dim_groups.items() for d in ds}

    def encode(fit_rows, pred_rows):
        enc = ChemNAEncoder("col_mean").fit(X_raw.iloc[fit_rows])
        A, B = enc.transform(X_raw.iloc[fit_rows]), enc.transform(X_raw.iloc[pred_rows])
        if add_oav:
            A = pd.concat([A, oav_class_sums(A, thresholds)], axis=1)
            B = pd.concat([B, oav_class_sums(B, thresholds)], axis=1)
        return list(A.columns), A.to_numpy(np.float64), B.to_numpy(np.float64)

    def fit_dims(a, b, out, pos):
        names, A, B = encode(a, b)
        for j, d in enumerate(dims):
            best = chosen[group_of[d]]
            params = {k: best[k] for k in PARAM_KEYS}
            cst = monotone_cst(names, d, best["mono"])
            out[pos, j] = make_hgb(params, cst).fit(A, Y[a, j]).predict(B)

    inner = np.full((len(tr), len(dims)), np.nan)
    for a, b in inner_splits(tr, groups, style, inner_seed):
        fit_dims(a, b, inner, np.searchsorted(tr, b))
    oof = np.full((len(te), len(dims)), np.nan)
    fit_dims(tr, te, oof, np.arange(len(te)))
    return {"train_idx": tr, "test_idx": te, "oof": oof, "inner": inner}


def inner_ratio(inner, y, train_idx) -> np.ndarray:
    yt = np.asarray(y)[np.sort(train_idx)]
    sd = np.std(yt, axis=0, ddof=0)
    rmse = np.sqrt(np.mean((yt - inner) ** 2, axis=0))
    out = np.full(yt.shape[1], np.nan)
    ok = sd > 0
    out[ok] = rmse[ok] / sd[ok]
    return out


def exceed_count(X_raw, thresholds) -> int:
    n = 0
    for c, (thr, is_log) in thresholds.items():
        if c not in X_raw.columns:
            continue
        v = concentration(X_raw[c].to_numpy(), is_log)
        if np.nansum(v > thr) > 0:
            n += 1
    return n
