# code/src/q2/rationality.py
"""Q2.4 合理性证据：附件 8 原始标签、OAV、breadth、风格内关联与 bootstrap CI、共变标记、模块可分检验、等级。"""
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd

import config as C
import core
from data import fill_chem, load_q2_dataset
from labels import attachment8_labels, check_against_dictionary, transform_kind


def style_design(style):
    s = pd.Series(style)
    dummies = pd.get_dummies(s, drop_first=True).to_numpy(float)
    return np.column_stack([np.ones(len(s)), dummies])


def _resid(D, M):
    beta, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ beta


def _corr_cols(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (A * B).sum(0) / (na * nb)
    r[(na < 1e-10) | (nb < 1e-10)] = np.nan
    return r


def style_within(ds, pairs, n_boot, seed):
    """固定 (维度, 化合物)：风格残差秩相关的估计值与品评团簇 bootstrap 95% CI。"""
    comps = sorted({c for _, c in pairs})
    dims = list(ds.delivery_sensory_cols)
    ci = [comps.index(c) for _, c in pairs]
    di = [dims.index(d) for d, _ in pairs]
    Xraw = ds.chem_raw[comps].to_numpy(float)
    Yraw = ds.sensory_raw[dims].to_numpy(float)

    def stat(idx):
        Zx, _ = core.zrank(core.fill_array(Xraw[idx], "mean"))
        Zy, _ = core.zrank(Yraw[idx])
        D = style_design(ds.style[idx])
        return _corr_cols(_resid(D, Zx)[:, ci], _resid(D, Zy)[:, di])

    est = stat(np.arange(len(ds.style)))
    idx_list = core.bootstrap_indices(ds.panels, n_boot, seed, "panel")
    boot = np.array([stat(idx) for idx in idx_list])
    n_nan = np.isnan(boot).sum(0)
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
    return est, lo, hi, n_nan


def module_split_test(ds, df, mod_map, q1):
    """§3.2 推翻检验：同模块成员 k 在 S∪{j} 之后的偏 r 是否超过 q_d。"""
    Zx = ds.X_rank
    Zy = ds.Y_rank
    n = len(Zx)
    out = []
    for _, row in df.iterrows():
        g = df[df["dimension"] == row["dimension"]].sort_values("rank")
        S = g["compound"].tolist()[: int(row["rank"])]
        mates = [c for c, m in mod_map.items() if m == mod_map[row["compound"]] and c != row["compound"]]
        if not mates:
            out.append("")
            continue
        Q = np.column_stack([np.ones(n), Zx[S].to_numpy()])
        H = Q @ np.linalg.pinv(Q)
        y = Zy[row["dimension"]].to_numpy()
        ey = y - H @ y
        E = Zx[mates].to_numpy() - H @ Zx[mates].to_numpy()
        r = (E.T @ ey) / (np.linalg.norm(E, axis=0) * np.linalg.norm(ey))
        hit = [f"{c}:{v:.3f}" for c, v in zip(mates, r) if abs(v) > q1[row["dimension"]]]
        out.append("|".join(hit))
    return out


def main(n_boot=C.N_BOOTSTRAP, seed=C.SEED, out_dir=C.Q2_DIR):
    ds = load_q2_dataset()
    df = pd.read_csv(C.Q2_DIR / "final_top3_with_stability.csv")
    core.check_147(df, "final_top3_with_stability")

    lab = attachment8_labels(ds.chem_cols)
    n_checked = check_against_dictionary(lab)
    df = df.merge(lab.drop(columns=["unit"]).reset_index(), on="compound", how="left", validate="m:1")

    kind = transform_kind(ds.chem_cols)
    oav, n_oav = {}, {}
    for c in df["compound"].unique():
        thr = lab.at[c, "threshold_mean"]
        if pd.isna(thr) or kind[c] == "uncalibrated":
            oav[c], n_oav[c] = np.nan, np.nan
            continue
        x = ds.chem_raw[c].dropna()
        conc = 10.0 ** x if kind[c] == "log10" else x
        oav[c], n_oav[c] = float((conc / thr >= 1).mean()), int(len(x))
    df["transform"] = df["compound"].map(kind)
    df["oav_ge_1_ratio"] = df["compound"].map(oav)
    df["oav_n_beers"] = df["compound"].map(n_oav)

    breadth = df.groupby("compound")["dimension"].nunique()
    df["n_dims_top3"] = df["compound"].map(breadth)

    pairs = list(zip(df["dimension"], df["compound"]))
    est, lo, hi, n_nan = style_within(ds, pairs, n_boot, seed)
    df["r_style"] = est
    df["r_style_ci_lower"] = lo
    df["r_style_ci_upper"] = hi
    df["r_style_boot_nan"] = n_nan
    df["r_style_n_boot"] = n_boot
    has_ci = np.isfinite(lo) & np.isfinite(hi)
    if not has_ci.all():
        print(f"风格内区间缺失 {int((~has_ci).sum())} 项，flag_style_ci0 保留缺失")
    flag = [bool(l <= 0 <= h) if ok else None for l, h, ok in zip(lo, hi, has_ci)]
    df["flag_style_ci0"] = pd.array(flag, dtype="boolean")
    df["flag_style_ci_ordered"] = pd.array(
        [bool(-1 <= l <= h <= 1) if ok else None for l, h, ok in zip(lo, hi, has_ci)],
        dtype="boolean")
    df["flag_breadth5"] = df["n_dims_top3"] >= C.BREADTH_FLAG
    df["flag_oav_lt1"] = df["oav_ge_1_ratio"] == 0
    df["covariation"] = (df["flag_style_ci0"] | df["flag_breadth5"].astype("boolean")
                         | df["flag_oav_lt1"].astype("boolean"))

    q1 = df[df["rank"] == 1].set_index("dimension")["q_d"].to_dict()
    mod_map = dict(zip(ds.candidate_cols, core.modules(
        ds.corr.loc[ds.candidate_cols, ds.candidate_cols].abs().values, C.TAU_MAIN).astype(int)))
    df["module_split_members"] = module_split_test(ds, df, mod_map, q1)

    df["sign"] = np.where(df["score"] >= 0, "+", "-")
    df["method"] = "main"
    df["note"] = np.where(df["dimension"] == "stale_hops", C.STALE_HOPS_NOTE, "")
    df["interpretation"] = ""

    regrade = {}
    for dim, g in df[df["rank"] == 1].groupby("dimension"):
        regrade[dim] = core.grade_from(bool(g["is_significant"].iloc[0]), float(g["pi_c"].iloc[0]))
    if (df["dimension"].map(regrade) != df["grade"]).any():
        raise AssertionError("等级复算与 stability 输出不一致")

    core.check_147(df, "final_top3_with_rationality")
    df.to_csv(Path(out_dir) / "final_top3_with_rationality.csv", index=False)
    print(f"附件 8 标签核对 {n_checked} 个单元一致；有阈值 Top-3 化合物 "
          f"{df.loc[df['threshold_mean'].notna(), 'compound'].nunique()} 个；"
          f"风格内 CI 含 0 {int((df['flag_style_ci0'] == True).sum())}/147；"
          f"breadth≥5 化合物 {breadth[breadth >= 5].to_dict()}；"
          f"共变标记 {int((df['covariation'] == True).sum())}/147；"
          f"模块可分假设被推翻的项 {int((df['module_split_members'] != '').sum())}")


if __name__ == "__main__":
    main()
