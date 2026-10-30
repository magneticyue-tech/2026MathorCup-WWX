# code/src/q2/full_controls.py
"""Q2 对照 M2（ElasticNet）、M3（问题 1 GBM + 外层测试折置换重要度）、风格内选择；合并入最终表。"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parents[1]
for _p in (CURRENT_DIR, CODE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import ElasticNetCV  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import config as C  # noqa: E402
import core  # noqa: E402
from data import load_q2_dataset  # noqa: E402
from labels import attachment8_labels, candidate_type  # noqa: E402
from rationality import style_design  # noqa: E402
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q1.features import monotone_cst  # noqa: E402
from src.q1.model import PARAM_KEYS, make_hgb  # noqa: E402

DIM_TO_GROUP = {d: g for g, ds_ in DIM_GROUPS.items() for d in ds_}


def q1_m3_setup(ds):
    """读取 Q1 的外层划分与逐折组内参数。

    外层划分直接由 q1_main_oof.npz 的 fold_id 重建，不调用 outer_splits 重算：
    StratifiedGroupKFold 的打乱随 sklearn 版本变化，Q1 正式产物是权威来源。
    """
    z = np.load(C.Q1_MAIN_OOF_NPZ, allow_pickle=False)
    chosen = json.loads(str(z["chosen_params"]))
    fold_id = z["fold_id"]
    q1d = np.load(C.Q1_DATA_NPZ, allow_pickle=False)
    style = q1d["style"]
    if not (style == ds.style).all():
        raise ValueError("Q1 style 与附件 1 不一致")
    if fold_id.shape != (C.M3_N_REPEATS_SPLIT, 250) or set(np.unique(fold_id)) != set(range(C.M3_N_SPLITS)):
        raise ValueError(f"Q1 fold_id 应为 {C.M3_N_REPEATS_SPLIT}×250，取值 0..{C.M3_N_SPLITS - 1}")
    idx = np.arange(fold_id.shape[1])
    splits = [[(idx[fold_id[r] != k], idx[fold_id[r] == k]) for k in range(C.M3_N_SPLITS)]
              for r in range(C.M3_N_REPEATS_SPLIT)]
    for r, rep in enumerate(splits):
        for k, (tr, te) in enumerate(rep):
            if set(ds.date_group[tr]) & set(ds.date_group[te]):
                raise AssertionError(f"重复 {r} 折 {k}：日期组跨训练/测试")
            for g in DIM_GROUPS:
                if f"{r}_{k}_{g}" not in chosen:
                    raise KeyError(f"Q1 chosen_params 缺少 {r}_{k}_{g}")
    return splits, chosen


def m3_dimension(Xraw, y, cand, dim, splits, chosen, repeats, folds):
    group = DIM_TO_GROUP[dim]
    imps, recs = [], []
    for r in repeats:
        for k in folds:
            tr, te = splits[r][k]
            best = chosen[f"{r}_{k}_{group}"]
            params = {p: best[p] for p in PARAM_KEYS}
            cst = monotone_cst(cand, dim, bool(best["mono"]))
            mu = Xraw.iloc[tr].mean()
            Xtr = Xraw.iloc[tr].fillna(mu).to_numpy(float)
            Xte = Xraw.iloc[te].fillna(mu).to_numpy(float)
            model = make_hgb(params, cst).fit(Xtr, y[tr])
            pi = permutation_importance(model, Xte, y[te], n_repeats=C.M3_PERM_REPEATS,
                                        random_state=C.SEED + 10 * r + k, n_jobs=1)
            imps.append(pi.importances_mean)
            recs.append({"dimension": dim, "repeat": r, "fold": k, "group": group,
                         "n_train": len(tr), "n_test": len(te), **{f"param_{p}": best[p] for p in PARAM_KEYS},
                         "mono": bool(best["mono"]), "n_monotone": int(np.count_nonzero(cst))})
    return np.array(imps), recs


def run_m3(ds, dims, repeats, folds, n_jobs):
    splits, chosen = q1_m3_setup(ds)
    Xraw = ds.chem_raw[ds.candidate_cols]
    Y = ds.Y_rank
    res = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(m3_dimension)(Xraw, Y[d].to_numpy(), ds.candidate_cols, d, splits, chosen, repeats, folds)
        for d in dims
    )
    imp = np.stack([r[0] for r in res])
    recs = [x for r in res for x in r[1]]
    return imp, pd.DataFrame(recs)


def run_m2(a, groups):
    cv = list(GroupKFold(n_splits=C.ENET_N_SPLITS).split(a.Zx, groups=groups))
    coef = np.zeros((len(a.cand), len(a.dims)))
    alphas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for d in range(len(a.dims)):
            m = ElasticNetCV(l1_ratio=C.ENET_L1_RATIO, cv=cv, max_iter=20000, n_jobs=1).fit(a.Zx, a.Zy[:, d])
            coef[:, d] = m.coef_
            alphas.append(m.alpha_)
    return coef, np.array(alphas)


def run_within(ds, a):
    D = style_design(ds.style)
    def res_std(Z):
        R = Z - D @ np.linalg.lstsq(D, Z, rcond=None)[0]
        return (R - R.mean(0)) / R.std(0)
    Zx, Zy = res_std(a.Zx), res_std(a.Zy)
    Cxx, Cxy = core.corr_mats(Zx, np.ones(Zx.shape[1], bool), Zy, np.ones(Zy.shape[1], bool))
    sel, score = core.forward_select(Cxx, Cxy, a.mod, a.okx)
    return sel, score


def smoke(ds):
    dim = ds.delivery_sensory_cols[0]
    imp, recs = run_m3(ds, [dim], repeats=[0], folds=range(5), n_jobs=1)
    splits, chosen = q1_m3_setup(ds)
    group = DIM_TO_GROUP[dim]
    for _, rr in recs.iterrows():
        best = chosen[f"{rr['repeat']}_{rr['fold']}_{group}"]
        for p in PARAM_KEYS:
            assert rr[f"param_{p}"] == best[p]
        tr, te = splits[0][rr["fold"]]
        assert not set(ds.date_group[tr]) & set(ds.date_group[te])
    C.SMOKE_DIR.mkdir(exist_ok=True)
    recs.to_csv(C.SMOKE_DIR / "m3_smoke_records.csv", index=False)
    print(f"M3 小样本：维度 {dim}，组 {group}，{imp.shape[1]} 折重要度记录，参数与 Q1 逐项一致，日期组不相交")


def main(n_jobs=16):
    ds = load_q2_dataset()
    a = core.prepare_arm(ds, core.ArmSpec("main"))
    rat = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    rat = rat[rat["method"] == "main"].copy()
    core.check_147(rat, "主方法")
    sel = core.sel_from_table(rat, a.cand, a.dims)
    ctrl = pd.read_csv(C.Q2_DIR / "controls_sensitivity_results.csv").set_index("dimension")
    _, Cxy = core.corr_mats(a.Zx, a.okx, a.Zy, a.oky)

    print("M2 ElasticNet（日期组分组 5 折）...", flush=True)
    coef, alphas = run_m2(a, ds.date_group)
    m2 = core.ranked_select(coef, a.mod, a.okx, tie=Cxy)
    print("风格内选择...", flush=True)
    wsel, wscore = run_within(ds, a)
    print("M3 GBM（Q1 参数，5×5 外层，测试折置换重要度 n_repeats=5）...", flush=True)
    imp, recs = run_m3(ds, a.dims, repeats=range(C.M3_N_REPEATS_SPLIT), folds=range(C.M3_N_SPLITS), n_jobs=n_jobs)
    if imp.shape[1] != 25:
        raise AssertionError("M3 应有 25 个外层折")
    imp_mean = imp.mean(axis=1).T
    m3 = core.ranked_select(imp_mean, a.mod, a.okx, tie=Cxy)
    np.savez_compressed(C.Q2_DIR / "q2_m3_importance.npz", importance=imp, dims=np.array(a.dims),
                        cand=np.array(a.cand), enet_coef=coef, enet_alpha=alphas)
    recs.to_csv(C.Q2_DIR / "m3_fold_records.csv", index=False)

    names = lambda s: [a.cand[j] for j in s]  # noqa: E731
    rows = []
    for d, dim in enumerate(a.dims):
        b = sel[d]
        row = {"dimension": dim, "main_top3": "|".join(names(b))}
        for m in ("M0", "M1"):
            s = [a.cand.index(c) for c in ctrl.at[dim, f"{m}_top3"].split("|")]
            row[f"{m}_top3"] = ctrl.at[dim, f"{m}_top3"]
            row[f"jaccard_{m}"] = core.jaccard(b, s)
            row[f"jaccard_mod_{m}"] = core.jaccard(a.mod[b], a.mod[s])
            row[f"{m}_top1_match"] = bool(b[0] == s[0])
        for m, s in (("M2", m2[d]), ("M3", m3[d]), ("Within", wsel[d])):
            row[f"{m}_top3"] = "|".join(names(s))
            row[f"jaccard_{m}"] = core.jaccard(b, s)
            row[f"jaccard_mod_{m}"] = core.jaccard(a.mod[b], a.mod[s])
            row[f"{m}_top1_match"] = bool(b[0] == s[0])
        row["M2_alpha"] = alphas[d]
        row["M3_top1_importance"] = imp_mean[m3[d][0], d]
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(C.Q2_DIR / "full_controls_results.csv", index=False)

    for m in ("M0", "M1", "M2", "M3", "Within"):
        sets = res.set_index("dimension")[f"{m}_top3"].str.split("|").to_dict()
        rat[f"in_{m}"] = [c in sets[dim] for dim, c in zip(rat["dimension"], rat["compound"])]

    lab = attachment8_labels(ds.chem_cols)
    extra = []
    for d, dim in enumerate(a.dims):
        j = m3[d][0]
        if j == sel[d][0]:
            continue
        c = a.cand[j]
        fold_imp = imp[d, :, j]
        row = {col: np.nan for col in rat.columns}
        row.update({
            "dimension": dim, "rank": 1, "compound": c, "compound_type": candidate_type(c),
            "module_id": int(a.mod[j]),
            "same_module_compounds": "|".join(x for x, m in zip(a.cand, a.mod) if m == a.mod[j] and x != c),
            "score_type": "m3_permutation_importance", "score": float(imp_mean[j, d]),
            "method": "M3", "source_label": "M3 GBM 外层测试折置换重要度（25 折均值）",
            "m3_importance_fold_q025": float(np.quantile(fold_imp, 0.025)),
            "m3_importance_fold_q975": float(np.quantile(fold_imp, 0.975)),
            "r_marginal": float(Cxy[j, d]),
            "main_top1": a.cand[sel[d][0]],
            "not_applicable": "q_d、显著性、CI、π^c、π^m、稳定性、等级、风格内区间、共变标记：M3 对照行不适用",
            "note": C.STALE_HOPS_NOTE if dim == "stale_hops" else "", "interpretation": "",
        })
        for f in ("chemical_name", "origin_paper", "off_flavors", "aging", "threshold_mean"):
            row[f] = lab.at[c, f]
        for m in ("M0", "M1", "M2", "Within"):
            row[f"in_{m}"] = c in res.at[d, f"{m}_top3"].split("|")
        row["in_M3"] = True
        extra.append(row)
    rat["source_label"] = "主方法：图约束前向选择"
    out = pd.concat([rat, pd.DataFrame(extra)], ignore_index=True)
    out.to_csv(C.Q2_DIR / "final_top3_with_rationality.csv", index=False)

    for m in ("M2", "M3", "Within"):
        print(f"主方法 vs {m}：化合物层 Jaccard {res[f'jaccard_{m}'].mean():.3f}，模块层 "
              f"{res[f'jaccard_mod_{m}'].mean():.3f}，Top-1 相同比例 {res[f'{m}_top1_match'].mean():.2f}")
    print(f"M3 追加行 {len(extra)}（= M3 与主方法 Top-1 不同的维度数）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()
    if args.smoke:
        smoke(load_q2_dataset())
    else:
        main(args.n_jobs)
