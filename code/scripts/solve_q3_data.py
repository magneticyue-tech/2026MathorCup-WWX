import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.composite import ALPHAS  # noqa: E402
from src.q3.config import FLOOR, SRC_STYLE  # noqa: E402
from src.q3.data import defect_table, load_chem, load_dictionary, threshold  # noqa: E402
from src.q3.generator import _within_resid  # noqa: E402
from src.q3.marg import Marg  # noqa: E402
from src.q3.meta import dump_json, make_check, run_meta  # noqa: E402
from src.q3.modules import modules_from_q2, n_multi  # noqa: E402
from src.q3.paths import PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402

PERM_SEED = 0
N_PERM_ETA = 200


def eta2(U, codes, n_groups):
    cnt = np.bincount(codes, minlength=n_groups)
    sums = np.zeros((n_groups, U.shape[1]))
    np.add.at(sums, codes, U)
    gm = U.mean(0)
    means = sums / np.maximum(cnt, 1)[:, None]
    between = (cnt[:, None] * (means - gm) ** 2).sum(0)
    return between / ((U - gm) ** 2).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    dic = load_dictionary()
    chem = load_chem(dic=dic)
    XG = chem.XG
    S = chem.style
    na = np.isnan(XG)
    colmean = np.nanmean(XG, 0)
    XGf = np.where(na, colmean, XG)
    mg = Marg().fit(XG)
    U = mg.transform(XGf)
    x_src = mg.inverse(U)
    mod = modules_from_q2(chem.gen_cols)

    checks = []
    cnt = np.unique(S, return_counts=True)[1]
    checks.append(make_check("生成列数", len(chem.gen_cols), 225))
    checks.append(make_check("风格数与样本量范围", [len(cnt), cnt.min(), cnt.max()], [22, 3, 31]))
    mins = XGf.min(0)
    tied = (XGf == mins).sum(0)
    floor = np.isclose(XGf, FLOOR)
    checks.append(
        make_check(
            "最小值并列列数与单元格数",
            [int((tied >= 2).sum()), int(tied[tied >= 2].sum())],
            [156, 7592],
        )
    )
    checks.append(
        make_check(
            "共同下限列数与单元格数", [int(floor.any(0).sum()), int(floor.sum())], [142, 6807]
        )
    )
    v = U.var(0)
    checks.append(make_check("潜变量方差最小值", float(v.min()), 0.46, 0.01))
    checks.append(make_check("潜变量方差 10% 分位", float(np.quantile(v, 0.1)), 0.87, 0.01))
    obs = ~na
    checks.append(
        make_check("非缺失往返最大误差", float(np.abs(x_src[obs] - XG[obs]).max()), 0.0, 1e-9)
    )
    ja = chem.gen_cols.index("amyl.acetate")
    checks.append(make_check("amyl.acetate NA 数", int(na[:, ja].sum()), 27))

    styles = chem.styles
    codes = np.array([styles.index(s) for s in S])
    eta = eta2(U, codes, len(styles))
    rng = np.random.default_rng(PERM_SEED)
    null = np.array([eta2(U, rng.permutation(codes), len(styles)) for _ in range(N_PERM_ETA)])
    q = np.quantile(eta, [0.5, 0.25, 0.75])
    checks.append(make_check("η² 中位与四分位", q, [0.250, 0.178, 0.347], 0.005))
    checks.append(make_check("η² 置换零分布中位", float(np.median(null)), 0.081, 0.003))
    checks.append(
        make_check("η² 理论值 (S−1)/(n−1)", (len(styles) - 1) / (len(S) - 1), 0.084, 0.0005)
    )
    n_sig = int((eta > np.quantile(null, 0.95, axis=0)).sum())
    checks.append(make_check("超过置换 95% 分位的列数", n_sig, 198, 2))
    ev = np.linalg.eigvalsh(np.cov(_within_resid(U, S).T))[::-1]
    cs = np.cumsum(ev) / ev.sum()
    checks.append(
        make_check(
            "风格内残差累计方差占比 PC1/10/30", [cs[0], cs[9], cs[29]], [0.18, 0.50, 0.72], 0.01
        )
    )

    Xc = chem.X[chem.composite_cols].to_numpy(float)
    mgc = Marg().fit(Xc)
    VC = mgc.transform(np.where(np.isnan(Xc), np.nanmean(Xc, 0), Xc))
    r2 = {}
    for i, c in enumerate(chem.composite_cols):
        pr = cross_val_predict(
            RidgeCV(alphas=ALPHAS), U, VC[:, i], cv=KFold(5, shuffle=True, random_state=SEED)
        )
        y = VC[:, i]
        r2[c] = float(1 - ((pr - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    checks.append(make_check("aroma_hops_sum 岭回归 R²", r2["aroma_hops_sum"], 0.78, 0.01))
    checks.append(make_check("esters_sum 岭回归 R²", r2["esters_sum"], 0.94, 0.01))
    others = [r2[c] for c in ("acids_sum", "sulfur_sum", "kcalperc")]
    checks.append(
        make_check("其余三项 R² 介于两者之间", bool(all(0.77 <= o <= 0.95 for o in others)), True)
    )

    jd = chem.gen_cols.index("X2.3.butanedione")
    na_d = na[:, jd]
    pils_rows = np.where(na_d & (S == SRC_STYLE))[0]
    th = threshold("X2.3.butanedione", dic)
    nonmiss = XG[~na_d, jd]
    pct = float((nonmiss < colmean[jd]).mean() * 100)
    checks.append(
        make_check(
            "双乙酰 NA / Pils NA / 非缺失",
            [int(na_d.sum()), len(pils_rows), len(nonmiss)],
            [9, 2, 241],
        )
    )
    checks.append(make_check("双乙酰列均值", float(colmean[jd]), 2.7458238779128634, 1e-9))
    checks.append(make_check("双乙酰列均值分位(%)", pct, 93.8, 0.5))
    checks.append(make_check("双乙酰列均值 / 阈值", float(colmean[jd] / th), 9.2, 0.1))
    dev = np.abs(x_src[na] - XGf[na])
    checks.append(make_check("NA 单元格数", int(na.sum()), 60))
    checks.append(make_check("NA 往返全部偏离", bool((dev > 0).all()), True))
    checks.append(make_check("NA 往返最大偏差", float(dev.max()), 9.3e-4, 1e-4))
    checks.append(
        make_check(
            "双乙酰 NA 往返值", float(np.unique(np.round(x_src[na_d, jd], 9))[0]), 2.746749, 1e-6
        )
    )

    dt = defect_table(chem, dic)
    dt.to_csv(out_dir / "q3_defect_table.csv", index=False, encoding="utf-8-sig")
    d = dt.set_index("field")
    exp = {
        "DES": {"all": 0.18},
        "DMS": {"all": 0.67, "pils": 0.27, "tripel": 0.89, "kriek": 0.10, "lambic": 0.82},
        "H2S": {"all": 0.92, "pils": 1.00, "tripel": 1.00, "kriek": 0.90, "lambic": 0.91},
        "X2.3.butanedione": {
            "all": 0.45,
            "pils": 0.09,
            "tripel": 0.46,
            "kriek": 0.80,
            "lambic": 0.36,
        },
        "X4VG": {"all": 0.06, "pils": 0, "tripel": 0, "kriek": 0, "lambic": 0},
    }
    for f in ("acetaldehyde", "SO2.mg.L."):
        exp[f] = {k: 0 for k in ("all", "pils", "tripel", "kriek", "lambic")}
    for f, e in exp.items():
        keys = list(e)
        checks.append(
            make_check(
                f"缺陷表均值口径 {f}",
                [float(d.loc[f, f"oav_ge1_mean_{k}"]) for k in keys],
                [e[k] for k in keys],
                0.01,
            )
        )
    med = [d.loc[f, "oav_ge1_median_all"] for f in ("X2.3.butanedione", "DMS", "SO2.mg.L.", "X4VG")]
    checks.append(make_check("缺陷表中位口径全体", med, [0.69, 0.62, 0.004, 0.07], 0.01))
    checks.append(make_check("缺陷表行数", len(dt), 9))
    nm = n_multi(mod)
    checks.append(
        {
            "name": "Q2 模块表多成员模块数（记录，预运行 33）",
            "expected": 33,
            "observed": nm,
            "tol": None,
            "pass": True,
            "note": "使用 Q2 正式标签，不要求等于预运行值",
        }
    )

    np.savez_compressed(
        out_dir / "q3_data.npz",
        gen_cols=np.array(chem.gen_cols),
        chem_cols=np.array(chem.chem_cols),
        style=S,
        beer=chem.beer,
        row_index=np.arange(250),
        na_mask_gen=na,
        x_filled_gen=XGf,
        u_full=U,
        x_src=x_src,
        module_q2=mod,
        eta2=eta,
        composite_r2=np.array([r2[c] for c in chem.composite_cols]),
        diacetyl_na_pils_rows=pils_rows,
        diacetyl_na_pils_beer=chem.beer[pils_rows],
        seed=SEED,
        meta=run_meta(),
    )
    dump_json(
        {
            "checks": checks,
            "n_multi_module_q2": nm,
            "n_modules_q2": int(len(np.unique(mod))),
            "composite_r2": r2,
            "diacetyl_na_pils": {"rows": pils_rows, "beer": chem.beer[pils_rows]},
        },
        out_dir / "q3_checks.json",
    )
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["name"], c["observed"], c["expected"])


if __name__ == "__main__":
    main()
