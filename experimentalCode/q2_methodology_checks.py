# 问题 2 方法论（model/q2_methodology.md）的预检查代码，共 8 个检查，各服务一个或几个设计决策。
# 在项目根目录（或其任一子目录）运行：python experimentalCode/q2_methodology_checks.py，约 4 分钟。
# 检查 4 的 GBM 对照约需 10 分钟，默认不运行；设环境变量 RUN_GBM=1 时运行。
# 只读 code/data/raw、experimentalCode/results/data_audit 与（检查 8）论文 Zenodo 补充工作簿，不写任何文件。各段以 "# %%" 分隔，可在 VS Code / Jupyter 中逐段运行。

# %% 公共部分
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

pd.set_option("display.width", 200)
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "code" / "data" / "raw" / "4.csv").exists())
RAW, AUDIT = ROOT / "code" / "data" / "raw", ROOT / "experimentalCode" / "results" / "data_audit"
X = pd.read_csv(RAW / "1.csv")           # 化学数据
y = pd.read_csv(RAW / "4.csv")           # 品评小组结果（标准化空间）
CHEM, SENS = list(X.columns[3:]), list(y.columns[3:])
R2 = pd.read_csv(RAW / "2.csv", index_col=0)
R2.index, R2.columns = CHEM, CHEM        # 列序与附件 1 一致（审计已逐系数复算）
DIC = pd.read_csv(AUDIT / "chemical_dictionary.csv").set_index("csv1_field")
assert len(CHEM) == 231 and len(SENS) == 50 and (X.beer.values == y.beer.values).all()

COMPOSITE = ["esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum", "kcalperc"]
INTERNAL_STD = DIC.index[DIC.origin_paper == "Internal standard solution"].tolist()
CAND = [c for c in CHEM if c not in COMPOSITE + INTERNAL_STD]
DIMS = [s for s in SENS if s != "barnyard"]     # barnyard 不做归因（memory/barnyard-stale-hops-duplicate.md）
STYLE = y.tasting_category_fine.values
TAU = 0.6
SEED = 0


def tie_pair_counts(z):
    """与 q1_methodology_checks.py 检查 3 相同：每对啤酒在多少维上 z 精确并列。"""
    cnt = {}
    for c in z.columns:
        for idx in pd.Series(np.arange(len(z))).groupby(z[c].round(9).values).groups.values():
            idx = sorted(idx)
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    cnt[(idx[a], idx[b])] = cnt.get((idx[a], idx[b]), 0) + 1
    return cnt


def components(cnt, k, n):
    e = [p for p, v in cnt.items() if v >= k]
    A = coo_matrix((np.ones(len(e)), ([i for i, _ in e], [j for _, j in e])), shape=(n, n))
    return connected_components(A, directed=False)[1]


PANEL = components(tie_pair_counts(y[SENS]), 3, len(y))       # 推断品评团，k=3，单例自成一团
PANEL_IDX = [np.where(PANEL == g)[0] for g in np.unique(PANEL)]


def na_mean(df):
    """问题 1 主设定：化学 NA 按列均值填充（memory/na-below-detection.md）。预检查在全部 250 款上取均值。"""
    return df.fillna(df.mean())


def na_lowest(df):
    """敏感性对照：单化合物 NA 排在最低（并列）。"""
    return df.fillna(-np.inf)


def zrank(M):
    """逐列平均秩后标准化；两列标准化秩的均值内积即 Spearman 系数。"""
    R = np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])])
    R = R - R.mean(0)
    s = R.std(0)
    s[s == 0] = 1
    return R / s


def modules(rabs, tau=TAU):
    """|ρ| 加权图上的平均连接聚类：两组之间平均 |ρ| ≥ τ 才合并。"""
    D = np.clip(1 - rabs, 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform((D + D.T) / 2, checks=False), "average"), 1 - tau, "distance")


def top3_graph(score, mod, k=3):
    """按得分降序取前 k 个，每个模块至多一个。"""
    sel, used = [], set()
    for j in np.argsort(-score):
        if mod[j] not in used:
            sel.append(j)
            used.add(mod[j])
            if len(sel) == k:
                break
    return sel


def top3_forward(RX, ry, mod, k=3, return_score=False):
    """图约束前向选择：每步取与残差秩相关（偏 Spearman）最大的化合物，再把已选化合物从秩中线性剔除。"""
    sel, used, score, r, RXr = [], set(), [], ry.copy(), RX.copy()
    for _ in range(k):
        sc = (RXr.T @ r) / (np.sqrt((RXr ** 2).sum(0) * (r ** 2).sum()) + 1e-12)
        a = np.abs(sc)
        a[[mod[j] in used for j in range(len(a))]] = -1
        j = int(np.argmax(a))
        sel.append(j)
        used.add(mod[j])
        score.append(sc[j])
        Q = np.column_stack([np.ones(len(r))] + [RX[:, s] for s in sel])
        P = Q @ np.linalg.pinv(Q)
        r, RXr = ry - P @ ry, RX - P @ RX
    return (sel, score) if return_score else sel


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def panel_bootstrap(rng):
    """品评团簇 bootstrap：有放回抽品评团，返回啤酒下标。"""
    pick = rng.integers(0, len(PANEL_IDX), len(PANEL_IDX))
    return np.concatenate([PANEL_IDX[p] for p in pick])


XC = na_mean(X[CAND]).values
RX, RY = zrank(XC), zrank(y[SENS].values)
RHO = RX.T @ RY / len(y)                                       # 225 × 50 Spearman
MOD = modules(R2.loc[CAND, CAND].abs().values)
KD = [SENS.index(s) for s in DIMS]
FWD = {k: top3_forward(RX, RY[:, k], MOD, return_score=True) for k in KD}
MAIN = {k: FWD[k][0] for k in KD}                              # 主方法：图约束前向选择
MARG = {k: top3_graph(np.abs(RHO[:, k]), MOD) for k in KD}     # 对照：图约束边际选择
print(f"候选 {len(CAND)} 个（231 − 综合指标 {len(COMPOSITE)} − 内标 {INTERNAL_STD}）；归因维度 {len(DIMS)} 个；"
      f"推断品评团 {len(PANEL_IDX)} 个（含单例）")


# %% 检查 1
# 检查 1 | 服务决策 D1（候选集合）与 D2（共表达图的阈值与模块）
from sklearn.metrics import adjusted_rand_score

err = np.abs(X[CHEM].corr(method="spearman").values - R2.values).max()
print(f"附件 2 与附件 1 逐对 Spearman 的最大差: {err:.1e}（bootstrap 可在重抽样本上重建该图）")

print("\n综合指标与其最相关单列:")
for c in COMPOSITE:
    r = R2[c].drop(COMPOSITE).abs()
    print(f"  {c:15s} → {r.idxmax():20s} |ρ| = {r.max():.2f}")
print(f"内标: {INTERNAL_STD}，附件 8 来源 = {DIC.loc[INTERNAL_STD[0], 'origin_paper']}，"
      f"附件 1 最小值 {X[INTERNAL_STD[0]].min():.2f}（无底值）")

A = R2.loc[CAND, CAND].abs().values
iu = np.triu_indices(len(CAND), 1)
q = np.quantile(A[iu], [.5, .9, .99])
neg = (R2.loc[CAND, CAND].values[iu] < 0) & (A[iu] >= .5)
print(f"\n候选间 |ρ|: 中位 {q[0]:.2f}，90% 分位 {q[1]:.2f}，99% 分位 {q[2]:.2f}；|ρ|≥0.5 的边中负相关 {neg.sum()} / {(A[iu] >= .5).sum()}")

origin = DIC.loc[CAND, "origin_paper"].fillna("NA").values
rng = np.random.default_rng(SEED)
boots = [panel_bootstrap(rng) for _ in range(100)]
boot_R = [X[CAND].iloc[ii].corr(method="spearman").abs().fillna(0).values for ii in boots]
for tau in (.5, .6, .7):
    m = modules(A, tau)
    s = pd.Series(m).value_counts()
    multi = s[s > 1].index
    purity = np.mean([pd.Series(origin[m == g]).value_counts(normalize=True).iloc[0] for g in multi])
    ari = [adjusted_rand_score(m, modules(Rb, tau)) for Rb in boot_R]
    print(f"τ={tau}: 边 {int((A[iu] >= tau).sum())} 条；模块 {len(s)} 个，其中多成员 {len(multi)} 个覆盖 {s[multi].sum()} 个化合物，"
          f"最大 {s.max()}；多成员模块来源纯度 {purity:.2f}；bootstrap ARI 中位 {np.median(ari):.2f} "
          f"[{np.quantile(ari, .05):.2f}, {np.quantile(ari, .95):.2f}]")
cc = connected_components(A >= TAU, directed=False)[1]
print(f"对照：τ={TAU} 的连通分量最大 {pd.Series(cc).value_counts().max()} 个化合物（链式连接），因此模块不用连通分量")
M = pd.Series(MOD, index=CAND)
for c in ["ethanol..v.v.", "isopentyl_acetate", "ethyl_octanoate", "color.EBC.", "linalool"]:
    print(f"  {c} 所在模块: {[x for x in M[M == M[c]].index]}")


# %% 检查 2
# 检查 2 | 服务决策 D3（关联度量）与 D4（显著性门槛）
rng = np.random.default_rng(SEED)
null = np.array([np.abs(RX.T @ RY[rng.permutation(len(y))] / len(y)).max(0) for _ in range(1000)])
q95 = np.quantile(null, .95, axis=0)
mx = np.abs(RHO).max(0)
print(f"每维 max|ρ| 的置换 95% 分位: 中位 {np.median(q95):.3f}；Top-1 超过该门槛的维度 {int((mx[KD] > q95[KD]).sum())} / {len(KD)}")
print("未超过的维度:", [SENS[k] for k in KD if mx[k] <= q95[k]])
pos3 = np.array([[abs(s) > q95[k] for s in FWD[k][1]] for k in KD])
print(f"主方法 Top-3 的选择得分（第 1 位为 ρ，第 2、3 位为偏 ρ）超过该门槛的个数（按位置）: {pos3.sum(0).tolist()} / {len(KD)}")

raw = {k: list(np.argsort(-np.abs(RHO[:, k]))[:3]) for k in KD}
dup = [SENS[k] for k in KD if len(set(MOD[raw[k]])) < 3]
print(f"\n不加图约束的边际 Top-3 中有 ≥2 个同模块化合物的维度: {len(dup)} / {len(KD)}")
print("  例:", {SENS[k]: [CAND[j] for j in raw[k]] for k in [SENS.index(s) for s in dup[:3]]})


# %% 检查 3
# 检查 3 | 服务决策 D5（边际选择还是前向选择）与 D2（τ 对稳定性的影响）
for tau in (.5, .6, .7):
    m0 = modules(A, tau)
    full = {"边际": {k: top3_graph(np.abs(RHO[:, k]), m0) for k in KD},
            "前向": {k: top3_forward(RX, RY[:, k], m0) for k in KD}}
    stat = {r: [np.zeros((len(KD), 3)), np.zeros((len(KD), 3)), []] for r in full}
    for ii, Rb in zip(boots, boot_R):
        mb = modules(Rb, tau)
        RXb, RYb = zrank(XC[ii]), zrank(y[SENS].values[ii])
        Cb = RXb.T @ RYb / len(ii)
        for i, k in enumerate(KD):
            for rule, sel in [("边际", top3_graph(np.abs(Cb[:, k]), mb)), ("前向", top3_forward(RXb, RYb[:, k], mb))]:
                f = full[rule][k]
                stat[rule][0][i] += [j in sel for j in f]
                stat[rule][1][i] += [m0[j] in set(m0[sel]) for j in f]
                stat[rule][2].append(jac(sel, f))
    for rule, (pc, pm, jj) in stat.items():
        pc, pm = pc / len(boots), pm / len(boots)
        print(f"τ={tau} {rule}: 化合物入选频率（按位置）{pc.mean(0).round(2).tolist()}，模块 {pm.mean(0).round(2).tolist()}；"
              f"bootstrap 与全样本 Top-3 的 Jaccard 均值 {np.mean(jj):.3f}；Top-1 频率 ≥0.6 的维度 {int((pc[:, 0] >= .6).sum())}")
        if tau == TAU and rule == "前向":
            band = pd.cut(pc.ravel(), [-.01, 1 / 3, .6, 1.01], labels=["<1/3", "1/3–0.6", "≥0.6"]).value_counts()
            print(f"  主方法 147 项的化合物入选频率分档: {band.to_dict()}")


# %% 检查 4
# 检查 4 | 服务决策 D6（对照方法与方法间 Jaccard）
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

warnings.filterwarnings("ignore")
D = pd.get_dummies(STYLE).values.astype(float)
P = D @ np.linalg.pinv(D)
RXs, RYs = RX - P @ RX, RY - P @ RY
RHO_S = (RXs / RXs.std(0).clip(1e-9)).T @ (RYs / RYs.std(0)) / len(y)   # 风格内（剔除风格均值后的秩相关）

gkf = list(GroupKFold(5).split(RX, groups=PANEL))
others = {"不加图约束的边际": raw,
          "图约束边际": MARG,
          "ElasticNet": {k: top3_graph(np.abs(ElasticNetCV(l1_ratio=.5, cv=gkf, alphas=30, max_iter=5000)
                                              .fit(RX, RY[:, k]).coef_) + 1e-9 * np.abs(RHO[:, k]), MOD) for k in KD},
          "风格内前向": {k: top3_forward(RXs, RYs[:, k], MOD) for k in KD},
          "敏感性：NA 最低位编码": {k: top3_forward(zrank(na_lowest(X[CAND]).values), RY[:, k], MOD) for k in KD}}
if os.environ.get("RUN_GBM") == "1":
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    Xg = XC                                            # 与主方法相同的列均值填充（问题 1 主设定）
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xg, STYLE, PANEL))
    gbm = {}
    for k in KD:
        imp = np.zeros(len(CAND))
        for tr, te in splits:
            mdl = HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=20, learning_rate=.05, max_iter=200,
                                                random_state=SEED).fit(Xg[tr], y[SENS[k]].values[tr])
            imp += permutation_importance(mdl, Xg[te], y[SENS[k]].values[te], n_repeats=3,
                                          random_state=SEED, n_jobs=-1).importances_mean
        gbm[k] = top3_graph(imp, MOD)
    others["GBM 置换重要度"] = gbm

rng = np.random.default_rng(SEED)
rand_j = np.mean([jac(rng.choice(len(CAND), 3, replace=False), rng.choice(len(CAND), 3, replace=False))
                  for _ in range(20000)])
print(f"随机 3 元集合之间的 Jaccard 期望: {rand_j:.3f}")
for name, o in others.items():
    print(f"主方法 vs {name}: 化合物 Jaccard {np.mean([jac(MAIN[k], o[k]) for k in KD]):.3f}，"
          f"模块 Jaccard {np.mean([jac(MOD[MAIN[k]], MOD[o[k]]) for k in KD]):.3f}，Top-1 相同 {np.mean([MAIN[k][0] == o[k][0] for k in KD]):.2f}")
print(f"图约束边际 vs ElasticNet: {np.mean([jac(MARG[k], others['ElasticNet'][k]) for k in KD]):.3f}")


# %% 检查 5
# 检查 5 | 服务决策 D7（风格内保持作为合理性证据）
eta = pd.Series([1 - ((RY[:, k] - pd.Series(RY[:, k]).groupby(STYLE).transform("mean")) ** 2).sum() / len(y)
                 for k in range(len(SENS))], index=SENS)
print(f"风格解释的感官秩方差 η²: 中位 {eta[DIMS].median():.2f}，范围 {eta[DIMS].min():.2f}–{eta[DIMS].max():.2f}")
rows = [(SENS[k], p + 1, CAND[j], RHO[j, k], RHO_S[j, k]) for k in KD for p, j in enumerate(MAIN[k])]
W = pd.DataFrame(rows, columns=["维度", "位置", "化合物", "ρ", "ρ_风格内"])
W["比值"] = W["ρ_风格内"] / W["ρ"]
print(f"主方法 Top-3 共 {len(W)} 项：风格内同号 {np.mean(np.sign(W['ρ']) == np.sign(W['ρ_风格内'])):.2f}，"
      f"风格内/合并 比值中位 {W['比值'].median():.2f}；风格内 |ρ| < 0.1 的 {int((W['ρ_风格内'].abs() < .1).sum())} 项:")
print(W[W["ρ_风格内"].abs() < .1].round(2).to_string(index=False))


# %% 检查 6
# 检查 6 | 服务决策 D8（化学合理性）——只输出阈值证据与通用轴；化学解释按结果事后撰写，不使用预期表（model/q2_methodology.md §6）
thr = DIC.loc[CAND][DIC.loc[CAND, "threshold_mean"].notna()]
in_top = sorted({CAND[j] for k in KD for j in MAIN[k]} & set(thr.index))
print(f"\n主方法 Top-3 中有阈值的化合物 {len(in_top)} 个，OAV≥1 的啤酒比例:")
for c in in_top:
    r = thr.loc[c]
    conc = 10 ** X[c] if str(r.transform_interpretation).startswith("log10") else X[c]
    print(f"  {c:20s} {(conc / r.threshold_mean >= 1).mean():.2f}")

breadth = pd.Series([CAND[j] for k in KD for j in MAIN[k]]).value_counts()
print(f"\n在 ≥5 个维度的 Top-3 中出现的化合物: {breadth[breadth >= 5].to_dict()}")


# %% 检查 7
# 检查 7 | 服务决策 D5（边际选择还是前向选择）——模拟还原：在真实化学矩阵上植入 3 个已知关键化合物，看哪种规则能找回
pan = np.unique(PANEL, return_inverse=True)[1]
for wname, w in [("1, 0.75, 0.5", np.array([1, .75, .5])), ("1, 1, 1", np.array([1., 1, 1]))]:
    rng = np.random.default_rng(SEED)
    for target in (.2, .3, .45):
        rec = {"图约束边际": np.zeros(3), "图约束前向": np.zeros(3)}
        n_rep = 200
        for _ in range(n_rep):
            while True:                                      # 3 个植入化合物来自不同模块
                js = rng.choice(len(CAND), 3, replace=False)
                if len(set(MOD[js])) == 3:
                    break
            S = RX[:, js] @ w
            S = (S - S.mean()) / S.std()
            a = min(.99, target / np.corrcoef(S, RX[:, js[0]])[0, 1])   # 使第 1 个植入化合物的 ρ ≈ target
            e = rng.normal(0, 1, len(y)) + rng.normal(0, .15, pan.max() + 1)[pan]   # 含品评团偏移
            e = (e - e.mean()) / e.std()
            ry = zrank((a * S + np.sqrt(1 - a ** 2) * e)[:, None])[:, 0]
            rec["图约束边际"] += [j in top3_graph(np.abs(RX.T @ ry / len(y)), MOD) for j in js]
            rec["图约束前向"] += [j in top3_forward(RX, ry, MOD) for j in js]
        print(f"权重 {wname}，第 1 个植入化合物 ρ≈{target}: " +
              "；".join(f"{r} 找回率 {np.round(v / n_rep, 2).tolist()}" for r, v in rec.items()))


# %% 检查 8
# 检查 8 | 服务决策 D9（外部因果核验）——Zenodo 加标实验，仅报告；需要 ZENODO_XLSX（见 q1_supplement_checks.py）
ZEN = Path(os.environ.get("ZENODO_XLSX", Path(os.environ.get("TEMP", ".")) / "zenodo_10653704" / "Supplemental Files and Figure source files.xlsx"))
if not ZEN.exists():
    print(f"跳过：找不到 Zenodo 工作簿 {ZEN}")
else:
    ATT = ["Malt", "Sweetness", "Hops", "Bitterness", "Esters", "Fullness (body)", "Alcohol (warming)", "Overall appreciation"]
    V = pd.concat([pd.read_excel(ZEN, sheet_name=f"Validation Results {k}") for k in (1, 2, 3)], ignore_index=True)
    V["date"] = pd.to_datetime(V.Time).dt.date
    V = V.drop_duplicates(subset=["date", "Name", "Beer"] + ATT)        # Taster 25 在 07-13 重复提交一次
    det = pd.read_excel(ZEN, sheet_name="Validation Details")
    det = det[det.SetName.notna()].copy()
    det["date"] = pd.to_datetime(det.Date).dt.date
    V = V.merge(det[["date", "Code", "BeerStyle", "SampleName", "SetName"]], left_on=["date", "Beer"],
                right_on=["date", "Code"], how="left")
    assert V.SampleName.notna().all()
    wide = V.pivot_table(index=["date", "Name", "BeerStyle", "SetName"], columns=V.SampleName.str.strip(), values=ATT)
    out = {}
    for key, g in wide.groupby(level=["BeerStyle", "SetName"]):
        d = pd.DataFrame({a: g[(a, "Spiked")] - g[(a, "Control")] for a in ATT}).dropna()
        out[f"{key[0]} | {key[1]} | n={len(d)}"] = (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))).round(1).astype(str) \
            .radd(d.mean().round(2).astype(str) + " (t=").add(")")
    print("加标杯 − 对照杯 的同人配对差，0–6 分（t 值）:")
    print(pd.DataFrame(out).to_string())
