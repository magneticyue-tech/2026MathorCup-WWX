# 问题 3 方法论（model/q3_methodology.md）的预检查代码，共 7 个检查，各服务一个或几个设计决策。
# 在项目根目录（或其任一子目录）运行：python experimentalCode/q3_methodology_checks.py，约 11 分钟（检查 2 约 7 分钟）。
# 只读 code/data/raw 与 experimentalCode/results/data_audit，不写任何文件。各段以 "# %%" 分隔，可在 VS Code / Jupyter 中逐段运行；
# 检查 3、6、7 使用检查 2 选出的超参数（未运行检查 2 时取下方默认值）。
# 检查 6、7 的感官画像用固定超参数的 HistGBR 代替问题 1 的正式模型，正式运行改用问题 1 的全数据重拟合模型与选定映射。

# %% 公共部分
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, cross_val_score, cross_val_predict

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "code" / "data" / "raw" / "4.csv").exists())
RAW, AUDIT = ROOT / "code" / "data" / "raw", ROOT / "experimentalCode" / "results" / "data_audit"
X = pd.read_csv(RAW / "1.csv")           # 化学数据
y = pd.read_csv(RAW / "4.csv")           # 品评小组结果（标准化空间）
CHEM, SENS = list(X.columns[3:]), list(y.columns[3:])
R2 = pd.read_csv(RAW / "2.csv", index_col=0)
R2.index, R2.columns = CHEM, CHEM        # 列序与附件 1 一致（问题 2 检查 1 已逐系数复算）
DIC = pd.read_csv(AUDIT / "chemical_dictionary.csv", encoding="utf-8-sig").set_index("csv1_field")
assert len(CHEM) == 231 and len(SENS) == 50 and (X.beer.values == y.beer.values).all()

COMPOSITE = ["esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum", "kcalperc"]
INTERNAL_STD = DIC.index[DIC.origin_paper == "Internal standard solution"].tolist()
GEN = [c for c in CHEM if c not in COMPOSITE + INTERNAL_STD]       # 生成列，与问题 2 的候选集合相同
S = X.tasting_category_fine.values
STYLES = sorted(set(S))
DEFECT_SENS = ["dms", "metallic", "X4vg", "t2n", "diacetyl", "stale_hops"]                # 附件 3 模板的缺陷组
DEFECT_CHEM = list(DIC.index[DIC.off_flavors.notna()]) + ["X4VG", "X.E..2.nonenal"]     # 附件 8 有缺陷描述 + 题面点名
XG = X[GEN].values
TAU, SEED = 0.6, 0
K_STAR, KAP_STAR = 15, 2.0               # 检查 2 会按留出似然重设


def is_log(c):
    return str(DIC.loc[c, "transform_interpretation"]).startswith("log10")


def modules(rabs, tau=TAU):
    """与问题 2 相同：|ρ| 上的平均连接聚类，在高度 1−τ 处切开。"""
    Dm = np.clip(1 - rabs, 0, None)
    np.fill_diagonal(Dm, 0)
    return fcluster(linkage(squareform((Dm + Dm.T) / 2, checks=False), "average"), 1 - tau, "distance")


MOD = modules(R2.loc[GEN, GEN].abs().values)


class Marg:
    """逐列秩变换。训练折的每个取值块（含下限、均值填充造成的并列）取标准正态在该块概率区间上的条件期望 e；
    新值在相邻块之间线性插值。反变换为原子感知的逆经验分布：并列块的整个概率区间映回该块取值，
    单值块的结点放在 Φ(e) 处，使训练值往返变换后不变；单值点之间线性插值，不外推。"""

    def fit(self, M):
        self.blocks = []
        for j in range(M.shape[1]):
            x = M[:, j][~np.isnan(M[:, j])]
            vals, cnt = np.unique(x, return_counts=True)
            hi = np.cumsum(cnt) / len(x)
            lo = hi - cnt / len(x)
            e = (norm.pdf(norm.ppf(lo)) - norm.pdf(norm.ppf(hi))) / (hi - lo)
            kp, kv = [], []
            for v, c, l, h, ek in zip(vals, cnt, lo, hi, e):
                if c >= 2:
                    kp += [l + 1e-9, h - 1e-9]
                    kv += [v, v]
                else:
                    kp.append(norm.cdf(ek))
                    kv.append(v)
            self.blocks.append((vals, e, np.array(kp), np.array(kv)))
        return self

    def transform(self, M):
        return np.column_stack([np.interp(M[:, j], b[0], b[1]) for j, b in enumerate(self.blocks)])

    def inverse(self, U):
        P = norm.cdf(U)
        return np.column_stack([np.interp(P[:, j], b[2], b[3]) for j, b in enumerate(self.blocks)])


def within_resid(U, s):
    R = U.copy()
    for st in set(s):
        R[s == st] -= U[s == st].mean(0)
    return R


class Gen:
    """风格条件高斯 copula：潜空间风格均值向全局均值收缩（κ），风格内协方差在各风格间合并共享。
    kind = diag | lw | fa | famod；famod 为 k 因子 + 模块块特异协方差（模块在训练折内按 τ 重算）。"""

    def __init__(self, kappa=KAP_STAR, kind="famod", k=K_STAR, global_only=False):
        self.kappa, self.kind, self.k, self.global_only = kappa, kind, k, global_only

    def fit(self, Xtr, s):
        self.mg = Marg().fit(Xtr)
        self.colmean = np.nanmean(Xtr, 0)
        Xf = np.where(np.isnan(Xtr), self.colmean, Xtr)
        U = self.mg.transform(Xf)
        self.U, self.s = U, s
        g = U.mean(0)
        if self.global_only:
            self.mu = {st: g for st in STYLES}
            R, dof = U - g, len(U) - 1
        else:
            self.mu = {}
            for st in STYLES:
                n = (s == st).sum()
                self.mu[st] = g if n == 0 else (n * U[s == st].mean(0) + self.kappa * g) / (n + self.kappa)
            R, dof = within_resid(U, s), len(U) - len(set(s))
        R = R * np.sqrt(len(U) / dof)
        if self.kind == "diag":
            self.C = np.diag(R.var(0))
        elif self.kind == "lw":
            self.C = LedoitWolf(assume_centered=True).fit(R).covariance_
        else:
            fa = FactorAnalysis(n_components=self.k, random_state=SEED).fit(R)
            if self.kind == "fa":
                self.C = fa.get_covariance()
            else:
                LL = fa.components_.T @ fa.components_
                Res = R.T @ R / len(R) - LL
                mod = modules(pd.DataFrame(Xf).corr("spearman").abs().fillna(0).values)
                B = np.diag(np.diag(Res).clip(1e-3))
                for m in np.unique(mod):
                    idx = np.where(mod == m)[0]
                    if len(idx) > 1:
                        B[np.ix_(idx, idx)] = Res[np.ix_(idx, idx)]
                w, V = np.linalg.eigh(B)
                self.C = LL + (V * w.clip(1e-3)) @ V.T
        self.L = np.linalg.cholesky(self.C + 1e-9 * np.eye(len(self.C)))
        return self

    def latent(self, Xn):
        return self.mg.transform(np.where(np.isnan(Xn), self.colmean, Xn))

    def ll(self, Xte, ste):
        U = self.latent(Xte)
        M = np.array([self.mu[st] for st in ste])
        Z = np.linalg.solve(self.L, (U - M).T)
        return -0.5 * (Z ** 2).sum(0) - np.log(np.diag(self.L)).sum() - 0.5 * len(self.C) * np.log(2 * np.pi)

    def sample_latent(self, st, n, rng):
        return self.mu[st] + rng.standard_normal((n, len(self.C))) @ self.L.T

    def sample(self, st, n, rng):
        return self.mg.inverse(self.sample_latent(st, n, rng))


def folds(n_rep=2):
    """按风格分层、以单款啤酒为单位的 5 折；最小风格 3 款，部分折不含该风格。"""
    for rep in range(n_rep):
        yield from StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(XG, S)


print(f"生成列 {len(GEN)} 个（231 − 综合指标 {len(COMPOSITE)} − 内标 {INTERNAL_STD}）；风格 {len(STYLES)} 个，"
      f"样本量 {pd.Series(S).value_counts().min()}–{pd.Series(S).value_counts().max()}")


# %% 检查 1
# 检查 1 | 服务决策 D1（在秩变换后的潜空间建模）与 D2（风格内协方差是否必要）
XGf = np.where(np.isnan(XG), np.nanmean(XG, 0), XG)
mins = XGf.min(0)
tied = (XGf == mins).sum(0)
floor = np.isclose(XGf, -1.81781854)
print(f"最小值并列（≥2 款）的列 {int((tied >= 2).sum())} 个，共 {int(tied[tied >= 2].sum())} 个单元格；"
      f"其中共同下限 −1.81781854 出现在 {int(floor.any(0).sum())} 列、{int(floor.sum())} 个单元格")
fr = pd.Series((XGf == mins).mean(0))
print(f"每列最小值占比：中位 {fr.median():.2f}，90% 分位 {fr.quantile(.9):.2f}，最大 {fr.max():.2f}；超过 50% 的列 {int((fr > .5).sum())} 个")
j = GEN.index("X2.methyltetrahydrofuran.3.one")
print(f"重尾示例 X2.methyltetrahydrofuran.3.one：中位 {np.nanmedian(XG[:, j]):.3f}，最大 {np.nanmax(XG[:, j]):.0f}")

mg = Marg().fit(XG)
U = mg.transform(XGf)
v = U.var(0)
print(f"潜变量方差：最小 {v.min():.2f}，10% 分位 {np.quantile(v, .1):.2f}，中位 {np.median(v):.2f}")
obs = ~np.isnan(XG)
print(f"往返变换 Q(Φ(φ(x))) 与 x 的最大差（非缺失训练值）: {np.abs(mg.inverse(U)[obs] - XG[obs]).max():.1e}")
ja = GEN.index("amyl.acetate")
print(f"amyl.acetate：NA {int(np.isnan(XG[:, ja]).sum())} 个；若用填充后数据构造逆分布，列均值处的原子占 {np.isnan(XG[:, ja]).mean():.1%}")


def eta2(U, s):
    out = []
    for jj in range(U.shape[1]):
        u = U[:, jj]
        gr = pd.Series(u).groupby(s)
        out.append((gr.count() * (gr.mean() - u.mean()) ** 2).sum() / ((u - u.mean()) ** 2).sum())
    return np.array(out)


eta = eta2(U, S)
rng = np.random.default_rng(SEED)
null = np.array([eta2(U, rng.permutation(S)) for _ in range(200)])
print(f"\n风格解释的潜变量方差 η²：中位 {np.median(eta):.3f}，四分位 {np.quantile(eta, .25):.3f}–{np.quantile(eta, .75):.3f}；"
      f"置换零分布中位 {np.median(null):.3f}（理论 (S−1)/(n−1) = {(len(STYLES) - 1) / (len(S) - 1):.3f}）；"
      f"超过置换 95% 分位的列 {int((eta > np.quantile(null, .95, axis=0)).sum())} / {len(GEN)}")
ev = np.linalg.eigvalsh(np.cov(within_resid(U, S).T))[::-1]
cs = np.cumsum(ev) / ev.sum()
print("合并风格内残差的累计方差占比（前 1、2、5、10、20、30 个主成分）:", [round(float(cs[k - 1]), 2) for k in (1, 2, 5, 10, 20, 30)])


# %% 检查 2
# 检查 2 | 服务决策 D3（协方差结构、因子数 k 与收缩强度 κ）
t0 = time.time()
CONFIGS = [("全局 LW", dict(global_only=True, kind="lw")), ("全局 因子15", dict(global_only=True, kind="fa", k=15))]
CONFIGS += [(f"{kind} κ={kap}", dict(kind=kind, kappa=kap)) for kind in ("diag", "lw") for kap in (0, 2, 5, 10, 20)]
CONFIGS += [(f"fa k={k} κ={kap}", dict(kind="fa", k=k, kappa=kap)) for k in (10, 15, 20, 25) for kap in (1, 2, 5, 10, 20)]
CONFIGS += [(f"fa k={k} κ=5", dict(kind="fa", k=k, kappa=5)) for k in (2, 5, 30)]
CONFIGS += [(f"famod k={k} κ={kap}", dict(kind="famod", k=k, kappa=kap)) for k in (10, 15, 20) for kap in (1, 2, 5)]
rows = []
for tr, te in folds():
    for name, cfg in CONFIGS:
        rows.append((name, Gen(**cfg).fit(XG[tr], S[tr]).ll(XG[te], S[te]).mean()))
LLT = pd.DataFrame(rows, columns=["设定", "ll"]).groupby("设定").ll.mean().sort_values(ascending=False)
print("留出条件对数似然 log p(x|s)（nat/款，5 折 × 2 次重复平均）:")
print(LLT.round(1).to_string())
best = LLT.drop([c for c in LLT.index if c.startswith("全局")]).idxmax()
cfg = dict(CONFIGS)[best]
K_STAR, KAP_STAR = cfg["k"], cfg["kappa"]
print(f"主模型比全局因子模型高 {LLT[best] - LLT['全局 因子15']:.1f} nat；纯因子 k={K_STAR} κ={KAP_STAR} 比全局因子模型低 {LLT['全局 因子15'] - LLT[f'fa k={K_STAR} κ={KAP_STAR}']:.1f} nat")
print(f"选中: {best}；H(s) = {-(pd.Series(S).value_counts(normalize=True) * np.log(pd.Series(S).value_counts(normalize=True))).sum():.2f} nat")

acc = []
for tr, te in folds():
    gm = Gen().fit(XG[tr], S[tr])
    LLs = np.column_stack([gm.ll(XG[te], [st] * len(te)) for st in STYLES])
    prior = np.log(pd.Series(S[tr]).value_counts(normalize=True).reindex(STYLES).fillna(1e-9).values)
    acc.append((np.array(STYLES)[(LLs + prior).argmax(1)] == S[te]).mean())
print(f"主模型按贝叶斯公式导出的风格分类器，留出准确率 {np.mean(acc):.3f}；多数类基线 {pd.Series(S).value_counts(normalize=True).max():.3f}")
print(f"用时 {time.time() - t0:.0f} s")


# %% 检查 3
# 检查 3 | 服务决策 D4（主模型是否优于对照生成器）与 D5（评价指标）
t0 = time.time()
NAMES = ["主模型", "纯因子", "G0 无风格", "G1 复制", "G2 独立边际", "G3 风格内插值"]
iu = np.triu_indices(len(GEN), 1)
inmod = (MOD[:, None] == MOD[None, :])[iu]
res = {n: {k: [] for k in ["auc", "dcr", "trts", "tstr", "floor", "floor_row", "corr", "inmod"]} for n in NAMES}
trtr, real_inmod, real_floor_row, disp = [], [], [], []
rng = np.random.default_rng(SEED)
for rep in range(2):
    real, gen = [], {n: [] for n in NAMES}
    dcr_real, dcr_gen = [], {n: [] for n in NAMES}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(XG, S):
        Xtr, Str, Xte, Ste = XG[tr], S[tr], XG[te], S[te]
        main = Gen().fit(Xtr, Str)
        g0 = Gen(global_only=True).fit(Xtr, Str)
        gfa = Gen(kind="fa").fit(Xtr, Str)
        Xtr_f, Xte_f = np.where(np.isnan(Xtr), main.colmean, Xtr), np.where(np.isnan(Xte), main.colmean, Xte)

        def draw(name, st, n):
            idx = np.where(Str == st)[0]
            if name == "主模型":
                return main.sample(st, n, rng)
            if name == "纯因子":
                return gfa.sample(st, n, rng)
            if name == "G0 无风格":
                return g0.sample(st, n, rng)
            if name == "G1 复制":
                return Xtr_f[rng.choice(idx, n)]
            if name == "G2 独立边际":
                return np.column_stack([Xtr_f[rng.choice(idx, n), jj] for jj in range(Xtr.shape[1])])
            a, b, lam = rng.choice(idx, n), rng.choice(idx, n), rng.uniform(size=(n, 1))
            return main.mg.inverse(lam * main.U[a] + (1 - lam) * main.U[b])

        rf = RandomForestClassifier(500, random_state=SEED, n_jobs=-1).fit(Xtr_f, Str)
        trtr.append((rf.predict(Xte_f) == Ste).mean())
        real.append(Xte_f)
        Ute = main.latent(Xte)
        dcr_real += list(np.sqrt(((Ute[:, None] - main.U[None]) ** 2).sum(-1)).min(1))
        disp += [(st, ((Ute[i] - main.mu[st]) ** 2 / np.diag(main.C)).mean()) for i, st in enumerate(Ste)]
        sts = [st for st in STYLES if (Str == st).sum() > 0]
        for n in NAMES:
            G = np.vstack([draw(n, st, 1) for st in Ste])
            gen[n].append(G)
            res[n]["trts"].append((rf.predict(G) == Ste).mean())
            Ug = main.latent(G)
            dcr_gen[n] += list(np.sqrt(((Ug[:, None] - main.U[None]) ** 2).sum(-1)).min(1))
            Gs = np.vstack([draw(n, st, (Str == st).sum()) for st in sts])
            Ss = np.concatenate([[st] * (Str == st).sum() for st in sts])
            res[n]["tstr"].append((RandomForestClassifier(500, random_state=SEED, n_jobs=-1).fit(Gs, Ss).predict(Xte_f) == Ste).mean())
    # C2ST 在每次重复内计算：两次重复合并会使同一真实啤酒出现两次，分类器可记住重复样本
    Rr = np.vstack(real)
    cr_real = pd.DataFrame(Rr).corr("spearman").values
    real_inmod.append(cr_real[iu][inmod].mean())
    fl_real = np.isclose(Rr, mins)
    has_floor = fl_real.mean(0) > 0.02
    real_floor_row.append(np.median(fl_real[:, has_floor].sum(1)))
    for n in NAMES:
        G = np.vstack(gen[n])
        Xc, yc = np.vstack([Rr, G]), np.r_[np.zeros(len(Rr)), np.ones(len(G))]
        res[n]["auc"].append(cross_val_score(RandomForestClassifier(500, random_state=SEED, n_jobs=-1), Xc, yc,
                                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc").mean())
        res[n]["dcr"].append(np.median(dcr_gen[n]) / np.median(dcr_real))
        fl_gen = np.isclose(G, mins)
        res[n]["floor"].append(np.abs(fl_gen.mean(0) - fl_real.mean(0))[has_floor].mean())
        res[n]["floor_row"].append(np.median(fl_gen[:, has_floor].sum(1)))
        cr = pd.DataFrame(G).corr("spearman").fillna(0).values
        res[n]["corr"].append(np.abs(cr - cr_real)[iu].mean())
        res[n]["inmod"].append(cr[iu][inmod].mean())
print(f"主模型 famod k={K_STAR} κ={KAP_STAR}。TRTR 风格判别准确率 {np.mean(trtr):.3f}；"
      f"真实留出：模块内平均 ρ {np.mean(real_inmod):.3f}，每款啤酒下限列数中位 {np.mean(real_floor_row):.1f}")
T3 = pd.DataFrame([dict(生成器=n, C2ST_AUC=np.mean(o["auc"]), DCR比=np.mean(o["dcr"]), TRTS=np.mean(o["trts"]), TSTR=np.mean(o["tstr"]),
                        下限比例误差=np.mean(o["floor"]), 每款下限列数=np.mean(o["floor_row"]), 相关误差=np.mean(o["corr"]),
                        模块内ρ=np.mean(o["inmod"])) for n, o in res.items()])
print(T3.round(3).to_string(index=False))
print("注：G1、G3 由训练啤酒复制或插值得到，生成池内有近重复样本，C2ST 的 AUC 低于 0.5 不表示更像真实啤酒；TRTS 对它们也不可比（判别器见过其来源）")
Dp = pd.DataFrame(disp, columns=["风格", "r"]).groupby("风格").r.agg(["mean", "count"])
Dp["相对中位"] = Dp["mean"] / Dp["mean"].median()
print("\n留出啤酒的标准化残差平方均值（按风格；相对 22 风格中位数）:")
print(Dp.sort_values("相对中位", ascending=False).round(2).T.to_string())
print(f"用时 {time.time() - t0:.0f} s")


# %% 检查 4
# 检查 4 | 服务决策 D6（综合指标由成分推导）
gfull = Gen().fit(XG, S)
mgC = Marg().fit(X[COMPOSITE].values)
UC = mgC.transform(X[COMPOSITE].fillna(X[COMPOSITE].mean()).values)
ALPHAS = np.logspace(-1, 4, 30)
for i, c in enumerate(COMPOSITE):
    pr = cross_val_predict(RidgeCV(alphas=ALPHAS), gfull.U, UC[:, i], cv=KFold(5, shuffle=True, random_state=SEED))
    print(f"{c:15s} 由 225 个生成列潜变量岭回归，5 折 R² = {1 - ((pr - UC[:, i]) ** 2).sum() / ((UC[:, i] - UC[:, i].mean()) ** 2).sum():.3f}")
RIDGE = [RidgeCV(alphas=ALPHAS).fit(gfull.U, UC[:, i]) for i in range(len(COMPOSITE))]
IS_MED = X[INTERNAL_STD[0]].median()


def full_profile(Ug):
    """潜变量 → 问题 1 模型需要的 231 列：225 个生成列反变换，综合指标由岭回归派生，内标取中位数。"""
    out = pd.DataFrame(gfull.mg.inverse(Ug), columns=GEN)
    xc = mgC.inverse(np.column_stack([r.predict(Ug) for r in RIDGE]))
    for i, c in enumerate(COMPOSITE):
        out[c] = xc[:, i]
    out[INTERNAL_STD[0]] = IS_MED
    return out[CHEM].values


# %% 检查 5
# 检查 5 | 服务决策 D7（化学层缺陷集合与阈值口径）
rows = []
for c in DEFECT_CHEM:
    conc = 10 ** X[c] if is_log(c) else X[c]
    r = dict(字段=c, 化合物=DIC.loc[c, "chemical_name"], 尺度="log10" if is_log(c) else "线性", 单位=DIC.loc[c, "unit"],
             阈值均值=DIC.loc[c, "threshold_mean"], 阈值中位=DIC.loc[c, "threshold_median"], NA=int(X[c].isna().sum()),
             NA_Pils=int(X[c][S == "Pils/Lager"].isna().sum()))
    for lab in ["threshold_mean", "threshold_median"]:
        th = DIC.loc[c, lab]
        if not pd.isna(th):
            o = conc / th >= 1
            for st, key in [(None, "全体"), ("Pils/Lager", "Pils"), ("Tripel", "Tripel"), ("Kriek", "Kriek"), ("Lambic", "Lambic")]:
                r[f"超阈{lab[10:]}_{key}"] = o.mean() if st is None else o[S == st].mean()
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("附件 8 off_flavors 非空的字段:", list(DIC.index[DIC.off_flavors.notna()]))


# %% 检查 6
# 检查 6 | 服务决策 D8（感官画像的验证参照）
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


t0 = time.time()
PANEL = components(tie_pair_counts(y[SENS]), 3, len(y))
XF = X[CHEM].fillna(X[CHEM].mean()).values
Z = y[SENS].values
HP = dict(max_depth=3, min_samples_leaf=20, learning_rate=0.05, max_iter=200, random_state=SEED)
Q1_FULL = [HistGradientBoostingRegressor(**HP).fit(XF, Z[:, d]) for d in range(50)]
OOF = np.column_stack([cross_val_predict(HistGradientBoostingRegressor(**HP), XF, Z[:, d], groups=PANEL, cv=GroupKFold(5)) for d in range(50)])
print(f"代理问题 1 模型（固定超参数，按品评团 5 折）OOF 50 维平均 Spearman {np.mean([spearmanr(OOF[:, d], Z[:, d])[0] for d in range(50)]):.3f}")


def portrait(xfull):
    return np.column_stack([m.predict(xfull) for m in Q1_FULL])


rng = np.random.default_rng(SEED)
rows = []
for st in STYLES:
    m = S == st
    P = portrait(full_profile(gfull.sample_latent(st, 1000, rng)))
    pg, po, zo = np.median(P, 0), np.median(OOF[m], 0), np.median(Z[m], 0)
    rows.append(dict(风格=st, n=int(m.sum()), 生成vs真实OOF=spearmanr(pg, po)[0], 生成vs观测=spearmanr(pg, zo)[0],
                     真实OOFvs观测=spearmanr(po, zo)[0], MAE生成vs真实OOF=np.abs(pg - po).mean()))
P6 = pd.DataFrame(rows)
print("各风格画像中位数在 50 维上的 Spearman：")
print(P6.round(3).to_string(index=False))
print("22 风格平均:", P6[["生成vs真实OOF", "生成vs观测", "真实OOFvs观测", "MAE生成vs真实OOF"]].mean().round(3).to_dict())
print(f"用时 {time.time() - t0:.0f} s")


# %% 检查 7
# 检查 7 | 服务决策 D9（迁移映射、约束投影与迁移强度）
t0 = time.time()
SRC, TGT = "Pils/Lager", "Tripel"
gi = {c: i for i, c in enumerate(GEN)}
DJ = [gi[c] for c in DEFECT_CHEM]
GRID = np.linspace(-6, 6, 4801)
XGRID = {j: np.interp(norm.cdf(GRID), gfull.mg.blocks[j][2], gfull.mg.blocks[j][3]) for j in DJ}
Cw, Cinv = gfull.C, np.linalg.inv(gfull.C)


def thr(c, kind="threshold_mean"):
    th = DIC.loc[c, kind]
    return np.nan if pd.isna(th) else (np.log10(th) if is_log(c) else th)


def chem_bounds(x0, rule="threshold", kind="threshold_mean"):
    """化学层上界（附件 1 尺度）：有阈值者 max(x, θ)，无阈值者 x；strict 规则一律为 x。再换成潜空间上界。
    预运行的 α星_严格 只把感官层换成连续等级不升高，化学层仍用阈值规则。"""
    b = []
    for c in DEFECT_CHEM:
        th = thr(c, kind)
        b.append(x0[gi[c]] if (rule == "strict" or np.isnan(th)) else max(x0[gi[c]], th))
    beta = np.array([GRID[XGRID[j] <= b[k] + 1e-12].max() for k, j in enumerate(DJ)])
    return np.array(b), beta


def project(ut, beta):
    """min (u−ũ)ᵀΣ⁻¹(u−ũ)  s.t. u_j ≤ β_j (j ∈ D_c)。有效集法：等式约束的解为高斯条件均值，乘子须非负。"""
    A = [k for k, j in enumerate(DJ) if ut[j] > beta[k]]
    for _ in range(50):
        u, drop = ut.copy(), []
        if A:
            ja = [DJ[k] for k in A]
            lam = np.linalg.solve(Cw[np.ix_(ja, ja)], ut[ja] - beta[A])
            u = ut - Cw[:, ja] @ lam
            drop = [A[i] for i in range(len(A)) if lam[i] < -1e-10]
        add = [k for k, j in enumerate(DJ) if k not in A and u[j] > beta[k] + 1e-9]
        if not add and not drop:
            return u, len(A)
        A = [k for k in A if k not in drop] + add
    return u, len(A)


lo, hi = Z.min(0), Z.max(0)
DSI = [SENS.index(d) for d in DEFECT_SENS]


def grade_cont(zz):
    """问题 1 §4.5 的线性连续等级 T_d（全体 250 款端点；正式运行用训练折端点与选定映射）。"""
    return np.clip(6 * (zz - lo) / (hi - lo), 0, 6)


def sens_ok(Gs, G0, rule):
    if rule == "grade":
        return np.all(np.round(Gs) <= np.round(G0), axis=1)
    return np.all(Gs <= G0 + 1e-9, axis=1)


skf = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(XG, S))
FOLD = np.empty(len(S), int)
for f, (_, te) in enumerate(skf):
    FOLD[te] = f
RFS = [RandomForestClassifier(500, random_state=SEED, n_jobs=-1).fit(XGf[tr], S[tr]) for tr, _ in skf]


def p_style(b, xg, st=TGT):
    """交叉拟合的风格判别器：评价啤酒 b 的迁移结果时，判别器没有见过 b。"""
    rf = RFS[FOLD[b]]
    return rf.predict_proba(np.atleast_2d(xg))[:, list(rf.classes_).index(st)]


def d2(u, st=TGT):
    return (u - gfull.mu[st]) @ Cinv @ (u - gfull.mu[st])


ALPHA = np.round(np.arange(0, 1.0001, 0.05), 2)
tgt_idx = np.where(S == TGT)[0]
src_idx = np.where(S == SRC)[0]
delta = gfull.mu[TGT] - gfull.mu[SRC]
rows = []
for b in src_idx:
    u, x0 = gfull.U[b], XGf[b]
    bx, beta = chem_bounds(x0)
    ut = u + delta
    up, nact = project(ut, beta)
    uc = ut.copy()
    uc[DJ] = np.minimum(ut[DJ], beta)
    xt = gfull.mg.inverse(ut[None])[0]
    viol = [c for k, c in enumerate(DEFECT_CHEM) if xt[gi[c]] > bx[k] + 1e-9]
    G0 = grade_cont(portrait(full_profile(u[None]))[0])[DSI]
    path = np.array([u + a * (up - u) for a in ALPHA])
    Gp = grade_cont(portrait(full_profile(path)))[:, DSI]
    ok_g, ok_s = sens_ok(Gp, G0, "grade"), sens_ok(Gp, G0, "strict")
    a_g, a_s = ALPHA[ok_g].max(), ALPHA[ok_s].max()
    ua = u + a_g * (up - u)
    Gt = grade_cont(portrait(full_profile(ut[None]))[0])[DSI]
    # 检索对照：满足化学层与感官层约束、马氏距离最近的真实 Tripel
    ok_r = [t for t in tgt_idx if np.all(XGf[t][[gi[c] for c in DEFECT_CHEM]] <= bx + 1e-12)
            and sens_ok(grade_cont(portrait(XF[t][None])[0])[DSI][None], G0, "grade")[0]]
    rt = min(ok_r, key=lambda t: (gfull.U[t] - ut) @ Cinv @ (gfull.U[t] - ut)) if ok_r else None
    rows.append(dict(啤酒=X.beer.iloc[b], 平移违反化学=";".join(viol) or "-", 投影有效约束=nact,
                     平移感官违反=";".join(DEFECT_SENS[k] for k in range(6) if np.round(Gt[k]) > np.round(G0[k])) or "-",
                     可行路径="".join("1" if o else "0" for o in ok_g), α星=a_g, α星_严格=a_s,
                     P源=p_style(b, x0)[0], P平移=p_style(b, xt)[0], P投影=p_style(b, gfull.mg.inverse(up[None])[0])[0],
                     P截断=p_style(b, gfull.mg.inverse(uc[None])[0])[0], Pα星=p_style(b, gfull.mg.inverse(ua[None])[0])[0],
                     P检索=p_style(rt, XGf[rt])[0] if rt is not None else np.nan, 检索可行数=len(ok_r),
                     d2平移=d2(ut), d2投影=d2(up), d2截断=d2(uc), d2α星=d2(ua),
                     保留=spearmanr(u - gfull.mu[SRC], up - gfull.mu[TGT])[0]))
R7 = pd.DataFrame(rows)
print(R7.round(3).to_string(index=False))
d_real = np.array([d2(gfull.U[t]) for t in tgt_idx])
p_real = np.array([p_style(t, XGf[t])[0] for t in tgt_idx])
print(f"参照：真实 Tripel 的交叉拟合 P(Tripel) 中位 {np.median(p_real):.3f}；真实 Tripel 的 d² 中位 {np.median(d_real):.1f}；"
      f"源 Pils/Lager 到 Tripel 均值的 d² 中位 {np.median([d2(gfull.U[b]) for b in src_idx]):.1f}")
print("中位数:", R7.drop(columns=["啤酒", "平移违反化学", "平移感官违反", "可行路径"]).median().round(3).to_dict())
contig = [int(np.argmin(np.r_[np.array(list(s)) == "1", False])) - 1 for s in R7["可行路径"]]
print(f"α* = 1 的啤酒 {int((R7['α星'] == 1).sum())} / {len(R7)}；α* = 0 的 {int((R7['α星'] == 0).sum())}；"
      f"可行 α 不连续的啤酒 {int(sum(ALPHA[c] < a for c, a in zip(contig, R7['α星'])))}")

# 原尺度平移对照
mu_raw = {st: np.nanmean(XG[S == st], 0) for st in STYLES}
raw = XGf[src_idx] + (mu_raw[TGT] - mu_raw[SRC])
lin = np.array([str(DIC.loc[c, "transform_interpretation"]).startswith("线性") for c in GEN])   # 33 个已核实的原值列
print(f"原尺度均值平移：超出观测范围的单元格 {int(((raw < np.nanmin(XG, 0)) | (raw > np.nanmax(XG, 0))).sum())} / {raw.size}；"
      f"有下限的列中低于下限的单元格 {int((raw[:, tied >= 2] < mins[tied >= 2] - 1e-12).sum())}；线性列中出现负值的单元格 {int((raw[:, lin] < 0).sum())}")

# 全部 22×21 个有序风格对：平移违反化学约束的比例与投影后有效约束数
pair = []
for s0 in STYLES:
    for s1 in STYLES:
        if s0 == s1:
            continue
        v, na = [], []
        for b in np.where(S == s0)[0]:
            bx, beta = chem_bounds(XGf[b])
            ut = gfull.U[b] + gfull.mu[s1] - gfull.mu[s0]
            v.append(np.any(ut[DJ] > beta + 1e-9))
            na.append(project(ut, beta)[1])
        pair.append((s0, s1, np.mean(v), np.mean(na)))
PR = pd.DataFrame(pair, columns=["源", "目标", "违反比例", "有效约束数"])
print(f"\n462 个有序风格对：平移后至少违反一项化学约束的啤酒比例，中位 {PR['违反比例'].median():.2f}，"
      f"范围 {PR['违反比例'].min():.2f}–{PR['违反比例'].max():.2f}；投影的平均有效约束数中位 {PR['有效约束数'].median():.1f}")
print(f"用时 {time.time() - t0:.0f} s")
