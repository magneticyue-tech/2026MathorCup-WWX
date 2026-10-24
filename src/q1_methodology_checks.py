# 问题 1 方法论（notebooks/q1_methodology.md）第 3 节"六个决策检查"的代码，由原 q1_methodology.ipynb 的 6 个代码单元原样拆出。
# 在项目根目录（或其任一子目录）运行：python src/q1_methodology_checks.py，约 5 秒。
# 只读 data/raw 与 results/data_audit，不写任何文件。各段以 "# %%" 分隔，可在 VS Code / Jupyter 中逐段运行。

# %% 检查 1
# 检查 1 | 服务决策 D1（输出哪些维度）与 D2（barnyard 怎么处理）
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

pd.set_option("display.width", 160)
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw" / "4.csv").exists())
RAW, AUDIT = ROOT / "data" / "raw", ROOT / "results" / "data_audit"
y = pd.read_csv(RAW / "4.csv")           # 品评小组结果（标准化空间）
X = pd.read_csv(RAW / "1.csv")           # 化学数据
SENS = list(y.columns[3:])
CHEM = list(X.columns[3:])
assert len(SENS) == 50 and len(CHEM) == 231 and (X.beer.values == y.beer.values).all()

GROUPS = {
    "Aroma块": [c for c in SENS if c.startswith("A_")],
    "Flavour块": [c for c in SENS if c.startswith("F_")],
    "缺陷": ["dms", "metallic", "X4vg", "t2n", "diacetyl", "stale_hops"],
    "香料": ["orange", "coriander", "clove"],
    "酸啤": ["lactic", "acetic", "barnyard"],
    "基本味与口感": ["body", "alcohol", "bitternes", "sweetness", "co2", "acidity", "aftertaste"],
    "总体": ["overall"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(SENS)
CORE42 = sum([GROUPS[k] for k in ["Aroma块", "Flavour块", "缺陷", "香料", "酸啤"]], [])
print("各组维度数:", {k: len(v) for k, v in GROUPS.items()}, "| 主报告集合:", len(CORE42))

tpl = pd.read_csv(AUDIT / "template_field_mapping.csv").set_index("template_field")
print("模板公式单元格  barnyard:", tpl.loc["barnyard", "formula_cell"], "  stale_hops:", tpl.loc["stale_hops", "formula_cell"])

rho = y[SENS].corr(method="spearman")
upper = rho.where(np.triu(np.ones(rho.shape, dtype=bool), 1)).stack().sort_values(ascending=False)
print("\n50 维两两 Spearman 最高的 3 对:")
print(upper.head(3).round(3).to_string())
d = (y.barnyard - y.stale_hops).abs()
print(f"\nbarnyard 与 stale_hops 的 |z 差|: 中位 {d.median():.3f}，95% 分位 {d.quantile(.95):.3f}，"
      f"差 > 0.1 的啤酒 {int((d > 0.1).sum())} / {len(y)} 款")


# %% 检查 2
# 检查 2 | 服务决策 D3（化学缺失）、D4（底值）与 D5（模型族）
na = X[CHEM].isna().sum()
na = na[na > 0]
row_na = X[CHEM].isna().any(axis=1)
print(f"附件1 缺失: {int(na.sum())} 格 / {len(na)} 列 / {int(row_na.sum())} 行；整行删除会丢 {row_na.mean():.1%} 的样本")

cmp = pd.read_csv(AUDIT / "calibrated_missing_comparison.csv")
m1 = cmp[cmp.csv1_missing]
print(f"附件1 的 NA 在附件8 参考表中对应 0 的: {int(m1.calibrated_zero.sum())} / {len(m1)}（其余为综合指标，参考表无此列）")

rows = []
for c in na.index:
    v = X[c].dropna()
    rows.append({"字段": c, "NA数": int(na[c]),
                 "均值填补落在的分位": (v < v.mean()).mean(),
                 "观测最小值": f"{v.min():.2g}", "观测为0的格": int((v == 0).sum())})
print("\n若按论文做均值填补，'未检出'的啤酒会被放到这些位置:")
print(pd.DataFrame(rows).round(3).to_string(index=False))

FLOOR = -1.81781854
at_floor = X[CHEM].round(8).eq(FLOOR)
frac = at_floor.mean()
print(f"\n共同底值 {FLOOR}: 出现在 {int((frac > 0).sum())} 个字段、{int(at_floor.values.sum())} 格；"
      f"字段内占比中位 {frac[frac > 0].median():.2f}，最高 {frac.max():.2f}，超过一半的字段 {int((frac > .5).sum())} 个")


# %% 检查 3
# 检查 3 | 服务决策 D7（交叉验证怎么分组）与 D10（换算的零点）
def tie_pair_counts(z):
    """统计每对啤酒在多少个维度上 z 值完全相同（同一场次全员打 0 分才会产生这种并列）。"""
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

cnt = tie_pair_counts(y[SENS])
style = y.tasting_category_fine.values
same_style = np.mean([style[i] == style[j] for i, j in cnt])
ns = pd.Series(style).value_counts().values
print(f"至少在 1 个维度上并列的啤酒对: {len(cnt)}，其中同风格占 {same_style:.0%}"
      f"（任取两款同风格的概率只有 {(ns * (ns - 1)).sum() / (len(y) * (len(y) - 1)):.0%}）")
for k in (2, 3, 4, 5):
    g = components(cnt, k, len(y))
    s = pd.Series(g).value_counts()
    multi = s[s > 1]
    purity = np.mean([pd.Series(style[g == i]).value_counts(normalize=True).iloc[0] for i in multi.index])
    print(f"阈值 k={k}: 多款组 {len(multi)} 个，覆盖 {multi.sum()} 款，最大 {multi.max()} 款，平均风格纯度 {purity:.2f}")
SESSION = components(cnt, 3, len(y))
SIZE = pd.Series(SESSION).map(pd.Series(SESSION).value_counts()).values


# %% 检查 4
# 检查 4 | 服务决策 D9（零膨胀结构）与 D10（z → 0–6 换算）
def session_zero(col):
    """多款场次内的最小值若出现 ≥2 次，视为该场'全员 0 分'对应的 z 值。"""
    z = y[col].round(9)
    out = {}
    for g in np.unique(SESSION[SIZE > 1]):
        v = z[SESSION == g]
        if (v == v.min()).sum() >= 2:
            out[g] = v.min()
    return pd.Series(out, dtype=float)

Z0, rows = {}, []
for c in SENS:
    z0 = session_zero(c)
    Z0[c] = z0
    at0 = sum(int(((y[c].round(9) == z0[g]) & (SESSION == g)).sum()) for g in z0.index)
    rows.append({"维度": c, "有零点场次": len(z0), "场次零点占比": at0 / len(y),
                 "零点z最小": z0.min(), "零点z最大": z0.max()})
ZT = pd.DataFrame(rows).set_index("维度")
print("场次零点占比最高的 10 个维度（同一维度的'0 分'在不同场次对应不同 z）:")
print(ZT.sort_values("场次零点占比", ascending=False).round(3).head(10).to_string())
W = pd.DataFrame(Z0)                      # 行 = 场次，列 = 维度
spread = (W.max(axis=1) - W.min(axis=1))[W.notna().sum(axis=1) >= 5]
print(f"\n同一场次内、不同维度的零点 z 极差: 中位 {spread.median():.2f}（{len(spread)} 个场次）；"
      f"同一维度、不同场次的零点 z 标准差: 中位 {W.std().median():.3f}")
ZI = list(ZT.index[ZT["场次零点占比"] >= 0.3])
NOANCHOR = list(ZT.index[ZT["有零点场次"] == 0])
print(f"\n零点占比 ≥30% 的维度 {len(ZI)} 个；完全没有零点锚的维度 {len(NOANCHOR)} 个: {NOANCHOR}")

# 格点：场次内高于零点的台阶应是 1/(n·σ_p) 的整数组合；用 log(最小台阶) ~ 场次 + 维度 检验能否借此恢复量表
recs = []
for g in np.unique(SESSION[SIZE >= 5]):
    for c in SENS:
        if g in Z0[c].index:
            v = np.sort(y[c].round(9).values[SESSION == g])
            step = v - Z0[c][g]
            step = step[step > 1e-6]
            if len(step):
                recs.append((g, c, np.log(step.min())))
L = pd.DataFrame(recs, columns=["g", "c", "ls"])
D = pd.get_dummies(L[["g", "c"]].astype(str), drop_first=True).astype(float)
D.insert(0, "const", 1.0)
beta, *_ = np.linalg.lstsq(D.values, L.ls.values, rcond=None)
r2 = 1 - ((L.ls - D.values @ beta) ** 2).sum() / ((L.ls - L.ls.mean()) ** 2).sum()
print(f"\n格点检验: {len(L)} 个(场次, 维度)最小台阶，'场次+维度'两因素只能解释 log 台阶方差的 {r2:.0%}")

# m 模型换算：s = m·r/(1+r²)·(z − z0)，r = −(列中位零点)，m = E[x²]/E[x] ∈ [1, 6]
def to_raw(col, z, m, sess=None):
    z0_all = Z0[col]
    z0 = pd.Series(sess).map(z0_all).fillna(z0_all.median()).values if sess is not None else z0_all.median()
    r = -z0_all.median()
    return m * r / (1 + r ** 2) * (z - z0)

ANCH = [c for c in SENS if len(Z0[c])]
grades = {m: pd.DataFrame({c: np.clip(np.round(to_raw(c, y[c].values, m, SESSION)), 0, 6).astype(int) for c in ANCH}) for m in (2, 3, 4)}
M_MAX = pd.Series({c: 6 / np.max(to_raw(c, y[c].values, 1.0, SESSION)) for c in ANCH})   # 最高分的啤酒不能超过 6 分
print(f"\n数据给出的 m 上限（逐维度）: 最小 {M_MAX.min():.2f}（{M_MAX.idxmin()}），中位 {M_MAX.median():.2f}；"
      f"m=3 对全部 {len(ANCH)} 维可行，m=4 对 {int((M_MAX < 4).sum())} 维不可行: {list(M_MAX.index[M_MAX < 4])}")
raw4 = pd.DataFrame({c: to_raw(c, y[c].values, 4, SESSION) for c in ANCH})
for name, cols in [(f"零膨胀 {len(ZI)} 维", ZI), (f"全部有锚 {len(ANCH)} 维", ANCH)]:
    print(f"\n{name}: 等级 m=2 与 m=4 不同的格占 {(grades[2][cols] != grades[4][cols]).values.mean():.0%}，"
          f"m=3 与 m=4 不同 {(grades[3][cols] != grades[4][cols]).values.mean():.0%}；m=4 时超出 6 分的格 {(raw4[cols] > 6).values.mean():.1%}")
    print("  m=3 时等级分布:", pd.Series(grades[3][cols].values.ravel()).value_counts().sort_index().astype(int).to_dict())

# 子 ≤ 父：父项（总体强度）没有零点锚，用约定 C-mid: s = 3 + 3·z/max|z|
viol = {}
for m in (2, 3, 4):
    bad = tot = 0
    for parent in [c for c in SENS if c.endswith("_all")]:
        p_raw = 3 + 3 * y[parent] / y[parent].abs().max()
        for child in [c for c in SENS if c.startswith(parent[:-3]) and c != parent and c in Z0 and len(Z0[c])]:
            ch = to_raw(child, y[child].values, m, SESSION)
            bad += int((ch > p_raw + 0.5).sum()); tot += len(y)
    viol[m] = bad / tot
print("子项换算值超过父项 0.5 分以上的比例（父项用 C-mid 约定）:", {m: f"{v:.1%}" for m, v in viol.items()})


# %% 检查 5
# 检查 5 | 服务决策 D6（是否用风格）、D7（划分方式）与成功标准
v = y[SENS].var()
REL = pd.DataFrame({f"n={n}": ((v - 1 / n) / (1 - 1 / n)).clip(lower=0) / v for n in (8, 12, 16)})
print("小组平均分信度 R_n（可达 R² 的粗略上限）按出席人数:")
print(REL.describe().loc[["min", "50%", "max"]].round(2).to_string())

def style_baseline(holdout, k=5):
    """风格均值基线（向训练集全局均值收缩 k 款），测试单元 holdout 内的啤酒全部留出。"""
    Y = y[SENS]
    S = Y.groupby(style).transform("sum"); C = Y.groupby(style).transform("count")
    key = pd.Series(holdout).astype(str) + "|" + pd.Series(style)
    Sg = Y.groupby(key.values).transform("sum"); Cg = Y.groupby(key.values).transform("count")
    tot, ntot = Y.sum(), len(Y)
    hs = Y.groupby(holdout).transform("sum"); hn = pd.Series(holdout).map(pd.Series(holdout).value_counts()).values
    mu = (tot.values - hs.values) / (ntot - hn)[:, None]
    return ((S - Sg).values + k * mu) / ((C - Cg).values + k)

def score(pred):
    out = {}
    for j, c in enumerate(SENS):
        e = y[c].values - pred[:, j]
        out[c] = (1 - (e ** 2).sum() / ((y[c] - y[c].mean()) ** 2).sum(), spearmanr(pred[:, j], y[c])[0])
    return pd.DataFrame(out, index=["R2", "Spearman"]).T

loo = score(style_baseline(np.arange(len(y))))
lso = score(style_baseline(SESSION))
tab = pd.DataFrame({"留一 R²": loo.R2, "留场次 R²": lso.R2, "留一 ρ": loo.Spearman, "留场次 ρ": lso.Spearman})
print("\n风格均值基线（50 维 / 主报告 42 维）:")
print(pd.DataFrame({"50维均值": tab.mean(), "50维中位": tab.median(), "42维均值": tab.loc[CORE42].mean()}).round(3).T.to_string())
print(f"留场次后 R² 下降的维度 {int((tab['留场次 R²'] < tab['留一 R²']).sum())} / 50；论文化学模型测试 R² ≈ 0.21（GBR，随机划分）")

# 对照：把场次标签在同风格内随机打乱 → 组大小与风格构成不变，只去掉'同场次'这层关系
rng = np.random.default_rng(0)
perm = []
for _ in range(50):
    lab = SESSION.copy()
    for s in np.unique(style):
        idx = np.where(style == s)[0]
        lab[idx] = rng.permutation(lab[idx])
    perm.append(score(style_baseline(lab)).R2.mean())
print(f"伪场次对照（50 次置换）R² 均值 {np.mean(perm):.3f} ± {np.std(perm):.3f}；"
      f"真实场次 {lso.R2.mean():.3f} → 场次本身带来的差 {np.mean(perm) - lso.R2.mean():.3f}")


# %% 检查 6
# 检查 6 | 服务决策 D5（单调约束）与 D11（OAV 特征）——只做方向与覆盖检查，不用于特征筛选
PRIORS = [("isopentyl_acetate", "A_esters_isoaa", +1), ("ethyl_acetate", "A_esters_ethac", +1),
          ("X4VG", "X4vg", +1), ("DMS", "dms", +1), ("X2.3.butanedione", "diacetyl", +1),
          ("acetic_acid.g.L.", "acetic", +1), ("lactic_acid.mg.L.", "lactic", +1), ("pH", "acidity", -1),
          ("bitterness", "bitternes", +1), ("ethanol..v.v.", "alcohol", +1), ("X4.ethyl.phenol", "barnyard", +1),
          ("linalool", "coriander", +1)]
rows = []
for chem, sens, sign in PRIORS:
    r_all = X[CHEM].corrwith(y[sens], method="spearman")
    r = r_all[chem]
    rows.append({"化合物": chem, "感官维度": sens, "预期方向": "+" if sign > 0 else "−", "Spearman": r,
                 "方向一致": np.sign(r) == sign, "在231个中的排名": int((r_all.abs() > abs(r)).sum()) + 1})
P = pd.DataFrame(rows)
print(P.round(3).to_string(index=False))

dic = pd.read_csv(AUDIT / "chemical_dictionary.csv").set_index("csv1_field")
thr = dic[dic.threshold_mean.notna()]
oav_rows = []
for c, r in thr.iterrows():
    conc = 10 ** X[c] if r.transform_interpretation.startswith("log10") else X[c]
    oav_rows.append({"字段": c, "组": "log10" if r.transform_interpretation.startswith("log10") else "线性",
                     "OAV≥1 的啤酒比例": (conc / r.threshold_mean >= 1).mean()})
O = pd.DataFrame(oav_rows)
print(f"\n有阈值的化合物 {len(O)} 个（{O['组'].value_counts().to_dict()}）；"
      f"从未超过阈值 {int((O['OAV≥1 的啤酒比例'] == 0).sum())} 个，"
      f"所有啤酒都超过 {int((O['OAV≥1 的啤酒比例'] == 1).sum())} 个，其余 {int(O['OAV≥1 的啤酒比例'].between(0, 1, inclusive='neither').sum())} 个在样本间跨越阈值")
print("双乙酰（X2.3.butanedione）OAV≥1 的啤酒比例:", round(float(O.set_index("字段").loc["X2.3.butanedione"].iloc[1]), 3),
      "（阈值均值口径）→ 近半数啤酒超过阈值，方向相反不能只用'低于阈值闻不到'解释")
