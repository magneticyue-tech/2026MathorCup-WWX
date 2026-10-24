# 问题1方法论补充（notebooks/q1_methodology_supplement.md）的复现脚本。
# 在项目根目录运行：python src/q1_supplement_checks.py
# 只读取 data/raw、notebooks/q1_methodology.ipynb 与论文 Zenodo 补充工作簿，不写任何文件。
# Zenodo 工作簿路径用环境变量 ZENODO_XLSX 指定；项目根目录可用 Q1_ROOT 指定（默认当前目录）。
import contextlib, io, json, os
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(os.environ.get("Q1_ROOT", "."))
RAW = ROOT / "data" / "raw"
ZEN = Path(os.environ.get("ZENODO_XLSX", Path(os.environ.get("TEMP", ".")) / "zenodo_10653704" / "Supplemental Files and Figure source files.xlsx"))
if not ZEN.exists():
    raise SystemExit(f"找不到 Zenodo 工作簿 {ZEN}。从 https://doi.org/10.5281/zenodo.10653704 下载"
                     " 'Supplemental Files and Figure source files.xlsx'（md5 52645e38d90aafda272532e19e2fea1c），再设置 ZENODO_XLSX。")

# 复用方法论 notebook 的代码单元 1–4（场次推断、场次零点、m 模型），不重写一份
nb = json.load(open(ROOT / "notebooks" / "q1_methodology.ipynb", encoding="utf-8"))
g = {}
cwd = os.getcwd(); os.chdir(ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    for c in [c for c in nb["cells"] if c["cell_type"] == "code"][:4]:
        exec("".join(c["source"]), g)
os.chdir(cwd)
y, SESSION, SIZE, Z0, to_raw, ANCH, M_MAX, CORE42 = (g[k] for k in "y SESSION SIZE Z0 to_raw ANCH M_MAX CORE42".split())
SENS = list(y.columns[3:])

# A. 附件与论文公开数据逐格对照
for k in (1, 2, 4, 5, 6, 7):
    a = pd.read_csv(RAW / f"{k}.csv"); z = pd.read_excel(ZEN, sheet_name=f"Supplementary File S{k}")
    a = a.select_dtypes("number").drop(columns=["beer_id"], errors="ignore")
    z = z.select_dtypes("number").drop(columns=["beer_id"], errors="ignore").dropna(how="all")
    A, B = a.to_numpy(float), z.to_numpy(float)[: len(a)]
    print(f"A{k}: 数值块 {A.shape} vs {B.shape}，最大差 {np.nanmax(np.abs(A - B)):.1e}，缺失位置相同 {np.array_equal(np.isnan(A), np.isnan(B))}")
z3 = pd.read_excel(ZEN, sheet_name="Supplementary File S3", header=None).astype(str)
for sh in [f"beer_{k}" for k in range(1, 13)]:
    a3 = pd.read_excel(RAW / "3.xlsx", sheet_name=sh, header=None).astype(str)
    d3 = np.argwhere((a3 != z3).to_numpy())
    print(f"A3 [{sh}] 与 S3 不同的单元格（行, 列, 附件, 论文）: {[(int(i) + 1, int(j) + 1, a3.iat[i, j], z3.iat[i, j]) for i, j in d3]}")
for s in ["Sensory thresholds", "Sensory names", "GBM importance MDI", "GBM SHAP Values", "Calibrated chemical dataset", "Beer Prices"]:
    a = pd.read_excel(RAW / "8.xlsx", sheet_name=s, header=None); z = pd.read_excel(ZEN, sheet_name=s, header=None)
    print(f"A8 [{s}] 完全相同: {a.shape == z.shape and a.fillna('NA').astype(str).equals(z.fillna('NA').astype(str))}")

# B. 论文自己的 70/30 划分上：论文 GBR（表 S3）对比"只按风格猜"（收缩 k=5）
names = pd.read_excel(ZEN, sheet_name="Sensory names")
lab = names[names.dataset == "Trained panel"].set_index("CleanName_PanelFeedback")["R_name"]
gbr = pd.read_excel(ZEN, sheet_name="Supplementary Table S3").set_index("aspect")["GradientBoost"].rename(index=lab).reindex(SENS)
style = y.tasting_category_fine
tr, te = train_test_split(np.arange(len(y)), test_size=0.30, random_state=0, shuffle=True, stratify=style)
Y = y[SENS]; gm = Y.iloc[tr].mean(); S = Y.iloc[tr].groupby(style.iloc[tr].values)
pred = np.vstack([((S.sum().loc[s] + 5 * gm) / (S.count().loc[s] + 5)).values for s in style.iloc[te]])
yt = Y.iloc[te].to_numpy()
r2 = pd.Series(1 - ((yt - pred) ** 2).sum(0) / ((yt - yt.mean(0)) ** 2).sum(0), index=SENS)
OTHER = [c for c in SENS if c not in CORE42]
for nm, cols in [("全部50维", SENS), ("主报告42维", CORE42), ("其余8维", OTHER)]:
    print(f"B {nm}: 论文GBR R² {gbr[cols].mean():.3f} | 风格均值 R² {r2[cols].mean():.3f} | GBR更好 {(gbr[cols] > r2[cols]).sum()}/{len(cols)}")

# C. 论文表 S2 的 12 款跨场次重复品评啤酒是否都落在推断的"单款组"
rep = [int(c.split()[1]) for c in pd.read_excel(ZEN, sheet_name="Supplementary Table S2").columns[1:]]
idx = [int(np.where(y.beer.values == b)[0][0]) for b in rep]
n1 = int((SIZE == 1).sum())
print(f"C 重复啤酒 {rep} 所在组大小 {SIZE[idx].tolist()}；单款组 {n1}/{len(y)}；巧合概率 {comb(n1, len(rep)) / comb(len(y), len(rep)):.1e}")

# D. Validation Results 1–3：同一品评小组的逐人 0–6 分（加标验证实验）
ATT = ["Malt", "Sweetness", "Hops", "Bitterness", "Esters", "Fullness (body)", "Alcohol (warming)", "Overall appreciation"]
R = []
for k in (1, 2, 3):
    d = pd.read_excel(ZEN, sheet_name=f"Validation Results {k}"); d["date"] = pd.to_datetime(d.Time).dt.date; R.append(d)
R = pd.concat(R, ignore_index=True)
n_all = len(R); R = R.drop_duplicates(subset=["date", "Name", "Beer"] + ATT)   # Taster 25 在 07-13 重复提交一次
x = R[ATT].to_numpy(float)
m_att = pd.Series((x ** 2).sum(0) / x.sum(0), index=ATT)
print(f"D 验证打分 {n_all} 行，去重后 {len(R)} 条、{R.Name.nunique()} 人、{R.date.min()} 至 {R.date.max()}；"
      f"m=E[x²]/E[x] 范围 {m_att.min():.2f}–{m_att.max():.2f}，中位 {m_att.median():.2f}")
print("  ", m_att.round(2).to_dict())

# E. 底酒与附件啤酒的化学匹配；对照样品检验 z → 0–6 换算
det = pd.read_excel(ZEN, sheet_name="Validation Details"); det = det[det.SetName.notna()].copy()
det["date"] = pd.to_datetime(det.Date).dt.date
R = R.merge(det[["date", "Code", "BeerStyle", "SampleName"]], left_on=["date", "Beer"], right_on=["date", "Code"], how="left")
assert R.SampleName.notna().all()
cal = pd.read_excel(RAW / "8.xlsx", sheet_name="Calibrated chemical dataset")
conc = pd.read_excel(ZEN, sheet_name="Validation Concentrations").drop_duplicates("BeerStyle").set_index("BeerStyle")
cmap = {"Ethanol (v/v)": "ethanol..v.v.", "Ethyl acetate (mg/l)": "ethyl_acetate", "Ethyl hexanoate (mg/l)": "ethyl_hexanoate",
        "Isoamyl acetate (mg/l)": "isopentyl_acetate", "Phenylethyl acetate (mg/l)": "phenethyl_acetate",
        "Glycerol (g/l)": "glycerol", "Lactic acid (g/l)": "lactic_acid.mg.L."}   # 原表乳酸标 g/l，数值与附件8 的 mg/L 一致
L = np.log10(cal[list(cmap.values())].to_numpy(float) + 1e-3); sd = np.nanstd(L, 0)
BASE = {}
for st in conc.index:
    v = np.log10(conc.loc[st, list(cmap)].to_numpy(float) + 1e-3)
    dist = np.sqrt(np.nansum(((L - v) / sd) ** 2, 1)); o = np.argsort(dist)
    BASE[st] = int(cal.beer.iloc[o[0]])
    print(f"E 底酒 {st}: 最接近 {BASE[st]} 号（{cal.tasting_category_fine.iloc[o[0]]}），标准化距离 {dist[o[0]]:.2f}；第二近 {dist[o[1]]:.2f}")
    print("   验证表:", conc.loc[st, list(cmap)].astype(float).round(3).tolist(), "| 附件8:", cal.iloc[o[0]][list(cmap.values())].astype(float).round(3).tolist())
# Validation Details 自带 BaseBeer 列：按附件编号去查，风格和酒精度都对不上，说明那是实验室内部编号
for st, codes in det.groupby("BeerStyle").BaseBeer.unique().items():
    hit = cal[cal.beer.isin(codes)][["beer", "tasting_category_fine", "ethanol..v.v."]].values.tolist()
    print(f"E BaseBeer 列 {st}: {codes.tolist()}；当作附件编号查到 {hit}")

ctrl = R[R.SampleName == "Control"]
MAP = {"Malt": ["A_malt_all", "F_malt_all"], "Sweetness": ["sweetness"], "Hops": ["A_hops_all", "F_hops_all"],
       "Bitterness": ["bitternes"], "Esters": ["A_esters_all", "F_esters_all"], "Fullness (body)": ["body"],
       "Alcohol (warming)": ["alcohol"], "Overall appreciation": ["overall"]}
rows = []
for st, b in BASE.items():
    i = int(np.where(y.beer.values == b)[0][0]); cb = ctrl[ctrl.BeerStyle.str.lower() == st.lower()]
    for a, cols in MAP.items():
        for c in cols:
            z = y[c].values[i]; rec = dict(beer=b, attr=a, col=c, n=len(cb), raw=cb[a].mean(), se=cb[a].std(ddof=1) / np.sqrt(len(cb)), z=z)
            if c in ANCH:
                unit = float(to_raw(c, np.array([z]), 1.0, np.array([SESSION[i]]))[0])
                rec.update(m2=2 * unit, m3=3 * unit, mcap=min(4, M_MAX[c]) * unit, m4=4 * unit,
                           m_implied=rec["raw"] / unit if unit > 0 else np.nan,
                           m_max=M_MAX[c], n_zero_sess=len(Z0[c]))
            else:
                rec["c_mid"] = 3 + 3 * z / np.max(np.abs(y[c].values))   # notebook 的约定刻度 C-mid
            rows.append(rec)
T = pd.DataFrame(rows)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(T.round(2).to_string(index=False))
K = T[T.col.isin(["sweetness", "A_hops_all", "bitternes"])]
for m in ("m2", "m3", "mcap", "m4"):   # mcap = min(4, 数据上限)，即 notebook 的敏感性上端；m4 不受上限约束
    e = K[m] - K.raw
    print(f"  {m}: 平均误差 {e.mean():+.2f}，平均绝对误差 {e.abs().mean():.2f}，等级一致 {(np.clip(np.round(K[m]), 0, 6) == np.round(K.raw)).sum()}/{len(K)}")
# 要让对照分数与"最高的啤酒不超过 6 分"同时成立，零点最高只能到多少（线性刻度下 z0 ≤ (z_b − q·z_max)/(1 − q)，q = 原始分/6）
for c in ["sweetness", "A_hops_all", "bitternes"]:
    bound = {r.beer: round((r.z - r.raw / 6 * y[c].max()) / (1 - r.raw / 6), 2) for r in K[K.col == c].itertuples()}
    print(f"  {c}: 现用零点 {Z0[c].median():+.2f}（{len(Z0[c])} 个场次），数据最小 z {y[c].min():+.2f}，零点上限 {bound}")
for nm, cols in [("酯类取香气", ["A_malt_all", "F_hops_all", "A_esters_all", "body", "alcohol", "overall"]),
                 ("酯类取风味", ["A_malt_all", "F_hops_all", "F_esters_all", "body", "alcohol", "overall"])]:
    e = T[T.col.isin(cols)].eval("c_mid - raw")
    print(f"  C-mid 无锚维度（{nm}）: 平均绝对误差 {e.abs().mean():.2f}，误差超过 1 分 {(e.abs() > 1).sum()}/{len(e)}")
print("  两款酒之间的方向（64 号减 223 号）:")
for a, cols in MAP.items():
    t64, t223 = T[(T.beer == BASE["blond"]) & (T.attr == a)], T[(T.beer == BASE["NABLAB"]) & (T.attr == a)]
    print(f"    {a:22s} 原始分差 {t64.raw.iloc[0] - t223.raw.iloc[0]:+.2f} | z 差 {dict(zip(cols, (t64.z.values - t223.z.values).round(2)))}")

# 同一人同场给对照、加标两杯打分：残差的协方差占比 ≈ 个人系统性偏高/偏低的份额
print("  品评员个人偏移份额（35 对左右，单个份额的标准误约 0.17）:")
R["cell"] = R.date.astype(str) + "_" + R.Beer.astype(str) + "_" + R.BeerStyle
for a in ATT:
    R["res"] = R[a] - R.groupby("cell")[a].transform("mean")
    sub = R[R.groupby("cell")[a].transform("size") >= 3]
    piv = sub.pivot_table(index=["date", "Name", "BeerStyle"], columns="SampleName", values="res", aggfunc="first").dropna()
    share = np.cov(piv.Control, piv.Spiked)[0, 1] / np.mean([piv.Control.var(), piv.Spiked.var()])
    print(f"    {a:22s} 配对 {len(piv):2d}，份额 {share:+.2f}")

# F. 零点有效性：按场次零点（无则用中位零点）换算时低于零点、会被截成 0 分的啤酒占比
Sx = pd.Series(SESSION)
below = pd.Series({c: float((y[c].values < Sx.map(Z0[c]).fillna(Z0[c].median()).values - 1e-9).mean()) for c in ANCH})
print("F 低于零点占比 > 5% 的有锚维度:", below[below > 0.05].round(3).sort_values(ascending=False).to_dict())
print("  只靠 1 个场次零点的维度:", [c for c in ANCH if len(Z0[c]) == 1])
