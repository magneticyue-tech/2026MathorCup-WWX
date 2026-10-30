# 线程 review/2026-10-29-q3-w3-acceptance-a.md 的 Modeling 复核代码，两个检查。
# 在项目根目录运行：python experimentalCode/q3_w3_acceptance_checks.py。不写任何文件。
# 检查 A：合成数据上，"两个经验量之差的绝对值平均"类指标在按样本重抽时的偏移。
# 检查 B：用正式选参重建重复 0 的主模型配对池，比较留出真实啤酒与生成样本的标准化离散，及其与 C2ST 折外概率的关系。
# 本机库版本与正式运行不同（正式运行 sklearn 1.7.2），检查 B 的数值只作机制核对，不替代正式结果。

# %% 公共部分
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "code" / "src" / "q3").exists())
sys.path.insert(0, str(ROOT / "code"))
from src.q3.gen_metrics import c2st, corr_error, floor_metrics  # noqa: E402

# %% 检查 A：bootstrap 偏移（完美生成器与有偏生成器）
rng = np.random.default_rng(0)
n, p, k = 250, 225, 10
L = rng.normal(0, 0.5, (p, k))
Sig = L @ L.T + np.eye(p)


def draw(Sigma, m):
    return rng.multivariate_normal(np.zeros(p), Sigma, m)


Lb = L.copy()
Lb[:, 0] *= 0.6  # 有偏生成器：第一因子载荷减弱
Sig_b = Lb @ Lb.T + np.eye(p)
R, Ga, Gb = draw(Sig, n), draw(Sig, n), draw(Sig_b, n)
big = 20000
pop_b = corr_error(draw(Sig, big), draw(Sig_b, big))  # 有偏生成器的总体差异（近似）
pt_a, pt_b = corr_error(R, Ga), corr_error(R, Gb)
B = 200
bt_a, bt_b, bt_d = [], [], []
for _ in range(B):
    i = rng.integers(0, n, n)
    a, b = corr_error(R[i], Ga[i]), corr_error(R[i], Gb[i])
    bt_a.append(a)
    bt_b.append(b)
    bt_d.append(b - a)
q = lambda v: np.round(np.percentile(v, [2.5, 97.5]), 4)  # noqa: E731
print("检查 A：相关误差")
print(f"  完美生成器 点估计 {pt_a:.4f}  bootstrap 区间 {q(bt_a)}  总体值 0")
print(f"  有偏生成器 点估计 {pt_b:.4f}  bootstrap 区间 {q(bt_b)}  总体值约 {pop_b:.4f}")
print(f"  配对差     点估计 {pt_b - pt_a:.4f}  bootstrap 区间 {q(bt_d)}")
# 下限比例误差：40 列有 30% 下限，真实与完美生成器同分布
F = 40
fr = (rng.random((n, F)) < 0.3).astype(float)
fg = (rng.random((n, F)) < 0.3).astype(float)
mins = np.zeros(F)
pt_f = floor_metrics(fr, fg, mins)[0]
bt_f = [floor_metrics(fr[i], fg[i], mins)[0] for i in (rng.integers(0, n, n) for _ in range(B))]
print(f"  下限比例误差（完美生成器） 点估计 {pt_f:.4f}  bootstrap 区间 {q(bt_f)}  总体值 0")

# %% 检查 B：重复 0 的主模型配对池
from src.q3.data import load_chem  # noqa: E402
from src.q3.genval import gen_rep  # noqa: E402

REP = 0
t0 = time.time()
z = np.load(ROOT / "code/data/processed/q3/q3_select.npz", allow_pickle=False)
sel = {
    (int(r), int(f)): {"kind": str(kd), "k": int(kk), "kappa": float(ka)}
    for r, f, kd, kk, ka in zip(
        z["outer_rep"], z["outer_fold"], z["chosen_kind"], z["chosen_k"], z["chosen_kappa"]
    )
}
chem = load_chem()
X, S, styles = chem.XG, np.asarray(chem.style), chem.styles
pool = gen_rep(X, S, styles, REP, sel, {}, ("main",))  # gi=0 与正式运行的主模型随机数流相同
print(f"\n检查 B：重复 {REP} 重建用时 {time.time() - t0:.0f} s")

gv = np.load(ROOT / "code/data/processed/q3/q3_genval.npz", allow_pickle=True)
keys = list(gv["metric_keys"])
rec = {k: gv["metric_rep"][keys.index(k), REP] for k in ("main.dcr_ratio", "main.floor_err")}
dcr = np.median(pool["d_gen"][0]) / np.median(pool["d_real"])
fe = floor_metrics(pool["real_x"], pool["gen_x"][0], np.nanmin(X, 0))[0]
print(f"  复现核对 DCR 比 {dcr:.4f}（正式 {rec['main.dcr_ratio']:.4f}）")
print(f"  复现核对 下限比例误差 {fe:.4f}（正式 {rec['main.floor_err']:.4f}）")

# 生成样本与留出真实啤酒在同一折模型下的标准化离散 (1/p)Σ(u−μ̂_s)²/Σ_jj
si = {st: i for i, st in enumerate(styles)}
d_gen = np.empty(len(S))
for f in range(5):
    m = pool["fold"] == f
    mu = pool["mu"][f][[si[s] for s in S[m]]]
    v = np.diag(pool["C"][f])
    d_gen[m] = ((pool["gen_u"][0][m] - mu) ** 2 / v).mean(1)
d_real = pool["disp"]
print(f"  标准化离散 中位：留出真实 {np.median(d_real):.3f}，生成 {np.median(d_gen):.3f}")
print(f"  比值（真实/生成）均值 {d_real.mean() / d_gen.mean():.3f}")
print(f"  真实 > 配对生成 的比例 {np.mean(d_real > d_gen):.3f}")
ns = np.array([(S[pool["fold"] != pool["fold"][b]] == S[b]).sum() for b in range(len(S))])
for lo, hi in ((1, 5), (6, 15), (16, 40)):
    m = (ns >= lo) & (ns <= hi)
    print(
        f"  训练集同风格 {lo}–{hi} 款（{m.sum()} 款）：真实/生成 {d_real[m].mean() / d_gen[m].mean():.3f}"
    )

# C2ST 折外概率与离散的关系
res = c2st(pool["real_x"], pool["gen_x"][0])
oof, y = res["oof"], res["y"]
print(
    f"  合并 C2ST AUC {res['auc']:.3f}（正式重复 {REP}："
    f"{gv['metric_rep'][keys.index('main.auc_marg'), REP]:.3f}）"
)
dd = np.r_[d_real, d_gen]
print(
    f"  P(生成) 与离散的 Spearman：真实行 {spearmanr(oof[y == 0], d_real)[0]:.3f}，"
    f"生成行 {spearmanr(oof[y == 1], d_gen)[0]:.3f}"
)
# 只用离散这一个标量能否区分真实与生成
from sklearn.metrics import roc_auc_score  # noqa: E402

print(f"  单用标准化离散区分生成（离散越小越像生成）的 AUC {roc_auc_score(y, -dd):.3f}")

# 只计入风格均值抽样误差时的期望比值 1 + n_s/(n_s+κ)²（n_s 为该折训练集中同风格啤酒数）
from src.q3.selection import outer_splits  # noqa: E402

kap = np.array([sel[(REP, int(pool["fold"][b]))]["kappa"] for b in range(len(S))])
print(f"  参照：只计风格均值抽样误差的期望比值 {np.mean(1 + ns / (ns + kap) ** 2):.3f}")
# 留出真实啤酒超出训练折观测范围的单元格（生成样本按构造为 0）
oor = np.zeros(len(S))
for r, f, tr, te in outer_splits(S):
    if r != REP:
        continue
    lo, hi = np.nanmin(X[tr], 0), np.nanmax(X[tr], 0)
    xt = X[te]
    oor[te] = ((xt < lo) | (xt > hi)).sum(1)
print(
    f"  留出真实啤酒超出训练范围的列数：均值 {oor.mean():.2f}，至少 1 列的比例 {np.mean(oor > 0):.3f}"
)
print(
    f"  无越界的啤酒（{(oor == 0).sum()} 款）真实/生成 {d_real[oor == 0].mean() / d_gen[oor == 0].mean():.3f}"
)
big4 = sorted(styles, key=lambda st: -(S == st).sum())[:4]
for st in big4:
    m = S == st
    print(f"  {st}（{m.sum()} 款）真实/生成 {d_real[m].mean() / d_gen[m].mean():.3f}")
# 把留出真实啤酒截到训练折观测范围后重做合并 C2ST（只作诊断，生成样本不变）
Xw = pool["real_x"].copy()
for r, f, tr, te in outer_splits(S):
    if r != REP:
        continue
    Xw[te] = np.clip(Xw[te], np.nanmin(X[tr], 0), np.nanmax(X[tr], 0))
for lab, real in (("原值", pool["real_x"]), ("截到训练范围", Xw)):
    a = [c2st(real, pool["gen_x"][0], seed=sd)["auc"] for sd in range(3)]
    print(f"  合并 C2ST（{lab}）RF 种子 0–2：{np.round(a, 3)}")
# 分特征组的合并 C2ST：下限指示矩阵；有下限的列；无下限的列
mins = np.nanmin(X, 0)
fr_, fg_ = np.isclose(pool["real_x"], mins), np.isclose(pool["gen_x"][0], mins)
has = fr_.mean(0) > 0.02
cont = ~has
groups = {
    f"下限指示（{has.sum()} 列，0/1）": (fr_[:, has].astype(float), fg_[:, has].astype(float)),
    f"有下限的列原值（{has.sum()} 列）": (pool["real_x"][:, has], pool["gen_x"][0][:, has]),
    f"无下限的列原值（{cont.sum()} 列）": (pool["real_x"][:, cont], pool["gen_x"][0][:, cont]),
}
for lab, (a_, b_) in groups.items():
    print(f"  合并 C2ST {lab}：{c2st(a_, b_)['auc']:.3f}")
# 标准化离散只在未越界的单元格上计算
oor_cells = np.zeros_like(pool["real_x"], bool)
for r, f, tr, te in outer_splits(S):
    if r != REP:
        continue
    oor_cells[te] = (X[te] < np.nanmin(X[tr], 0)) | (X[te] > np.nanmax(X[tr], 0))
dr2, dg2 = np.empty(len(S)), np.empty(len(S))
for f in range(5):
    m = np.where(pool["fold"] == f)[0]
    mu = pool["mu"][f][[si[s] for s in S[m]]]
    v = np.diag(pool["C"][f])
    zr = (pool["real_u"][m] - mu) ** 2 / v
    zg = (pool["gen_u"][0][m] - mu) ** 2 / v
    keep = ~oor_cells[m]
    dr2[m] = (zr * keep).sum(1) / keep.sum(1)
    dg2[m] = (zg * keep).sum(1) / keep.sum(1)
print(
    f"  去掉越界单元格后 真实/生成 {dr2.mean() / dg2.mean():.3f}，单标量 AUC {roc_auc_score(y, -np.r_[dr2, dg2]):.3f}"
)
# 下限状态的联合结构：每款下限列数的离散程度；下限指示两两相关（phi）的平均
cr_, cg_ = fr_[:, has].sum(1), fg_[:, has].sum(1)
print(
    f"  每款下限列数 中位 {np.median(cr_):.1f}/{np.median(cg_):.1f}，"
    f"标准差 真实 {cr_.std():.2f}，生成 {cg_.std():.2f}"
)
iu = np.triu_indices(has.sum(), 1)
phi_r = np.nan_to_num(np.corrcoef(fr_[:, has].T.astype(float)))[iu]
phi_g = np.nan_to_num(np.corrcoef(fg_[:, has].T.astype(float)))[iu]
print(
    f"  下限指示两两相关 平均 真实 {phi_r.mean():.3f}，生成 {phi_g.mean():.3f}；"
    f"|phi|>0.3 的列对 {np.mean(np.abs(phi_r) > 0.3):.3f}/{np.mean(np.abs(phi_g) > 0.3):.3f}"
)
print(
    f"  单用每款下限列数区分（AUC，方向不定取 max）"
    f"{max(roc_auc_score(y, np.r_[cr_, cg_]), 1 - roc_auc_score(y, np.r_[cr_, cg_])):.3f}"
)
print(f"  总用时 {time.time() - t0:.0f} s")
