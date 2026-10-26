# -*- coding: utf-8 -*-
"""MathorCup 2026 A题 —— 数据预处理章节的统计量与图表生成"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

ROOT = Path(r"C:\Users\Administrator\Desktop\MathorCupWWX")
AUDIT = ROOT / "results" / "data_audit"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RAW1 = Path(r"C:\Users\Administrator\Documents\xwechat_files"
            r"\wxid_2evcckfezb4k22_806a\msg\file\2026-10\1(1).csv")

out = {}

# ------------------------------------------------ 附件1 化合物矩阵
d1 = pd.read_csv(RAW1)
ident = d1.columns[0]
chem_cols = [c for c in d1.columns if c != ident]
na_per_col = d1[chem_cols].isna().sum()
na_per_col = na_per_col[na_per_col > 0].sort_values(ascending=False)
out["a1"] = {
    "rows": int(d1.shape[0]), "cols": int(d1.shape[1]),
    "chem_cols": len(chem_cols),
    "na_total": int(d1[chem_cols].isna().sum().sum()),
    "na_cells_total": int(d1[chem_cols].size),
    "na_rate": float(d1[chem_cols].isna().sum().sum() / d1[chem_cols].size),
    "na_cols": int(na_per_col.size),
    "na_detail": {k: int(v) for k, v in na_per_col.items()},
}

# ------------------------------------------------ 附件4 / 附件7 剖析
for tag, f in (("a4", "4_columns.csv"), ("a7", "7_columns.csv")):
    p = pd.read_csv(AUDIT / f)
    num = p[p["fully_numeric"] == True]  # noqa: E712
    tukey = num["tukey_1_5_iqr_flags"].astype(float)
    q25 = pd.to_numeric(num["q25"], errors="coerce")
    q75 = pd.to_numeric(num["q75"], errors="coerce")
    iqr0 = num[(q25 == q75) & q25.notna()]
    full_empty = p[p["missing"] == 250]
    floored = num[num["negative"].astype(float) > 125]
    out[tag] = {
        "cols": int(p.shape[0]), "num_cols": int(num.shape[0]),
        "na_total": int(p["missing"].astype(int).sum()),
        "cols_with_na": int((p["missing"].astype(int) > 0).sum()),
        "tukey_total": int(tukey.sum()),
        "cols_with_tukey": int((tukey > 0).sum()),
        "tukey_rate": float(tukey.sum() / (num.shape[0] * 250)),
        "iqr0_cols": int(iqr0.shape[0]),
        "iqr0_names": list(iqr0["column"])[:10],
        "full_empty_cols": list(full_empty["column"]),
        "floored_cols": int(floored.shape[0]),
        "num": num,
    }

# ------------------------------------------------ 对数底数判定
tv = pd.read_csv(AUDIT / "transform_verification.csv")
tv = tv[tv["positive_pairs"] > 0]
# 仅保留经交叉核验确认做了对数变换的化合物列；理化参数（乙醇、色度、
# 苦度、pH、CO2）按题面保留原始量纲，不参与底数判定
tv = tv[tv["log10_mae"] < 1e-6].sort_values("log10_mae")
out["log"] = {
    "fields": int(tv.shape[0]),
    "log10_mae_max": float(tv["log10_mae"].max()),
    "log10_mae_mean": float(tv["log10_mae"].mean()),
    "ln_mae_min": float(tv["ln_mae"].min()),
    "ln_mae_max": float(tv["ln_mae"].max()),
    "ln_mae_mean": float(tv["ln_mae"].mean()),
    "identity_mae_min": float(tv["identity_mae"].min()),
    "identity_mae_max": float(tv["identity_mae"].max()),
    "identity_mae_mean": float(tv["identity_mae"].mean()),
    "floor_ratio_min": float(tv["floor_over_min_positive"].min()),
    "floor_ratio_max": float(tv["floor_over_min_positive"].max()),
    "n_with_floor": int((tv["nonpositive_log_distinct"] > 0).sum()),
}

# ------------------------------------------------ 检出限地板
ft = pd.read_csv(AUDIT / "floors_and_ties.csv").sort_values(
    "min_fraction", ascending=False)
out["floor"] = {
    "n_fields": int(ft.shape[0]),
    "n_heavy": int((ft["min_fraction"] >= 0.10).sum()),
    "n_any": int((ft["min_fraction"] > 0).sum()),
    "top": [{"field": r.field, "frac": float(r.min_fraction),
             "uniq": int(r.unique_values)}
            for r in ft.head(8).itertuples()],
}

miss_cols = list(na_per_col.index)
pos = d1[miss_cols].isna().astype(int).values

# ================================================= 图1 缺失值
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8),
                         gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
labels = [c.replace(".", "-") for c in na_per_col.index]
bars = ax.barh(range(len(na_per_col))[::-1], na_per_col.values,
               color="#D95F4B", edgecolor="black", linewidth=.4)
ax.set_yticks(range(len(na_per_col))[::-1])
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("缺失个数")
ax.set_title("(a) 附件1 中存在缺失的 9 个化合物变量", fontsize=10)
for i, v in enumerate(na_per_col.values):
    ax.text(v + .4, len(na_per_col) - 1 - i, f"{v}  ({v/250:.1%})",
            va="center", fontsize=7.5)
ax.set_xlim(0, max(na_per_col.values) * 1.45)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
rr, cc = np.nonzero(pos)          # rr=样本行号, cc=变量列号
ax.scatter(rr, cc, s=11, c="#2E6F9E", alpha=.75, linewidths=0)
ax.set_yticks(range(len(miss_cols)))
ax.set_yticklabels([c.replace(".", "-") for c in miss_cols], fontsize=8)
ax.set_xlabel("啤酒样本编号")
ax.set_title(f"(b) 缺失位置分布（{out['a1']['na_total']} 个空值）", fontsize=10)
ax.set_ylim(-.8, len(miss_cols) - .2)
ax.set_xlim(-8, 255)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_missing.png", bbox_inches="tight")
plt.close(fig)

# ================================================= 图2 对数底数判定
fig, ax = plt.subplots(figsize=(6.2, 3.6))
names = [r.field.replace("_", "-") for r in tv.head(10).itertuples()]
x = np.arange(len(names)); w = .26
ax.bar(x - w, tv["log10_mae"].head(10), w, label=r"$\log_{10}$",
       color="#2E8B57", edgecolor="black", linewidth=.3)
ax.bar(x, tv["ln_mae"].head(10), w, label=r"$\ln$",
       color="#D99B3B", edgecolor="black", linewidth=.3)
ax.bar(x + w, tv["identity_mae"].head(10), w, label="恒等变换",
       color="#B44B4B", edgecolor="black", linewidth=.3)
ax.set_yscale("log")
ax.set_ylabel("反演误差 MAE（对数尺度）")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
ax.legend(frameon=False, ncol=3, fontsize=9)
ax.set_title(r"附件1 对数变换底数判定：仅 $\log_{10}$ 使反演误差降至机器精度",
             fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(1e-9, ls="--", c="grey", lw=.8)
ax.text(len(names) - .4, 2e-9, "机器精度", fontsize=7.5, color="grey", ha="right")
fig.tight_layout()
fig.savefig(FIG / "fig_logbase.png", bbox_inches="tight")
plt.close(fig)

# ================================================= 图3 异常值
num7 = out["a7"].pop("num")
tukey7 = num7["tukey_1_5_iqr_flags"].astype(float)
q25 = pd.to_numeric(num7["q25"], errors="coerce")
q75 = pd.to_numeric(num7["q75"], errors="coerce")
flag_rate = tukey7 / 250.0
degenerate = (q25 == q75) & q25.notna()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
ax = axes[0]
SENS_MASK = ~num7["column"].isin({"beer", "beer_id", "tasting_category_fine"})
degenerate = degenerate & SENS_MASK
tukey7_s = tukey7[SENS_MASK]
bins = np.linspace(0, 1, 21)
ax.hist(tukey7_s[~degenerate[SENS_MASK]] / 250, bins=bins, alpha=.75,
        label=f"IQR>0（{int((~degenerate[SENS_MASK]).sum())} 列）",
        color="#2E6F9E", edgecolor="black", linewidth=.3)
ax.hist(tukey7_s[degenerate[SENS_MASK]] / 250, bins=bins, alpha=.75,
        label=f"IQR=0（{int(degenerate[SENS_MASK].sum())} 列）",
        color="#D95F4B", edgecolor="black", linewidth=.3)
ax.set_xlabel("被朴素 Tukey 规则标记的比例")
ax.set_ylabel("感官维度个数")
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("(a) 附件7 各维度被标记为异常值的比例", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ID_COLS = {"beer", "beer_id", "tasting_category_fine"}
sens = num7[~num7["column"].isin(ID_COLS)].copy()
dmask = ((q25 == q75) & q25.notna())[sens.index]
pos_ratio = sens["positive"].astype(float) / 250
flag_ratio = sens["tukey_1_5_iqr_flags"].astype(float) / 250
ax.scatter(pos_ratio[~dmask], flag_ratio[~dmask], s=22, c="#2E6F9E",
           alpha=.7, linewidths=0, label=f"IQR>0（{int((~dmask).sum())} 列）")
ax.scatter(pos_ratio[dmask], flag_ratio[dmask], s=26, c="#D95F4B",
           alpha=.8, linewidths=0, marker="s", label=f"IQR=0（{int(dmask.sum())} 列）")
ax.plot([0, 1], [0, 1], ls="--", c="black", lw=.9)
ax.text(.62, .56, "落在该线上 =\n真实信号被全部标记", fontsize=8, rotation=38)
ax.set_xlabel("真实评分信号占比")
ax.set_ylabel("被判定为异常值的占比")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("(b) 退化维度的真实信号几乎被整体误判", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_outlier.png", bbox_inches="tight")
plt.close(fig)

# ================================================= 图4 检出限地板
fig, ax = plt.subplots(figsize=(7.2, 3.4))
top = ft.head(12)
ax.barh(range(len(top))[::-1], top["min_fraction"] * 100,
        color="#7B5EA7", edgecolor="black", linewidth=.4)
ax.set_yticks(range(len(top))[::-1])
ax.set_yticklabels(top["field"].str.replace("_", "-"), fontsize=8.5)
ax.set_xlabel("落在地板值上的样本占比 / %")
ax.set_title("附件1 中检出限地板效应显著的化合物", fontsize=10)
for i, v in enumerate((top["min_fraction"] * 100).values):
    ax.text(v + 1.2, len(top) - 1 - i, f"{v:.1f}%", va="center", fontsize=7.5)
ax.set_xlim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_floor.png", bbox_inches="tight")
plt.close(fig)

# ================================================= 图5 流程图
fig, ax = plt.subplots(figsize=(9.2, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, text, fc, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec="black", lw=.6, alpha=.92))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.45)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=.9, color="black"))


box(1, 18, 15, 11, "原始附件\n1/2/3/4/5/6/7/8", "#E8EDF2")
box(21, 29, 21, 14, "缺失值处理\n· 定位 NA 并分类\n· 全空列直接剔除\n· 浅层 GBDT 填补",
    "#D6E8F5")
box(21, 7, 21, 14, "异常值处理\n· Tukey 初筛\n· 识别 IQR 退化维度\n· 改用地板感知判据",
    "#F7DFD8")
box(47, 18, 18, 11, "数据校准\n· log10 反演核验\n· 评委零点归一", "#E4E1F2")
box(70, 29, 20, 13, "附件1 化合物\n训练集\n（化学→评分）", "#D9EEDD")
box(70, 4, 20, 13, "标准化感官\n面板\n（0–6 还原）", "#D9EEDD")
arrow(16, 23.5, 21, 33); arrow(16, 23.5, 21, 14)
arrow(42, 36, 47, 26); arrow(42, 14, 47, 21)
arrow(65, 26, 70, 33); arrow(65, 21, 70, 11)
ax.text(50, 1.2, "图5  数据预处理总体流程", ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig_flow.png", bbox_inches="tight")
plt.close(fig)

# ================================================= 控制台输出
for k in ("log", "floor"):
    out[k].pop("top", None)
out["a4"].pop("num", None)
out["a7"]["iqr0_names"] = out["a7"]["iqr0_names"][:6]
print(json.dumps(out, ensure_ascii=False, indent=2))
print("\n图件已保存至:", FIG)
for f in sorted(FIG.glob("fig_*.png")):
    print("  ", f.name)
