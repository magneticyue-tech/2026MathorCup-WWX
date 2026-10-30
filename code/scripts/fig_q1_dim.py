"""图：逐维增益与可预测性（fig:q1-dim）。

左栏：主模型减风格收缩基线 B1 的逐维等级 MAE 差（负值 = 主模型更优），
     零线区分胜负，并标注各组胜出维度数。
右栏：逐维连续 R2（点），配同维度半组信度（竖线）作噪声参照。
50 个维度按六个维度组（DIM_GROUPS）排列，组间画分隔线。

输入：code/data/processed/q1/q1_metrics.npz
输出：figures/q1/q1_dim.pdf（并复制一份到 paper/figures/ 供 LaTeX 引用）
运行：python code/scripts/fig_q1_dim.py
"""
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q1.paths import FIG_DIR, PROCESSED_DIR, ROOT  # noqa: E402

NPZ = PROCESSED_DIR / "q1_metrics.npz"
PAPER_FIG_DIR = ROOT / "paper" / "figures"


def main() -> None:
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    p = plotstyle.PALETTE

    z = np.load(NPZ, allow_pickle=True)
    dims_all = [str(d) for d in z["dims"]]
    idx = {d: i for i, d in enumerate(dims_all)}
    mae_main = z["main__median__mae_dim"]
    mae_b1 = z["b1__median__mae_dim"]
    r2 = z["r2_dim"]
    rel = z["reliability"]

    # 按维度组排列：Aroma, Flavour, Defect, Spice, Sour, Basic
    order, groups = [], []
    for g, ds in DIM_GROUPS.items():
        start = len(order)
        order += [d for d in dims_all if d in set(ds)]
        groups.append((g, start, len(order)))
    assert len(order) == 50, len(order)
    sel = [idx[d] for d in order]

    diff = (mae_main - mae_b1)[sel]  # <0 = 主模型 MAE 更低 = 胜出
    r2s, rels = r2[sel], rel[sel]
    wins = diff < 0
    n_win = int(wins.sum())
    print(f"主模型胜出 {n_win}/50 维；MAE main={mae_main.mean():.3f} B1={mae_b1.mean():.3f}")

    n = len(order)
    y = np.arange(n)
    short = [d.replace("A_", "").replace("F_", "") for d in order]

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(13, 8.8), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]}
    )

    # ---- 左栏：逐维 MAE 差 ----
    colors = [p["main"] if w else p["ctrl"] for w in wins]
    axl.hlines(y, 0, diff, color=colors, lw=1.6)
    axl.plot(diff, y, "o", color="white", mec=colors, ms=4.2, mew=1.2, zorder=3)
    axl.axvline(0, ls="--", color="#444444", lw=0.9)
    axl.invert_yaxis()
    axl.set_yticks(y)
    axl.set_yticklabels(short, fontsize=6.5)
    axl.set_xlabel("等级 MAE 差（主模型 − 风格收缩基线 B1）")
    axl.grid(axis="y", visible=False)
    axl.set_title(f"逐维等级 MAE 差（主模型胜出 {n_win}/50 维）", fontsize=10)

    # 各组胜出数标注（右缘）
    for g, s, e in groups:
        w = int(wins[s:e].sum())
        yc = (s + e - 1) / 2
        axl.text(
            0.985,
            yc,
            f"{g}  {w}/{e - s}",
            transform=axl.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2},
        )

    # ---- 右栏：逐维 R2 与半组信度 ----
    axr.plot(r2s, y, "o", color=p["main"], ms=3.6, label="连续 $R^2$")
    for yi, xr in zip(y, rels):  # 半组信度：短竖线
        axr.plot([xr, xr], [yi - 0.32, yi + 0.32], color=p["ctrl"], lw=1.3)
    axr.axvline(0, ls="--", color="#444444", lw=0.9)
    axr.set_xlabel("连续 $R^2$  与  半组信度（竖线）")
    axr.grid(axis="y", visible=False)
    axr.set_title("逐维连续 $R^2$ 与半组信度", fontsize=10)
    axr.legend(loc="lower right", fontsize=8, frameon=False)

    # 组间分隔线（两栏同步）
    for _, s, _e in groups[1:]:
        for ax in (axl, axr):
            ax.axhline(s - 0.5, color="#CCCCCC", lw=0.7)

    fig.tight_layout(w_pad=1.0)
    out = plotstyle.save(fig, args.out, "q1_dim.pdf")
    print("已保存", out)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, PAPER_FIG_DIR / "q1_dim.pdf")
    print("已复制到", PAPER_FIG_DIR / "q1_dim.pdf")


if __name__ == "__main__":
    main()
