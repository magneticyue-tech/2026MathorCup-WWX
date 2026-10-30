import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q2 import config as C  # noqa: E402

from src import paperfig as pf  # noqa: E402

MARKS = {1: "o", 2: "s", 3: "^"}
HIGHLIGHT = (("pH", "lactic"), ("pH", "acidity"), ("ethanol..v.v.", "F_malt_all"))


def main():
    pf.apply()
    d = pd.read_csv(C.Q2_DIR / "q2_marginal.csv")
    ci0 = d["flag_style_ci0"].astype(bool)
    fig, ax = plt.subplots(figsize=(pf.TEXTWIDTH * 0.72, 3.9))
    lim = 0.85
    ax.plot([-lim, lim], [-lim, lim], color=pf.MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.axhline(0, color=pf.MUTED, lw=0.7)
    ax.axvline(0, color=pf.GRID, lw=0.6)
    for k, mk in MARKS.items():
        for flag, col in ((False, pf.BLUE), (True, pf.LIGHT)):
            s = d[(d["rank"] == k) & (ci0 == flag)]
            ax.scatter(
                s["r_pooled"],
                s["r_style"],
                marker=mk,
                s=18,
                color=col,
                ec="white",
                lw=0.4,
                zorder=3,
            )
    for comp, dim in HIGHLIGHT:
        r = d[(d["compound"] == comp) & (d["dimension"] == dim)]
        if len(r):
            x, y = float(r["r_pooled"].iloc[0]), float(r["r_style"].iloc[0])
            ax.annotate(
                f"{comp} → {dim}",
                (x, y),
                xytext=(6, 8 if y >= 0 else -10),
                textcoords="offset points",
                fontsize=6.5,
                color=pf.ORANGE,
                arrowprops={"arrowstyle": "-", "color": pf.ORANGE, "lw": 0.6},
            )
    same = (d["r_pooled"] * d["r_style"] > 0).mean()
    ratio = (d["r_style"] / d["r_pooled"]).median()
    n_small = int((d["r_style"].abs() < 0.1).sum())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, 0.7)
    ax.set_xlabel("合并边际 Spearman $r_{jd}$")
    ax.set_ylabel("风格内 $r^{(s)}_{jd}$")
    ax.set_title(
        f"{len(d)} 项中同号 {same:.0%}，比值中位数 {ratio:.2f}，$|r^{{(s)}}|<0.1$ 共 {n_small} 项",
        fontsize=7.5,
        color=pf.MUTED,
        loc="left",
    )
    handles = [
        Line2D([], [], marker=MARKS[k], ls="", color=pf.MUTED, ms=5, label=f"第 {k} 名")
        for k in MARKS
    ]
    handles += [
        Line2D([], [], marker="o", ls="", color=pf.BLUE, ms=5, label="风格内区间不含 0"),
        Line2D([], [], marker="o", ls="", color=pf.LIGHT, ms=5, label="风格内区间含 0"),
        Line2D([], [], color=pf.MUTED, lw=0.7, ls=(0, (3, 2)), label="$y=x$"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, ncol=2, handletextpad=0.3)
    fig.tight_layout()
    print(pf.save(fig, "q2", "fig_q2_scatter.pdf"))


if __name__ == "__main__":
    main()
