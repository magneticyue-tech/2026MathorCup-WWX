import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402

GEN_PANELS = [
    ("main.auc_marg", "合并 C2ST AUC"),
    ("main.tstr", "TSTR"),
    ("main.module_rho", "模块内 ρ"),
    ("main.wcorr_err", "风格内相关误差"),
]
TR_PANELS = [
    ("pils_tripel.alpha_median", "11 款 α* 中位数"),
    ("pils_tripel.p_median", "11 款 P(Tripel) 中位数"),
    ("all_pairs.alpha_median", "462 对 α* 中位数"),
    ("all_pairs.active_frac", "462 对约束起作用比例"),
]


def main():
    apply()
    df = pd.read_csv(PROCESSED_DIR / "q3_sensitivity.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    g = df[df["arm"].str.startswith("G-S")]
    for ax, (k, t) in zip(axes[0], GEN_PANELS):
        sub = g[g[f"{k}.diff"].notna()] if f"{k}.diff" in g else g.iloc[:0]
        y = np.arange(len(sub))[::-1]
        if len(sub):
            for yi, lo, hi in zip(y, sub[f"{k}.diff_lo"], sub[f"{k}.diff_hi"]):
                ax.plot([lo, hi], [yi, yi], color=PALETTE["main"], lw=1.4)
            ax.plot(sub[f"{k}.diff"], y, "o", color=PALETTE["main"], ms=4)
        ax.axvline(0, ls="--", color="#444444", lw=0.8)
        ax.set_yticks(y, sub["arm"], fontsize=7)
        ax.set_title(f"{t}：臂 − 基线（95% 配对区间）", fontsize=8)
    t = df[df["arm"].str.startswith("T-S") | (df["arm"] == "G-S8")]
    for ax, (k, ttl) in zip(axes[1], TR_PANELS):
        col = f"{k}.diff"
        sub = t[t[col].notna()] if col in t else t.iloc[:0]
        y = np.arange(len(sub))[::-1]
        ax.barh(y, sub[col] if len(sub) else [], color=PALETTE["main"])
        ax.axvline(0, ls="--", color="#444444", lw=0.8)
        ax.set_yticks(y, sub["arm"], fontsize=7)
        ax.set_title(f"{ttl}：臂 − 基线", fontsize=8)
    fig.tight_layout()
    save(fig, FIG_DIR, "fig_q3_sensitivity.pdf")


if __name__ == "__main__":
    main()
