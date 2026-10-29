import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import NAMED_STYLES  # noqa: E402
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402

TOP = 10


def main():
    apply()
    chem = pd.read_csv(PROCESSED_DIR / "q3_fingerprint.csv")
    sens = pd.read_csv(PROCESSED_DIR / "q3_sensory_fingerprint.csv")
    fig, axes = plt.subplots(len(NAMED_STYLES), 2, figsize=(12, 3.3 * len(NAMED_STYLES)))
    for row, st in zip(axes, NAMED_STYLES):
        c = chem[(chem["style"] == st) & (chem["rank"] <= TOP)].sort_values("rank", ascending=False)
        y = np.arange(len(c))
        cols = [PALETTE["better"] if v > 0 else PALETTE["main"] for v in c["delta"]]
        cols = [PALETTE["neutral"] if z else k for z, k in zip(c["ci_covers_zero"], cols)]
        ax = row[0]
        ax.barh(y, c["delta"], color=cols)
        ax.errorbar(
            c["delta"],
            y,
            xerr=[c["delta"] - c["delta_lo"], c["delta_hi"] - c["delta"]],
            fmt="none",
            ecolor="k",
            lw=0.8,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(c["compound"], fontsize=6.5)
        ax.axvline(0, color="#444444", lw=0.6)
        n = int(c["n_style"].iloc[0])
        ax.set_title(
            f"{st}（{n} 款）化学指纹前 {TOP} 项：潜空间均值偏移 δ（90% bootstrap 区间）", fontsize=8
        )
        ax.grid(axis="y", visible=False)

        s = (
            sens[sens["style"] == st]
            .nsmallest(TOP, "abs_rank")
            .sort_values("abs_rank", ascending=False)
        )
        y = np.arange(len(s))
        cols = [PALETTE["better"] if v > 0 else PALETTE["main"] for v in s["deviation_sd"]]
        ax = row[1]
        ax.barh(y, s["deviation_sd"], color=cols, hatch=None)
        for yi, (v, weak) in enumerate(zip(s["deviation_sd"], s["weak_dim"])):
            if weak:
                ax.barh(yi, v, color="none", edgecolor="#222222", hatch="///", lw=0)
        ax.set_yticks(y)
        ax.set_yticklabels(s["dim"], fontsize=6.5)
        ax.axvline(0, color="#444444", lw=0.6)
        ax.set_title(
            f"{st} 感官指纹前 {TOP} 维：生成画像中位数相对全体的偏移（单位：维度 SD）", fontsize=8
        )
        ax.grid(axis="y", visible=False)
    axes[-1, 0].set_xlabel("δ（正为高于全体，灰色为区间含 0）")
    axes[-1, 1].set_xlabel("偏移 / SD（斜线为问题一未优于 B1 的弱维度，只作定性参考）")
    fig.tight_layout()
    print(save(fig, FIG_DIR, "fig_q3_fingerprint_pair.pdf"))


if __name__ == "__main__":
    main()
