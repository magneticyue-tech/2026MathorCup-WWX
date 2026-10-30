import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import NAMED_STYLES  # noqa: E402
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402


def main():
    apply()
    df = pd.read_csv(PROCESSED_DIR / "q3_fingerprint.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k, (ax, st) in enumerate(zip(axes, NAMED_STYLES)):
        d = df[df["style"] == st].sort_values("rank", ascending=False)
        y = np.arange(len(d))
        col = [
            PALETTE["ctrl"] if c else (PALETTE["better"] if v > 0 else PALETTE["main"])
            for c, v in zip(d["ci_covers_zero"], d["delta"])
        ]
        ax.barh(y, d["delta"], color=col)
        ax.errorbar(
            d["delta"],
            y,
            xerr=[d["delta"] - d["delta_lo"], d["delta_hi"] - d["delta"]],
            fmt="none",
            ecolor="k",
            lw=0.8,
        )
        ax.set_yticks(y, d["compound"], fontsize=9)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_xlabel("标准化潜空间偏移 δ（风格内标准差）", fontsize=9)
        ax.set_title(f"({chr(97 + k)}) {st}（化学指纹前 10 项）", fontsize=11, loc="left")
    handles = [Patch(facecolor=PALETTE["neutral"], label="bootstrap 90% 区间含 0")]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, FIG_DIR, "fig_q3_fingerprint.pdf")


if __name__ == "__main__":
    main()
