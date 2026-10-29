import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import NAMED_STYLES  # noqa: E402
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402


def main():
    apply()
    z = np.load(PROCESSED_DIR / "q3_portraits.npz", allow_pickle=False)
    styles, dims = list(z["styles"]), list(z["dims"])
    x = np.arange(len(dims))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, st in zip(axes, NAMED_STYLES):
        si = styles.index(st)
        ax.fill_between(
            x,
            z["mu_q10"][si],
            z["mu_q90"][si],
            color=PALETTE["main"],
            alpha=0.2,
            label="生成画像 10%–90%",
        )
        ax.plot(x, z["mu_q50"][si], "-o", ms=2.5, color=PALETTE["main"], label="生成画像中位数")
        ax.plot(
            x,
            z["real_oof_q50"][si],
            "s",
            ms=3,
            color=PALETTE["better"],
            label="真实啤酒 Q1 折外预测中位数",
        )
        weak = z["weak_dim"]
        ax.plot(
            x[weak],
            np.full(weak.sum(), -0.15),
            "v",
            ms=3,
            color=PALETTE["ctrl"],
            label="化学模型未胜过 B1 的维度",
        )
        ax.set_ylabel("预测 0–6 分")
        ax.set_title(f"{st}（n={int(z['n_style'][si])}）")
        ax.set_ylim(-0.3, max(3.5, float(z["mu_q90"][si].max()) + 0.3))
    axes[0].legend(ncol=4, fontsize=7, loc="upper right")
    axes[-1].set_xticks(x, dims, rotation=90, fontsize=7)
    fig.tight_layout()
    save(fig, FIG_DIR, "fig_q3_portrait.pdf")


if __name__ == "__main__":
    main()
