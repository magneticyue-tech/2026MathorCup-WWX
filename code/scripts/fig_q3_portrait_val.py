import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402


def main():
    apply()
    z = np.load(PROCESSED_DIR / "q3_portrait_val.npz", allow_pickle=False)
    names = list(z["metric_names"])
    E, n, styles = z["style_metrics"], z["n_style"], list(z["styles"])
    ok = np.isfinite(E[:, 0])
    fig, ax = plt.subplots(figsize=(11, 7))
    a = E[:, names.index("rho_gen_realpred")]
    b = E[:, names.index("rho_realpred_obs")]
    c = E[:, names.index("rho_gen_obs")]
    ax.scatter(n[ok], a[ok], color=PALETTE["main"], label="生成画像 对 浅层梯度提升树折外预测")
    ax.scatter(
        n[ok],
        b[ok],
        color=PALETTE["better"],
        marker="s",
        label="浅层梯度提升树折外预测 对 品评小组观测均分（参照）",
    )
    ax.scatter(
        n[ok], c[ok], color=PALETTE["accent"], marker="^", label="生成画像 对 品评小组观测均分"
    )
    for i in np.where(ok)[0]:
        ax.annotate(styles[i], (n[i], a[i]), fontsize=9, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("风格样本量（啤酒数；日期组外层划分）", fontsize=9)
    ax.set_ylabel("50 维画像中位数的 Spearman 秩相关系数", fontsize=9)
    ax.tick_params(labelsize=9)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, FIG_DIR, "fig_q3_portrait_val.pdf")


if __name__ == "__main__":
    main()
