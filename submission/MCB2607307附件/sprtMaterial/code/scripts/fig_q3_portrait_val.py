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
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    a = E[:, names.index("rho_gen_realpred")]
    b = E[:, names.index("rho_realpred_obs")]
    c = E[:, names.index("rho_gen_obs")]
    ax.scatter(n[ok], a[ok], color=PALETTE["main"], label="生成画像 对 真实折外预测")
    ax.scatter(
        n[ok], b[ok], color=PALETTE["better"], marker="s", label="真实折外预测 对 观测均分（参照）"
    )
    ax.scatter(n[ok], c[ok], color=PALETTE["accent"], marker="^", label="生成画像 对 观测均分")
    for i in np.where(ok)[0]:
        ax.annotate(styles[i], (n[i], a[i]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("风格样本量（款）")
    ax.set_ylabel("50 维画像中位数的 Spearman")
    ax.set_title(f"画像验证（日期组外层划分；可评价风格 {int(ok.sum())}/22）")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, FIG_DIR, "fig_q3_portrait_val.pdf")


if __name__ == "__main__":
    main()
