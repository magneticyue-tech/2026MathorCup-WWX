import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    z = plotstyle.load_npz(args.src / "q1_mscale.npz")
    p = plotstyle.PALETTE
    m = z["m_grid"]
    raw = z["m_raw"][np.isfinite(z["m_raw"])]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax2 = ax.twinx()
    ax2.hist(raw, bins=np.arange(1, 7.01, 0.25), color=p["neutral"], alpha=0.35, density=True)
    ax2.set_ylabel("原始评分估计的 m 的密度（品评员 × 维度）", fontsize=9)
    ax2.tick_params(labelsize=9)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.plot(m, z["mae_A"], "o-", color=p["main"], label="有零点参照单元等级 MAE")
    ax.plot(m, z["bias_A"], "s--", color=p["better"], label="有零点参照单元平均偏差")
    ax.plot(
        m,
        z["share_changed"],
        "^:",
        color=p["accent"],
        label=f"相对 m={z['m_main']:g} 改变的有零点参照的等级比例",
    )
    ax.axhline(0, color="#444444", lw=0.6)
    ax.axvline(
        float(z["m_main"]),
        color=p["main"],
        ls="--",
        lw=0.8,
        label=f"交付设定 m={z['m_main']:g}（竖虚线）",
    )
    q = np.percentile(raw, [25, 50, 75])
    ax.axvspan(q[0], q[2], color=p["warn"], alpha=0.12)
    ax.text(
        4.8,
        ax.get_ylim()[0] * 0.55,
        f"原始评分 m 中位数 {q[1]:.2f}\n四分位 [{q[0]:.2f}, {q[2]:.2f}]（黄带）",
        ha="left",
        va="center",
        fontsize=9,
    )
    ax.set_xlabel("尺度参数 m = E[x²]/E[x]", fontsize=9)
    ax.set_ylabel("只用附件 4 的两阶段方案局部参照相对实测等级的误差", fontsize=9)
    ax.set_xlim(1, 7)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    print(plotstyle.save(fig, args.out, "fig_q1_m_sensitivity.pdf"))


if __name__ == "__main__":
    main()
