import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fig_q3_gen_metrics import bars  # noqa: E402
from src.q3.plotting import FIG_DIR, GEN_LABELS, PALETTE, apply, load_json, save  # noqa: E402


def main():
    apply()
    m = load_json("q3_genval.json")["metrics"]
    gens = list(GEN_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    g1 = m["G1.corr_err"]["estimate"]
    bars(axes[0], m, "corr_err", gens, g1, f"G1 抽样误差水平 {g1:.3f}")
    axes[0].set_ylabel("Spearman 矩阵逐元素绝对差均值")
    axes[0].set_title("相关误差")
    g1w = m["G1.wcorr_err"]["estimate"]
    bars(axes[1], m, "wcorr_err", gens, g1w, f"G1 抽样误差水平 {g1w:.3f}")
    axes[1].set_title("风格内相关误差")
    rr = m["real.real_module_rho"]["estimate"]
    bars(axes[2], m, "module_rho", gens, rr, f"真实留出 {rr:.3f}")
    axes[2].set_ylabel("Q2 模块内化合物对平均 Spearman")
    axes[2].set_title("模块内 ρ")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    axes[2].axhline(rr - 0.10, ls=":", color=PALETTE["warn"], lw=1)
    fig.tight_layout()
    save(fig, FIG_DIR, "fig_q3_corr.pdf")


if __name__ == "__main__":
    main()
