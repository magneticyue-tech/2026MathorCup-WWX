import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, GEN_LABELS, PALETTE, apply, load_json, save  # noqa: E402


def main():
    apply()
    js = load_json("q3_genval.json")
    m, perm = js["metrics"], js["permutation"]
    gens = list(GEN_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, var, panel in (
        (axes[0], "marg", "(a) 合并两样本分类检验（C2ST）"),
        (axes[1], "cond", "(b) 条件两样本分类检验（C2ST，加风格独热）"),
    ):
        x = np.arange(len(gens))
        est = [m[f"{g}.auc_{var}"]["estimate"] for g in gens]
        lo = [m[f"{g}.auc_{var}"]["ci95"]["lo"] for g in gens]
        hi = [m[f"{g}.auc_{var}"]["ci95"]["hi"] for g in gens]
        q95 = [perm.get(f"{g}.{var}", {}).get("perm_q95", np.nan) for g in gens]
        colors = [PALETTE["ctrl"] if g in ("G1", "G3") else PALETTE["main"] for g in gens]
        ax.bar(x, est, color=colors, alpha=0.85)
        ax.errorbar(
            x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="none", ecolor="k", lw=1
        )
        for xi, q in zip(x, q95):
            ax.plot([xi - 0.4, xi + 0.4], [q, q], color=PALETTE["warn"], lw=1.5)
        rr = m[f"real.rr_auc_{var}"]["estimate"]
        ax.axhline(rr, ls=":", color=PALETTE["accent"], lw=1.2, label=f"真实对真实参照 {rr:.3f}")
        ax.axhline(0.5, ls="--", color="#444444", lw=0.8)
        ax.set_xticks(x, [GEN_LABELS[g] for g in gens], rotation=30, ha="right")
        ax.tick_params(axis="x", labelsize=9)
        ax.set_title(panel, loc="left", fontsize=11)
        ax.set_ylim(0.2, 1.0)
        ax.plot([], [], color=PALETTE["warn"], lw=1.5, label="置换零分布 95% 分位")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("两样本分类检验曲线下面积（AUC），5 次重复均值，95% 区间")
    save(fig, FIG_DIR, "fig_q3_c2st.pdf")


if __name__ == "__main__":
    main()
