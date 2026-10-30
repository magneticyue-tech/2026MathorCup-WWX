import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import (  # noqa: E402
    FIG_DIR,
    GEN_LABELS,
    PALETTE,
    PROCESSED_DIR,
    apply,
    load_json,
    save,
)


def style_colors(styles):
    cmap = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Dark2").colors)
    return {s: cmap[i % len(cmap)] for i, s in enumerate(styles)}


def main():
    apply()
    js = load_json("q3_genval.json")
    m, perm = js["metrics"], js["permutation"]
    z = np.load(PROCESSED_DIR / "q3_pca.npz", allow_pickle=False)
    gens = list(GEN_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    x = np.arange(len(gens))
    for off, var, label, color in (
        (-0.2, "marg", "合并两样本分类检验（C2ST）", PALETTE["main"]),
        (0.2, "cond", "条件两样本分类检验（C2ST，加风格独热）", "#56B4E9"),
    ):
        est = np.array([m[f"{g}.auc_{var}"]["estimate"] for g in gens])
        lo = np.array([m[f"{g}.auc_{var}"]["ci95"]["lo"] for g in gens])
        hi = np.array([m[f"{g}.auc_{var}"]["ci95"]["hi"] for g in gens])
        ax.bar(x + off, est, width=0.38, color=color, label=label)
        ax.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="none", ecolor="k", lw=0.9)
        for xi, g in zip(x, gens):
            q = perm.get(f"{g}.{var}", {}).get("perm_q95")
            if q is not None:
                ax.plot([xi + off - 0.19, xi + off + 0.19], [q, q], color=PALETTE["warn"], lw=1.4)
    rr = m["real.rr_auc_marg"]["estimate"]
    ax.axhline(0.5, ls="--", color="#444444", lw=0.9, label="理想值 0.5")
    ax.axhline(rr, ls=":", color=PALETTE["accent"], lw=1.2, label=f"真实对真实 {rr:.3f}")
    ax.plot([], [], color=PALETTE["warn"], lw=1.4, label="置换零分布 95% 分位")
    ax.set_xticks(x, [GEN_LABELS[g] for g in gens], rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("两样本分类检验曲线下面积（AUC），5 次重复，95% 按啤酒重抽区间")
    ax.set_title("(a)", loc="left", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    styles = [str(s) for s in z["styles"]]
    style = np.array([str(s) for s in z["style"]])
    col = style_colors(styles)
    for si, st in enumerate(styles):
        ax.scatter(*z["synth"][si].T, s=5, color=col[st], alpha=0.4, linewidths=0)
    for st in styles:
        pts = z["real"][style == st]
        ax.scatter(*pts.T, s=20, color=col[st], edgecolor="black", linewidth=0.4, label=st)
    evr = z["evr"]
    ax.set_xlabel(f"主成分 1（PC1，解释方差 {evr[0]:.1%}）")
    ax.set_ylabel(f"主成分 2（PC2，解释方差 {evr[1]:.1%}）")
    ax.set_title("(b)", loc="left", fontsize=11)
    proxies = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="white",
            mec="black",
            mew=0.4,
            ms=7,
            label=f"真实啤酒（{len(z['real'])} 款）",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color="grey",
            alpha=0.5,
            ms=3,
            label=f"生成样本（每风格 {int(z['n_show'])} 个）",
        ),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=proxies + handles,
        labels=[h.get_label() for h in proxies] + labels,
        fontsize=9,
        frameon=False,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.tight_layout()
    print(save(fig, FIG_DIR, "fig_q3_gen_eval.pdf"))


if __name__ == "__main__":
    main()
