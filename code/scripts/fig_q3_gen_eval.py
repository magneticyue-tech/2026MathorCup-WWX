import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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

NEAR_COPY = ("G1", "G3")


def style_colors(styles):
    cmap = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Dark2").colors)
    return {s: cmap[i % len(cmap)] for i, s in enumerate(styles)}


def main():
    apply()
    js = load_json("q3_genval.json")
    m, perm = js["metrics"], js["permutation"]
    z = np.load(PROCESSED_DIR / "q3_pca.npz", allow_pickle=False)
    gens = list(GEN_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    x = np.arange(len(gens))
    for off, var, label, color in (
        (-0.2, "marg", "合并 C2ST", PALETTE["main"]),
        (0.2, "cond", "条件 C2ST（加风格独热）", "#56B4E9"),
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
    ax.set_xticks(
        x, [GEN_LABELS[g] + ("*" if g in NEAR_COPY else "") for g in gens], rotation=25, ha="right"
    )
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("C2ST AUC（5 次重复，95% 按啤酒 bootstrap 区间）")
    ax.set_title(
        "(a) 各生成器 C2ST AUC（* 生成池含近重复样本，AUC<0.5 不表示更接近真实）", fontsize=8
    )
    ax.legend(fontsize=7, frameon=False, loc="upper left")
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
    ax.set_xlabel(f"PC1（{evr[0]:.1%}）")
    ax.set_ylabel(f"PC2（{evr[1]:.1%}）")
    ax.set_title(
        f"(b) 潜空间 PCA（在真实啤酒上拟合）：描边大点为真实 {len(z['real'])} 款，"
        f"小点为每风格 {int(z['n_show'])} 个生成样本",
        fontsize=8,
    )
    ax.legend(
        fontsize=6,
        frameon=False,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        markerscale=0.8,
    )
    fig.tight_layout()
    print(save(fig, FIG_DIR, "fig_q3_gen_eval.pdf"))


if __name__ == "__main__":
    main()
