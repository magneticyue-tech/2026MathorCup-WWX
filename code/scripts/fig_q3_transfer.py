import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402

CAT_LABELS = {
    "reach": "达到参照范围",
    "improved": "改善",
    "kept": "保持原谱",
    "not_improved": "未改善",
}
CAT_COLORS = {
    "reach": "#009E73",
    "improved": "#56B4E9",
    "kept": "#999999",
    "not_improved": "#D55E00",
}


def closed(v):
    return np.append(v, v[:1])


def main():
    apply()
    z = np.load(PROCESSED_DIR / "q3_pca.npz", allow_pickle=False)
    tab = pd.read_csv(PROCESSED_DIR / "q3_transfer_pils_tripel.csv")
    por = pd.read_csv(PROCESSED_DIR / "q3_transfer_portraits.csv")
    tr = np.load(PROCESSED_DIR / "q3_transfer.npz", allow_pickle=False)
    src, tgt = str(z["src_style"]), str(z["tgt_style"])
    ref = float(tr["ref_q25"][[str(s) for s in tr["styles"]].index(tgt)])
    style = np.array([str(s) for s in z["style"]])
    rows = z["src_rows"].astype(int)
    tab = tab.set_index("row_index").loc[rows].reset_index()

    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1])

    ax = fig.add_subplot(gs[0])
    for st, col in ((src, PALETTE["main"]), (tgt, PALETTE["better"])):
        ax.scatter(
            *z["real"][style == st].T,
            s=22,
            color=col,
            alpha=0.35,
            marker="o" if st == src else "s",
            label=f"真实 {st}",
        )
    for k, cat in enumerate(tab["category"]):
        tj = z["traj"][k]
        a_idx = int(np.argmin(np.abs(z["alphas"] - z["alpha_star"][k])))
        ax.plot(*tj.T, color="#bbbbbb", lw=0.8, ls=":")
        ax.plot(*tj[: a_idx + 1].T, color=CAT_COLORS[cat], lw=1.4)
        ax.scatter(
            *z["end"][k],
            marker="*",
            s=70,
            color=CAT_COLORS[cat],
            edgecolor="black",
            lw=0.4,
            zorder=4,
        )
    sm = z["style_mean"][[str(s) for s in z["styles"]].index(tgt)]
    ax.scatter(
        *sm,
        marker="X",
        s=90,
        color=PALETTE["better"],
        edgecolor="black",
        zorder=5,
        label=f"{tgt} 风格均值",
    )
    for cat, lab in CAT_LABELS.items():
        if cat in set(tab["category"]):
            ax.plot([], [], color=CAT_COLORS[cat], lw=1.4, label=f"可行路径·{lab}")
    ax.plot([], [], color="#bbbbbb", lw=0.8, ls=":", label="不可行部分（至投影终点）")
    evr = z["evr"]
    ax.set_xlabel(f"PC1（{evr[0]:.1%}）")
    ax.set_ylabel(f"PC2（{evr[1]:.1%}）")
    ax.set_title(f"(a) 潜空间 PCA：{len(rows)} 款 {src} → 迁移终点（★）", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="best")

    ax = fig.add_subplot(gs[1])
    x = np.arange(len(tab))
    w = 0.26
    ax.bar(x - w, tab["p_src"], w, color="#bbbbbb", label="源谱")
    ax.bar(x, tab["p_t0"], w, color="#E69F00", label="T0 无约束平移")
    ax.bar(x + w, tab["p_main"], w, color=[CAT_COLORS[c] for c in tab["category"]])
    for cat, lab in CAT_LABELS.items():
        if cat in set(tab["category"]):
            ax.bar([np.nan], [np.nan], color=CAT_COLORS[cat], label=f"主方法（α*）·{lab}")
    ax.axhline(
        ref, ls="--", color="#444444", lw=0.9, label=f"真实 {tgt} 的 P(t) 25% 分位 {ref:.3f}"
    )
    ax.set_xticks(x, [str(b) for b in tab["beer"]])
    ax.set_xlabel("源啤酒编号")
    ax.set_ylabel(f"判别器给出的 P({tgt})")
    ax2 = ax.twinx()
    ax2.plot(x, tab["alpha_star"], "kD", ms=4, label="α*")
    ax2.set_ylim(0, 1.6)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_ylabel("最大可行步长 α*")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("(b) 逐款 α* 与目标风格概率（主方法柱色示结论类别）", fontsize=8.5)
    ax.grid(axis="x", visible=False)

    reach = tab[tab["category"] == "reach"]
    pick = int(
        (reach if len(reach) else tab).sort_values("p_main", ascending=False)["beer"].iloc[0]
    )
    dims = [c for c in por.columns if c not in ("beer", "profile")]
    order = [c for members in DIM_GROUPS.values() for c in members if c in dims]
    ang = closed(np.linspace(0, 2 * np.pi, len(order), endpoint=False))
    ax = fig.add_subplot(gs[2], projection="polar")
    sub = por[por["beer"] == pick].set_index("profile")
    for prof, lab, col, ls in (
        ("before", "迁移前", PALETTE["main"], "-"),
        ("after", "迁移后", "#009E73", "-"),
        ("target_real_oof", f"真实 {tgt} 折外预测中位数", PALETTE["better"], "--"),
    ):
        ax.plot(
            ang, closed(sub.loc[prof, order].to_numpy(float)), color=col, ls=ls, lw=1.3, label=lab
        )
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(order, fontsize=4.2)
    ax.tick_params(axis="y", labelsize=6)
    r = tab[tab["beer"] == pick].iloc[0]
    ax.set_title(
        f"(c) 代表款 {pick}：α*={r['alpha_star']:.2f}，"
        f"P({tgt}) {r['p_src']:.3f} → {r['p_main']:.3f}",
        fontsize=8.5,
        pad=16,
    )
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.tight_layout()
    print(save(fig, FIG_DIR, "fig_q3_transfer.pdf"))


if __name__ == "__main__":
    main()
