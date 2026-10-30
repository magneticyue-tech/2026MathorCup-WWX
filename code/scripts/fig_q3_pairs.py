import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.fullmodel import fit_full, load_final_config, variant_data  # noqa: E402
from src.q3.paths import PROCESSED_DIR  # noqa: E402

from src import paperfig as pf  # noqa: E402


def main():
    pf.apply()
    p = pd.read_csv(PROCESSED_DIR / "q3_transfer_pairs.csv")
    chem, _ = variant_data("base")
    gen, _ = fit_full(chem, load_final_config(PROCESSED_DIR / "q3_select.json"))
    styles = list(chem.styles)
    mu = np.stack([gen.mu[s] for s in styles])
    order = [styles[i] for i in leaves_list(linkage(mu, method="average"))]

    M = p.pivot(index="src", columns="tgt", values="frac_ref").reindex(index=order, columns=order)
    A = p.groupby("src")["alpha_star_mean"].mean().reindex(order)
    n = p.groupby("src")["n_src"].first().reindex(order)

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(pf.TEXTWIDTH, 5.6), gridspec_kw={"width_ratios": [5.2, 1]}, sharey=True
    )
    im = ax.imshow(
        M.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="none"
    )
    for i in range(len(order)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="white", zorder=2))
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6.3)
    ax.set_yticks(range(len(order)), [f"{s}  {int(n[s])}" for s in order], fontsize=6.3)
    ax.set_xlabel("目标风格")
    ax.set_ylabel("源风格与款数")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.3, 0.06, 0.45, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("源风格啤酒达到参照范围的比例")

    axb.barh(range(len(order)), A.to_numpy(), color=pf.BLUE, height=0.7)
    axb.set_xlim(0, 1)
    axb.set_xticks([0, 0.5, 1])
    axb.set_xlabel("平均 $\\alpha^*$")
    axb.grid(axis="x", visible=True)
    axb.set_axisbelow(True)
    axb.tick_params(axis="y", length=0)
    fig.text(
        0.01,
        -0.035,
        f"行列按风格潜空间均值的平均连接层次聚类排序；{len(p)} 个有序风格对共享源啤酒，彼此不独立",
        fontsize=7,
        color=pf.MUTED,
    )
    fig.subplots_adjust(left=0.24, right=0.98, top=0.98, bottom=0.27, wspace=0.05)
    print(pf.save(fig, "q3", "fig_q3_pairs.pdf"))


if __name__ == "__main__":
    main()
