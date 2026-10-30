import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.paths import PROCESSED_DIR  # noqa: E402

from src import paperfig as pf  # noqa: E402


def main():
    pf.apply()
    z = np.load(PROCESSED_DIR / "q3_fingerprint_dist.npz", allow_pickle=False)
    styles = [str(s) for s in z["styles"]]
    fig, axes = plt.subplots(len(styles), 1, figsize=(pf.TEXTWIDTH, 8.6))
    rng = np.random.default_rng(int(z["seed"]))
    for k, (ax, st) in enumerate(zip(axes, styles)):
        comp = [str(c) for c in z[f"s{k}_compound"]]
        syn, real = z[f"s{k}_syn"], z[f"s{k}_real"]
        y = np.arange(len(comp))
        ax.boxplot(
            [syn[:, i] for i in range(len(comp))],
            positions=y,
            vert=False,
            widths=0.55,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#dbe9f6", "edgecolor": pf.SKY, "lw": 0.7},
            medianprops={"color": pf.SKY, "lw": 1.0},
            whiskerprops={"color": pf.SKY, "lw": 0.7},
            capprops={"color": pf.SKY, "lw": 0.7},
        )
        for i in range(len(comp)):
            ax.scatter(
                real[:, i],
                i + rng.uniform(-0.18, 0.18, len(real)),
                s=6,
                color=pf.INK,
                alpha=0.55,
                lw=0,
                zorder=3,
            )
        d, lo, hi = z[f"s{k}_delta"], z[f"s{k}_lo"], z[f"s{k}_hi"]
        ax.errorbar(
            d, y, xerr=[d - lo, hi - d], fmt="D", color=pf.ORANGE, ms=4, lw=1.2, capsize=0, zorder=4
        )
        ax.axvline(0, color=pf.MUTED, lw=0.7)
        ax.set_yticks(y, comp, fontsize=6.8)
        ax.set_ylim(len(comp) - 0.5, -0.5)
        ax.set_xlim(-4, 4)
        ax.grid(axis="x", visible=True)
        ax.set_axisbelow(True)
        pf.panel_label(ax, f"{chr(97 + k)}  {st}，真实 {real.shape[0]} 款")
    axes[-1].set_xlabel("标准化潜空间取值，即相对全体均值的风格内标准差数")
    handles = [
        Patch(
            facecolor="#dbe9f6", edgecolor=pf.SKY, label=f"生成样本，{syn.shape[0]} 个，须为 5%–95%"
        ),
        Line2D([], [], marker="o", ls="", color=pf.INK, alpha=0.55, ms=3, label="该风格真实啤酒"),
        Line2D(
            [],
            [],
            marker="D",
            ls="-",
            color=pf.ORANGE,
            ms=4,
            label="化学指纹 $\\delta$ 与 90% bootstrap 区间",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.2)
    print(pf.save(fig, "q3", "fig_q3_fingerprint.pdf"))


if __name__ == "__main__":
    main()
