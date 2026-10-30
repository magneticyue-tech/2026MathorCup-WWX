import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q3.config import NAMED_STYLES  # noqa: E402
from src.q3.paths import Q1_DIR  # noqa: E402
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402


def closed(v):
    return np.append(v, v[..., :1], axis=-1)


def main():
    apply()
    z = np.load(PROCESSED_DIR / "q3_portraits.npz", allow_pickle=False)
    d = np.load(Q1_DIR / "q1_data.npz", allow_pickle=False)
    oof = np.load(Q1_DIR / "q1_main_oof.npz", allow_pickle=False)["oof_mu"].mean(axis=0)
    styles = [str(s) for s in z["styles"]]
    dims = [str(x) for x in z["dims"]]
    style = np.array([str(s) for s in d["style"]])
    order = [dims.index(c) for members in DIM_GROUPS.values() for c in members]
    ang = closed(np.linspace(0, 2 * np.pi, len(order), endpoint=False))
    theta = np.degrees(ang[:-1])

    idx = [styles.index(st) for st in NAMED_STYLES]
    top = max(z["mu_q90"][idx].max(), max(oof[style == st].max() for st in NAMED_STYLES))
    rmax = float(np.ceil(top))
    fig, axes = plt.subplots(
        1, len(NAMED_STYLES), figsize=(21, 8), subplot_kw={"projection": "polar"}
    )
    for ax, st in zip(axes, NAMED_STYLES):
        si = styles.index(st)
        real = oof[style == st][:, order]
        for r in real:
            ax.plot(ang, closed(r), color=PALETTE["better"], lw=0.5, alpha=0.35)
        ax.fill_between(
            ang,
            closed(z["mu_q10"][si, order]),
            closed(z["mu_q90"][si, order]),
            color=PALETTE["main"],
            alpha=0.2,
        )
        ax.plot(ang, closed(z["mu_q50"][si, order]), color=PALETTE["main"], lw=1.6)
        ax.plot(ang, closed(np.median(real, 0)), color=PALETTE["better"], lw=1.2, ls="--")
        ax.set_ylim(0, rmax)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels([])
        for i, (deg, j) in enumerate(zip(theta, order)):
            flipped = 90 < deg < 270
            ax.text(
                np.radians(deg),
                rmax * (1.04 + 0.12 * (i % 2)),
                dims[j],
                rotation=deg + 180 if flipped else deg,
                rotation_mode="anchor",
                ha="right" if flipped else "left",
                va="center",
                fontsize=9,
                clip_on=False,
            )
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", pad=6)
    handles = [
        plt.Line2D([], [], color=PALETTE["main"], lw=1.6, label="生成画像中位数"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["main"], alpha=0.2, label="生成画像 10%–90% 带"),
        plt.Line2D(
            [],
            [],
            color=PALETTE["better"],
            lw=0.6,
            alpha=0.6,
            label="同风格真实啤酒浅层梯度提升树折外预测",
        ),
        plt.Line2D(
            [], [], color=PALETTE["better"], lw=1.2, ls="--", label="浅层梯度提升树折外预测中位数"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0),
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.13, wspace=0.85)
    for k, (ax, st) in enumerate(zip(axes, NAMED_STYLES)):
        si = styles.index(st)
        bb = ax.get_position()
        fig.text(
            bb.x0,
            0.965,
            f"({chr(97 + k)}) {st}（真实 {int(z['n_style'][si])} 款啤酒）",
            fontsize=11,
            ha="left",
            va="top",
        )
    print(save(fig, FIG_DIR, "fig_q3_radar.pdf"))


if __name__ == "__main__":
    main()
