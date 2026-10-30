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

    idx = [styles.index(st) for st in NAMED_STYLES]
    top = max(z["mu_q90"][idx].max(), max(oof[style == st].max() for st in NAMED_STYLES))
    rmax = float(np.ceil(top))
    fig, axes = plt.subplots(
        1, len(NAMED_STYLES), figsize=(15, 5.8), subplot_kw={"projection": "polar"}
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
        ax.set_xticklabels([dims[j] for j in order], fontsize=4.5)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(f"{st}（真实 {int(z['n_style'][si])} 款）", fontsize=9, pad=16)
    handles = [
        plt.Line2D([], [], color=PALETTE["main"], lw=1.6, label="生成画像中位数"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["main"], alpha=0.2, label="生成画像 10%–90% 带"),
        plt.Line2D(
            [], [], color=PALETTE["better"], lw=0.6, alpha=0.6, label="同风格真实啤酒折外预测"
        ),
        plt.Line2D([], [], color=PALETTE["better"], lw=1.2, ls="--", label="真实折外预测中位数"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    n = z["mu"].shape[1]
    fig.suptitle(
        f"{'、'.join(NAMED_STYLES)} 的感官画像"
        f"（每风格 {n} 个生成谱经问题一全数据模型预测的 {len(dims)} 维连续评分）",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    print(save(fig, FIG_DIR, "fig_q3_radar.pdf"))


if __name__ == "__main__":
    main()
