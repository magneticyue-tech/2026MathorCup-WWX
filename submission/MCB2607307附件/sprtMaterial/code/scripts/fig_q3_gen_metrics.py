import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, GEN_LABELS, PALETTE, apply, load_json, save  # noqa: E402


def bars(ax, m, key, gens, ref=None, ref_label=None):
    x = np.arange(len(gens))
    est = np.array([m[f"{g}.{key}"]["estimate"] for g in gens])
    lo = np.array([m[f"{g}.{key}"]["ci95"]["lo"] for g in gens])
    hi = np.array([m[f"{g}.{key}"]["ci95"]["hi"] for g in gens])
    ax.bar(x, est, color=PALETTE["main"], alpha=0.85)
    ax.errorbar(
        x,
        est,
        yerr=[np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)],
        fmt="none",
        ecolor="k",
        lw=1,
    )
    if ref is not None:
        ax.axhline(ref, ls="--", color=PALETTE["accent"], lw=1.2, label=ref_label)
        ax.legend(fontsize=7)
    ax.set_xticks(x, [GEN_LABELS[g] for g in gens], rotation=30, ha="right")


def main():
    apply()
    m = load_json("q3_genval.json")["metrics"]
    gens = list(GEN_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    bars(axes[0, 0], m, "dcr_ratio", gens, 1.0, "真实留出 = 1")
    axes[0, 0].set_ylabel("DCR 比（中位数比）")
    bars(axes[0, 1], m, "near_copy", gens, 0.05, "参照 0.05")
    axes[0, 1].set_ylabel("近复制率")
    bars(
        axes[1, 0],
        m,
        "tstr",
        gens,
        m["real.trtr"]["estimate"],
        f"TRTR {m['real.trtr']['estimate']:.3f}",
    )
    axes[1, 0].set_ylabel("TSTR 风格判别准确率")
    bars(axes[1, 1], m, "floor_err", gens)
    axes[1, 1].set_ylabel("下限比例误差（绝对差均值）")
    for ax, t in zip(axes.ravel(), ("新颖性：DCR 比", "近复制率", "TSTR 与 TRTR", "下限比例误差")):
        ax.set_title(t)
    fig.tight_layout()
    save(fig, FIG_DIR, "fig_q3_gen_metrics.pdf")


if __name__ == "__main__":
    main()
