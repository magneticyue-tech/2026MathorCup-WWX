import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import GRID_K, GRID_KAPPA  # noqa: E402
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402

KIND_LABELS = {"famod": "因子加模块块", "fa": "纯因子", "lw": "Ledoit–Wolf"}


def main():
    apply()
    z = np.load(PROCESSED_DIR / "q3_select.npz", allow_pickle=False)
    cfgs = [tuple(json.loads(c)) for c in z["configs"]]
    L = z["final_inner_L"]
    final = json.loads(str(z["final_config"]))
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.7]}, layout="constrained"
    )
    vmin = min(L[i] for i, c in enumerate(cfgs) if c[0] in ("fa", "famod"))
    vmax = max(L[i] for i, c in enumerate(cfgs) if c[0] in ("fa", "famod"))
    for ax, kind, panel in (
        (axes[0], "famod", "(a) 因子加模块块"),
        (axes[1], "fa", "(b) 纯因子"),
    ):
        M = np.full((len(GRID_K), len(GRID_KAPPA)), np.nan)
        for i, c in enumerate(cfgs):
            if c[0] == kind:
                M[GRID_K.index(c[1]), GRID_KAPPA.index(c[2])] = L[i]
        im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f"{M[a, b]:.1f}", ha="center", va="center", fontsize=9, color="w")
        if kind == final["kind"]:
            ax.plot(
                GRID_KAPPA.index(final["kappa"]),
                GRID_K.index(final["k"]),
                "s",
                mfc="none",
                mec=PALETTE["warn"],
                ms=16,
                mew=2,
            )
        ax.set_xticks(range(len(GRID_KAPPA)), [str(k) for k in GRID_KAPPA])
        ax.set_yticks(range(len(GRID_K)), [str(k) for k in GRID_K])
        ax.tick_params(labelsize=9)
        ax.set_xlabel("收缩强度 κ")
        ax.set_ylabel("因子数 k")
        ax.set_title(panel, loc="left", fontsize=11)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.9, label="潜空间高斯评分 L（nat/啤酒）")
    cbar.ax.tick_params(labelsize=9)
    lab = [
        f"{KIND_LABELS[k]}\nk={a}, κ={b:g}"
        for k, a, b in zip(z["chosen_kind"], z["chosen_k"], z["chosen_kappa"])
    ]
    u, cnt = np.unique(lab, return_counts=True)
    axes[2].barh(range(len(u)), cnt, color=PALETTE["main"])
    axes[2].set_yticks(range(len(u)), u)
    axes[2].tick_params(labelsize=9)
    axes[2].set_xlabel("外层折数（共 25）")
    axes[2].set_title("(c) 外层各折选定设定", loc="left", fontsize=11)
    save(fig, FIG_DIR, "fig_q3_select_heatmap.pdf")


if __name__ == "__main__":
    main()
