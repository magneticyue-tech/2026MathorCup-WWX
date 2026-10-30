import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, PROCESSED_DIR, apply, save  # noqa: E402


def main():
    apply()
    df = pd.read_csv(PROCESSED_DIR / "q3_transfer_pairs.csv", encoding="utf-8-sig")
    styles = sorted(set(df["src"]) | set(df["tgt"]))
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax, col, label, cmap in (
        (
            axes[0],
            "alpha_star_median",
            "(a) 最大可行迁移强度 α* 的中位数（对角为同风格）",
            "viridis",
        ),
        (
            axes[1],
            "p_main_median",
            "(b) 迁移后判别器目标风格概率的中位数（对角为同风格）",
            "magma",
        ),
    ):
        M = np.full((len(styles), len(styles)), np.nan)
        for _, r in df.iterrows():
            M[styles.index(r["src"]), styles.index(r["tgt"])] = r[col]
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#dddddd")
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=1)
        for i in range(len(styles)):
            ax.text(i, i, "—", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(styles)), styles, rotation=90, fontsize=9)
        ax.set_yticks(range(len(styles)), styles, fontsize=9)
        ax.set_xlabel("目标风格")
        ax.set_ylabel("源风格")
        ax.set_title(label, fontsize=11, loc="left")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.75)
        cb.ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, FIG_DIR, "fig_q3_pairs.pdf")


if __name__ == "__main__":
    main()
