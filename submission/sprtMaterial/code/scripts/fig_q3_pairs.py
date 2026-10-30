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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax, col, title, cmap in (
        (axes[0], "alpha_star_median", "α* 中位数", "viridis"),
        (axes[1], "p_main_median", "迁移后 P_t 中位数", "magma"),
    ):
        M = np.full((len(styles), len(styles)), np.nan)
        for _, r in df.iterrows():
            M[styles.index(r["src"]), styles.index(r["tgt"])] = r[col]
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#dddddd")
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=1)
        for i in range(len(styles)):
            ax.text(i, i, "—", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(styles)), styles, rotation=90, fontsize=6)
        ax.set_yticks(range(len(styles)), styles, fontsize=6)
        ax.set_xlabel("目标风格")
        ax.set_ylabel("源风格")
        ax.set_title(f"{title}（对角为同风格，不适用）")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    save(fig, FIG_DIR, "fig_q3_pairs.pdf")


if __name__ == "__main__":
    main()
