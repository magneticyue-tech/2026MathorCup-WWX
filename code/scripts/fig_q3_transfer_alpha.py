import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402


def main():
    apply()
    z = np.load(PROCESSED_DIR / "q3_transfer_pils_paths.npz", allow_pickle=False)
    tab = pd.read_csv(PROCESSED_DIR / "q3_transfer_pils_tripel.csv", encoding="utf-8-sig")
    al = z["alphas"]
    F = z["feasible"].astype(float)
    code = np.where(z["feasible"], 2, np.where(~z["ok_chem"], 0, 1))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    cmap = ListedColormap(["#444444", "#f2c0a0", "#cfe3f2"])
    ax.imshow(
        code,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=2,
        extent=[al[0] - 0.025, al[-1] + 0.025, len(F) - 0.5, -0.5],
    )
    y = np.arange(len(F))
    ax.plot(
        z["alpha_star"],
        y,
        "o",
        color=PALETTE["main"],
        label="最大可行迁移强度 α*（最大可行网格点）",
    )
    ax.plot(
        z["alpha_fit"],
        y,
        "x",
        color=PALETTE["better"],
        ms=8,
        label="生成器后验最大的可行迁移强度 α_fit",
    )
    lab = [
        f"{b}{'（双乙酰源缺失）' if m else ''}"
        for b, m in zip(tab["beer"], tab["src_missing_diacetyl"])
    ]
    ax.set_yticks(y, lab, fontsize=9)
    ax.set_xlabel("迁移强度 α")
    ax.set_ylabel("源啤酒编号（Pils/Lager）")
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    color_handles = [
        Patch(facecolor="#cfe3f2"),
        Patch(facecolor="#f2c0a0"),
        Patch(facecolor="#444444"),
    ]
    color_labels = ["可行", "感官约束不满足", "化学约束不满足"]
    handles = [
        color_handles[0],
        handles[0],
        color_handles[1],
        handles[1],
        color_handles[2],
    ]
    labels = [color_labels[0], labels[0], color_labels[1], labels[1], color_labels[2]]
    ax.legend(
        handles,
        labels,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
    )
    save(fig, FIG_DIR, "fig_q3_transfer_alpha.pdf")


if __name__ == "__main__":
    main()
