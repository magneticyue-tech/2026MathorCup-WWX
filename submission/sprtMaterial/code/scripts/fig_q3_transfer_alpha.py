import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, PALETTE, PROCESSED_DIR, apply, save  # noqa: E402


def main():
    apply()
    z = np.load(PROCESSED_DIR / "q3_transfer_pils_paths.npz", allow_pickle=False)
    tab = pd.read_csv(PROCESSED_DIR / "q3_transfer_pils_tripel.csv", encoding="utf-8-sig")
    al = z["alphas"]
    F = z["feasible"].astype(float)
    code = np.where(z["feasible"], 2, np.where(~z["ok_chem"], 0, 1))
    fig, ax = plt.subplots(figsize=(8, 4.2))
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
    ax.plot(z["alpha_star"], y, "o", color=PALETTE["main"], label="α*（最大可行网格点）")
    ax.plot(z["alpha_fit"], y, "x", color=PALETTE["better"], ms=8, label="α_fit（生成器后验最大）")
    lab = [
        f"{b}{'（双乙酰源缺失）' if m else ''}"
        for b, m in zip(tab["beer"], tab["src_missing_diacetyl"])
    ]
    ax.set_yticks(y, lab, fontsize=7)
    ax.set_xlabel("迁移强度 α")
    ax.set_ylabel("源啤酒 beer")
    ax.set_title("Pils/Lager → Tripel：路径可行性（浅蓝可行，橙色感官约束不满足）")
    ax.grid(False)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, -0.32), ncol=2)
    save(fig, FIG_DIR, "fig_q3_transfer_alpha.pdf")


if __name__ == "__main__":
    main()
