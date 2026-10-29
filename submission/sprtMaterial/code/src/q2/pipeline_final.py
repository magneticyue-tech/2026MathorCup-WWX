# code/src/q2/pipeline_final.py
"""Q2 Top-1 热力图：49 维 × 入选 Top-1 化合物的边际 Spearman，标出 A/B/C 等级。"""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
import core  # noqa: E402
from data import load_q2_dataset  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_heatmap():
    fin = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    top1 = fin[(fin["method"] == "main") & (fin["rank"] == 1)]
    ds = load_q2_dataset()
    a = core.prepare_arm(ds, core.ArmSpec("main"))
    _, Cxy = core.corr_mats(a.Zx, a.okx, a.Zy, a.oky)
    comps = list(dict.fromkeys(top1.sort_values("compound")["compound"]))
    M = np.array([[Cxy[a.cand.index(c), a.dims.index(d)] for c in comps] for d in a.dims])
    fig, ax = plt.subplots(figsize=(0.32 * len(comps) + 4, 12))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    g = top1.set_index("dimension")
    for i, d in enumerate(a.dims):
        j = comps.index(g.at[d, "compound"])
        ax.text(j, i, g.at[d, "grade"], ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(comps)), comps, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(a.dims)), a.dims, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.5, label="边际 Spearman r")
    ax.set_title("各维度 Top-1 化合物（字母为维度等级）与这些化合物的边际 Spearman")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "top1_heatmap.png", dpi=300)
    plt.close(fig)
    print("图已更新：top1_heatmap")


if __name__ == "__main__":
    plot_heatmap()
