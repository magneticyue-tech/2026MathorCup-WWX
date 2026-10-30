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
from src.q3.paths import PROCESSED_DIR  # noqa: E402

from src import paperfig as pf  # noqa: E402

METHODS = (
    ("p_src", "原谱不变", pf.LIGHT),
    ("p_t0", "平移", pf.YELLOW),
    ("p_t1", "截断", "#f0c27b"),
    ("p_t3", "原尺度平移", pf.PINK),
    ("p_t5", "截断加感官搜索", pf.SKY),
    ("p_main", "主方法", pf.BLUE),
)
STATE_COLORS = ("#e8f1e4", "#7fbf7b", pf.ORANGE)


def main():
    pf.apply()
    t = pd.read_csv(PROCESSED_DIR / "q3_transfer_pils_tripel.csv")
    d = pd.read_csv(PROCESSED_DIR / "q3_transfer_defects.csv")
    tr = np.load(PROCESSED_DIR / "q3_transfer.npz", allow_pickle=False)
    ref = float(tr["ref_q25"][[str(s) for s in tr["styles"]].index("Tripel")])
    beers = list(t["beer"])

    fig, axes = plt.subplots(
        1, 2, figsize=(pf.TEXTWIDTH, 3.9), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    ax = axes[0]
    y = np.arange(len(METHODS))
    rng = np.random.default_rng(0)
    for yi, (col, lab, c) in zip(y, METHODS):
        v = t[col].to_numpy(float)
        ax.scatter(
            v, yi + rng.uniform(-0.15, 0.15, len(v)), s=12, color=c, ec="white", lw=0.3, zorder=3
        )
        ax.plot([np.median(v)] * 2, [yi - 0.3, yi + 0.3], color=pf.INK, lw=1.4, zorder=4)
        ax.text(0.53, yi, f"{np.median(v):.3f}", va="center", fontsize=6.8, color=pf.INK)
    ax.axvline(ref, color=pf.ORANGE, lw=0.8, ls=(0, (3, 2)))
    ax.text(
        ref,
        len(METHODS) - 0.4,
        f"真实 Tripel 25% 分位 {ref:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=pf.ORANGE,
    )
    ax.set_yticks(y, [lab for _, lab, _ in METHODS])
    ax.set_ylim(len(METHODS) - 0.3, -0.7)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("$P$(Tripel)，竖线为中位数")
    ax.grid(axis="x", visible=True)
    ax.set_axisbelow(True)
    pf.panel_label(ax, f"a  {len(beers)} 款 Pils/Lager 的 Tripel 概率")

    ax = axes[1]
    items = list(dict.fromkeys(d.loc[d["layer"] == "chem", "item"])) + list(
        dict.fromkeys(d.loc[d["layer"] == "sens", "item"])
    )
    d = d.set_index(["beer", "item"])
    state = np.zeros((len(beers), len(items)), int)
    for i, b in enumerate(beers):
        for j, it in enumerate(items):
            r = d.loc[(b, it)]
            if not bool(r["recheck_ok"]):
                state[i, j] = 2
            elif abs(float(r["after"]) - float(r["bound"])) <= 1e-9:
                state[i, j] = 1
    ax.imshow(
        state,
        cmap=ListedColormap(STATE_COLORS),
        vmin=0,
        vmax=2,
        aspect="auto",
        interpolation="none",
    )
    n_chem = int((d.reset_index()["layer"] == "chem").sum() / len(beers))
    ax.axvline(n_chem - 0.5, color="white", lw=2.5)
    ax.set_xticks(range(len(items)), items, rotation=60, ha="right", fontsize=6.3)
    ax.set_yticks(range(len(beers)), [str(b) for b in beers], fontsize=6.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text((n_chem - 1) / 2, -0.9, "化学层", ha="center", fontsize=7.5)
    ax.text(n_chem + (len(items) - n_chem - 1) / 2, -0.9, "感官层", ha="center", fontsize=7.5)
    n_bad = int((state == 2).sum())
    pf.panel_label(ax, f"b  主方法约束复查：违约 {n_bad} / {state.size} 格", y=1.08)
    ax.legend(
        handles=[
            Patch(color=STATE_COLORS[0], label="低于上界"),
            Patch(color=STATE_COLORS[1], label="等于上界，约束起作用"),
            Patch(color=STATE_COLORS[2], label="超出上界"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=3,
        handlelength=1.0,
    )
    ax.set_ylabel("源啤酒编号")
    n_t2 = int(t["p_t2"].isna().sum())
    if n_t2:
        fig.text(
            0.02,
            -0.02,
            f"检索真实 Tripel 对照在 {n_t2} / {len(beers)} 款上无满足两层约束的候选，未画入 a 栏",
            fontsize=7,
            color=pf.MUTED,
        )
    fig.tight_layout(w_pad=1.5)
    print(pf.save(fig, "q3", "fig_q3_transfer.pdf"))


if __name__ == "__main__":
    main()
