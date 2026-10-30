import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402
from src.q1.report_beers import REPORT_ROWS  # noqa: E402

N_PANELS = 6


def pick_rows(style, ext_rows):
    rows = [int(r) for r in ext_rows]
    used = {style[r - 1] for r in rows}
    for r in REPORT_ROWS:
        if len(rows) == N_PANELS:
            break
        if r not in rows and style[r - 1] not in used:
            rows.append(r)
            used.add(style[r - 1])
    return rows


def closed(v):
    return np.append(v, v[0])


def main():
    args = plotstyle.script_args(FIG_DIR)
    plotstyle.apply()
    d = plotstyle.load_npz(args.src / "q1_data.npz")
    s = plotstyle.load_npz(args.src / "q1_delivery_scores.npz")
    e = plotstyle.load_npz(args.src / "q1_external.npz")
    p = plotstyle.PALETTE
    dims = [str(x) for x in d["dims"]]
    style = [str(x) for x in d["style"]]
    order = [dims.index(c) for members in DIM_GROUPS.values() for c in members]
    ext_dims = [str(x) for x in e["dims"]]
    ext_rows = [int(r) + 1 for r in e["beer_rows"]]
    rows = pick_rows(style, ext_rows)

    ang = np.linspace(0, 2 * np.pi, len(order), endpoint=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 13), subplot_kw={"projection": "polar"})
    for ax, r in zip(axes.ravel(), rows):
        i = r - 1
        ax.fill_between(
            closed(ang),
            closed(s["lo"][i, order]),
            closed(s["hi"][i, order]),
            color=p["main"],
            alpha=0.15,
            label="预测 90% 区间",
        )
        ax.plot(
            closed(ang), closed(s["G_hat"][i, order]), color=p["main"], lw=1.4, label="预测等级"
        )
        ax.plot(
            closed(ang),
            closed(d["G"][i, order]),
            color=p["better"],
            lw=1.1,
            ls="--",
            label="实测等级",
        )
        if r in ext_rows:
            k = ext_rows.index(r)
            a = [ang[order.index(dims.index(c))] for c in ext_dims]
            ax.scatter(
                a,
                e["ctrl_grade"][k],
                s=22,
                marker="D",
                color=p["accent"],
                zorder=5,
                label="外部加标实验对照杯等级",
            )
        tag = ""
        if r in ext_rows:
            tag = f"（外部检查底酒 {int(e['beer_codes'][ext_rows.index(r)])}）"
        ax.text(
            0.5,
            -0.26,
            f"第 {r} 行 · {style[i]}{tag}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            clip_on=False,
        )
        ax.set_ylim(0, 6)
        ax.set_yticks(range(0, 7, 2))
        ax.set_xticks(ang)
        ax.set_xticklabels([])
        for j, theta in zip(order, ang):
            deg = np.degrees(theta)
            if deg <= 90:
                rot, ha = deg, "left"
            elif deg <= 270:
                rot, ha = deg - 180, "right"
            else:
                rot, ha = deg - 360, "left"
            ax.text(
                theta,
                6.55,
                dims[j],
                rotation=rot,
                rotation_mode="anchor",
                ha=ha,
                va="center",
                fontsize=8,
                clip_on=False,
            )
        ax.set_rlabel_position(22.5)
        ax.tick_params(axis="y", labelsize=9)
    handles, labels = [], []
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.96, top=0.92, bottom=0.13, wspace=0.7, hspace=0.5)
    print(plotstyle.save(fig, args.out, "fig_q1_radar.pdf"))


if __name__ == "__main__":
    main()
