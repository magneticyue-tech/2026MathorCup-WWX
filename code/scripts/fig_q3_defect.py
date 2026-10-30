import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.plotting import FIG_DIR, PROCESSED_DIR, apply, save  # noqa: E402


def grid(d, beers, items, col):
    return d.pivot(index="beer", columns="item", values=col).loc[beers, items]


def main():
    apply()
    d = pd.read_csv(PROCESSED_DIR / "q3_transfer_defects.csv")
    tab = pd.read_csv(PROCESSED_DIR / "q3_transfer_pils_tripel.csv")
    beers = list(tab["beer"])
    chem = d[d["layer"] == "chem"]
    sens = d[d["layer"] == "sens"]
    ci = list(dict.fromkeys(chem["item"]))
    si = list(dict.fromkeys(sens["item"]))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log10(grid(chem, beers, ci, "oav_after") / grid(chem, beers, ci, "oav_before"))
    raw = grid(chem, beers, ci, "after") - grid(chem, beers, ci, "before")
    ok_c = grid(chem, beers, ci, "recheck_ok").astype(bool)
    act = grid(chem, beers, ci, "active").astype(bool)
    oav_a = grid(chem, beers, ci, "oav_after")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [9, 6]})
    ax = axes[0]
    lim = float(np.nanmax(np.abs(ratio.to_numpy()))) or 1.0
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    shown = np.ma.masked_invalid(ratio.to_numpy(float))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(shown, cmap=cmap, norm=norm, aspect="auto")
    for r in range(len(beers)):
        for c in range(len(ci)):
            mark = "✓" if ok_c.iat[r, c] else "✗"
            if np.isnan(ratio.iat[r, c]):
                txt = f"{mark}\nΔ{raw.iat[r, c]:+.2f}"
            else:
                txt = f"{mark}\nOAV {oav_a.iat[r, c]:.2f}"
            ax.text(c, r, txt, ha="center", va="center", fontsize=5.5, fontfamily="DejaVu Sans")
            if act.iat[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="black", lw=1.4))
    ax.set_xticks(range(len(ci)), ci, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(beers)), [str(b) for b in beers], fontsize=7)
    ax.set_ylabel("源啤酒编号（Pils/Lager）")
    ax.grid(False)
    ax.set_title(
        f"化学层 {len(ci)} 个缺陷项：色块为 $\\log_{{10}}$(迁移后 OAV / 迁移前 OAV)；"
        "灰色为无阈值项（格内为原尺度变化量）；粗框为约束起作用",
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, shrink=0.75, label=r"$\log_{10}$ OAV 变化（红色升高）")

    ax = axes[1]
    gb = grid(sens, beers, si, "before").astype(int)
    ga = grid(sens, beers, si, "after").astype(int)
    ok_s = grid(sens, beers, si, "recheck_ok").astype(bool)
    diff = (ga - gb).to_numpy(float)
    im2 = ax.imshow(diff, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for r in range(len(beers)):
        for c in range(len(si)):
            ax.text(
                c,
                r,
                f"{'✓' if ok_s.iat[r, c] else '✗'}\n{gb.iat[r, c]}→{ga.iat[r, c]}",
                ha="center",
                va="center",
                fontsize=6,
                fontfamily="DejaVu Sans",
            )
    ax.set_xticks(range(len(si)), si, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(beers)), [str(b) for b in beers], fontsize=7)
    ax.grid(False)
    ax.set_title(f"感官层 {len(si)} 个缺陷维度：问题一预测等级（迁移前→后）", fontsize=8)
    fig.colorbar(im2, ax=ax, shrink=0.75, label="等级变化", ticks=[-2, -1, 0, 1, 2])
    n_ok = int(ok_c.to_numpy().sum() + ok_s.to_numpy().sum())
    n_all = ok_c.size + ok_s.size
    fig.suptitle(
        f"缺陷不恶化核查表（{len(beers)} 款 Pils/Lager → Tripel，"
        f"主方法 α* 终点原尺度复查：{n_ok}/{n_all} 格满足约束）",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    print(save(fig, FIG_DIR, "fig_q3_defect.pdf"))


if __name__ == "__main__":
    main()
