"""论文正文配图的公共版式：与 draft_main 的正文宽度、字号、配色和术语保持一致。"""

from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "figures"
TEXTWIDTH = 6.3

COLORS = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
BLUE, ORANGE, GREEN, YELLOW, PINK, SKY = COLORS
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
LIGHT = "#bdbdbd"

GROUP_LABELS = {
    "Aroma": "香气",
    "Flavour": "风味",
    "Defect": "缺陷",
    "Spice": "香料啤酒",
    "Sour": "酸啤酒",
    "Basic": "基本味口感与总体",
}


def apply() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "figure.dpi": 150,
        }
    )


def panel_label(ax, text, x=0.0, y=1.02) -> None:
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom"
    )


def save(fig, question, name) -> Path:
    if not name.endswith(".pdf"):
        raise ValueError(f"图名必须以 .pdf 结尾: {name}")
    out = FIG_ROOT / question
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path)
    return path
