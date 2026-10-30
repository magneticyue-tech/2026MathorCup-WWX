from pathlib import Path

import matplotlib as mpl

PALETTE = {
    "main": "#0072B2",
    "ctrl": "#999999",
    "worse": "#0072B2",
    "better": "#D55E00",
    "neutral": "#999999",
    "accent": "#009E73",
    "warn": "#E69F00",
}


def apply() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
        }
    )


def save(fig, out_dir, name) -> Path:
    if not name.endswith(".pdf"):
        raise ValueError(f"图名必须以 .pdf 结尾: {name}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    return path


def significance_color(lo, hi) -> str:
    if lo > 0:
        return PALETTE["worse"]
    if hi < 0:
        return PALETTE["better"]
    return PALETTE["neutral"]


def forest(ax, labels, point, lo, hi, colors) -> None:
    y = list(range(len(labels)))[::-1]
    for yi, p, a, b, c in zip(y, point, lo, hi, colors):
        ax.plot([a, b], [yi, yi], color=c, lw=1.4)
        ax.plot([p], [yi], "o", color=c, ms=4)
    ax.axvline(0, ls="--", color="#444444", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)


MODEL_LABELS = {
    "main": "主模型",
    "b0": "B0 训练折均值",
    "b1": "B1 风格收缩均值",
    "lin10": "线性回归（前 10 特征）",
    "enet": "ElasticNet",
    "rf": "随机森林",
    "chemstyle": "化学 + 风格",
    "k2": "K2 有序二分类",
    "c1": "C1 两阶段（只用附件）",
    "c2": "C2 两阶段校准",
}
RULE_LABELS = {
    "const": "常量（训练折众数）",
    "round": "四舍五入",
    "median": "预测分布中位数",
    "quant": "分位数切点",
    "qwk_cut": "QWK 切点",
    "median_knn": "近邻残差中位数",
    "sel": "选定映射",
}
ARM_LABELS = {
    "na_median_ind": "缺失：中位数 + 指示",
    "na_native_nan": "缺失：原生 NaN",
    "na_floor_ind": "缺失：最低位 + 指示",
    "drop_sum": "删除 4 个 _sum 列",
    "att8": "附件 8 校准值",
    "mono_on": "单调约束全开",
    "mono_off": "单调约束全关",
    "alpha": "目标：加性模型 α",
}
GROUP_MARKERS = {
    "Aroma": "o",
    "Flavour": "s",
    "Defect": "^",
    "Spice": "D",
    "Sour": "v",
    "Basic": "P",
}


def load_json(path):
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_npz(path):
    import numpy as np

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def script_args(default_out):
    import argparse

    from .paths import PROCESSED_DIR

    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=PROCESSED_DIR)
    p.add_argument("--out", type=Path, default=default_out)
    return p.parse_args()
