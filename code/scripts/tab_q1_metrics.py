import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402
from src.q1.tables import write_table  # noqa: E402

ROWS = (
    ("main", "main.median"),
    ("b0", "b0.median"),
    ("b1", "b1.median"),
    ("lin10", "lin10.median"),
    ("enet", "enet.median"),
    ("rf", "rf.median"),
    ("chemstyle", "chemstyle.median"),
    ("k2", "k2.median"),
    ("c1", "c1.sel"),
    ("c2", "c2.median"),
)
COLS = (
    ("mae", "等级 MAE"),
    ("exact", "完全一致率"),
    ("within1", "±1 以内"),
    ("qwk", "QWK"),
    ("balanced_acc", "平衡准确率"),
    ("rps", "RPS"),
    ("logloss", "对数损失"),
    ("cont_mae", "连续 MAE"),
    ("cont_r2", "R²"),
    ("cont_spearman", "Spearman"),
)


def build(src):
    j = plotstyle.load_json(src / "q1_metrics.json")
    b2 = plotstyle.load_npz(src / "q1_baselines.npz")["b2_r2"]
    records = []
    for key, entry in ROWS:
        m = j["models"][entry]
        rec = {"key": entry, "模型": plotstyle.MODEL_LABELS[key]}
        rec.update({f: (np.nan if m.get(f) is None else m[f]) for f, _ in COLS})
        records.append(rec)
    b2_row = {"key": "b2", "模型": "B2（对 z，75 款测试）", **{f: np.nan for f, _ in COLS}}
    b2_row["cont_r2"] = float(np.mean(b2))
    clim = {"key": "climatology", "模型": "等级频率气候学基线", **{f: np.nan for f, _ in COLS}}
    clim["rps"], clim["logloss"] = j["climatology"]["rps"], j["climatology"]["logloss"]
    return pd.DataFrame([*records, b2_row, clim])


def main():
    args = plotstyle.script_args(FIG_DIR)
    df = build(args.src)
    tex = df.drop(columns="key").rename(columns=dict(COLS))
    write_table(
        df,
        args.out,
        "tab_q1_metrics",
        caption="各模型的等级与连续指标（规则为预测分布中位数，C1 为其选定映射；--：不适用）",
        label="tab:q1_metrics",
        tex_df=tex,
        colfmt="l" + "r" * (tex.shape[1] - 1),
    )
    print(args.out / "tab_q1_metrics.csv")


if __name__ == "__main__":
    main()
