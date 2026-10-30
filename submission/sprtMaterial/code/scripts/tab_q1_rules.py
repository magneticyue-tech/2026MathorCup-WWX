import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402
from src.q1.tables import write_table  # noqa: E402

RULES = ("const", "round", "median", "quant", "qwk_cut", "median_knn")
FIELDS = (
    ("mae", "等级 MAE"),
    ("exact", "完全一致"),
    ("within1", "±1 以内"),
    ("qwk", "QWK"),
    ("balanced_acc", "平衡准确率"),
    ("share56", "5–6 级占比"),
)
SUBSETS = (("all", "全部 50 维"), ("zero_heavy", "零分为主 22 维"), ("other", "其余 28 维"))


def build(src):
    j = plotstyle.load_json(src / "q1_metrics.json")
    freq = j["grade_freq"]["truth"]
    rows = []
    for skey, sname in SUBSETS:
        for rule in RULES:
            m = j["models"][f"main.{rule}"]
            part = m if skey == "all" else m[skey]
            rec = {"subset": sname, "rule": rule, "规则": plotstyle.RULE_LABELS[rule]}
            rec.update({f: (np.nan if part.get(f) is None else part[f]) for f, _ in FIELDS})
            rows.append(rec)
        truth = {"subset": sname, "rule": "truth", "规则": "真值", **{f: np.nan for f, _ in FIELDS}}
        if skey == "all":
            truth["share56"] = freq[5] + freq[6]
        rows.append(truth)
    return pd.DataFrame(rows)


def main():
    args = plotstyle.script_args(FIG_DIR)
    df = build(args.src)
    tex = df.drop(columns="rule").rename(columns={"subset": "维度", **dict(FIELDS)})
    write_table(
        df,
        args.out,
        "tab_q1_rules",
        caption=(
            "主模型 6 个离散规则的比较（零分为主维度与其余维度的子表只含"
            "等级 MAE、完全一致与 QWK；--：该子表不适用）"
        ),
        label="tab:q1_rules",
        tex_df=tex,
        colfmt="ll" + "r" * len(FIELDS),
    )
    print(args.out / "tab_q1_rules.csv")


if __name__ == "__main__":
    main()
