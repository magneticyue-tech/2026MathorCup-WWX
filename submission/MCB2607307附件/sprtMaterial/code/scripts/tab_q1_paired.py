import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402
from src.q1.tables import write_table  # noqa: E402

STATS = (("mae", "等级 MAE"), ("exact", "完全一致"), ("qwk", "QWK"))


def fmt(p, k):
    pt, lo, hi = p["point"][k], p["lo"][k], p["hi"][k]
    if pt is None:
        return ""
    return f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def build(src):
    paired = plotstyle.load_json(src / "q1_metrics.json")["paired"]
    rows = []
    for p in paired:
        rec = {"a": p["a"], "b": p["b"]}
        for k, _ in STATS:
            rec[k] = fmt(p, k)
        rec["wins_a"] = p["wins_a"]
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    args = plotstyle.script_args(FIG_DIR)
    df = build(args.src)
    tex = df.rename(columns={**dict(STATS), "wins_a": "a 更优维度数"})
    write_table(
        df,
        args.out,
        "tab_q1_paired",
        caption="配对比较（a 减 b，点估计与按日期组 bootstrap 的 95\\% CI）",
        label="tab:q1_paired",
        tex_df=tex,
        colfmt="llrrrr",
    )
    print(args.out / "tab_q1_paired.csv")


if __name__ == "__main__":
    main()
