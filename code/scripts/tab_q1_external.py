import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402
from src.q1.tables import write_table  # noqa: E402

COLS = ("beer", "dim", "ctrl_mean", "ctrl_se", "ctrl_grade", "G_hat", "G_obs")
SOURCE = {"main": "化学主模型的折外交付评分", "chemstyle": "化学 + 风格回退模型的折外交付评分"}


def build(src):
    j = plotstyle.load_json(src / "q1_external.json")
    df = pd.DataFrame(j["rows"])[list(COLS)]
    summ = []
    for key, s in j["summary"].items():
        label = "合计" if key == "all" else f"beer {key}"
        for who, name in (("hat", "G_hat"), ("obs", "G_obs")):
            summ.append(
                {
                    "beer": label,
                    "dim": f"{name} 对 ctrl_grade",
                    "ctrl_mean": s[f"{who}_mae"],
                    "ctrl_se": float(s[f"{who}_exact"]),
                    "ctrl_grade": np.nan,
                    "G_hat": s[f"{who}_bias"],
                    "G_obs": np.nan,
                }
            )
    return pd.concat([df, pd.DataFrame(summ)], ignore_index=True), j


def main():
    args = plotstyle.script_args(FIG_DIR)
    df, j = build(args.src)
    n = {str(v["code"]): j["ctrl_n"][k] for k, v in j["match"].items()}
    caption = (
        f"外部检查：两款底酒的对照杯评分与交付评分（交付来源：{SOURCE[j['model_key']]}，"
        f"model\\_key={j['model_key']}）。仅两款啤酒；"
        + "；".join(f"啤酒 {code} 有 {k} 条对照杯评分" for code, k in n.items())
        + "，结果只报告，不用于选择。表尾汇总行中三列依次为等级 MAE、完全一致数、平均有符号误差。"
    )
    write_table(
        df,
        args.out,
        "tab_q1_external",
        caption=caption,
        label="tab:q1_external",
        colfmt="llrrrrr",
    )
    print(args.out / "tab_q1_external.csv")


if __name__ == "__main__":
    main()
