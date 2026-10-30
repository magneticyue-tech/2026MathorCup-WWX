import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_q2_jaccard import METHODS, random_jaccard  # noqa: E402
from src.q1.tables import write_table  # noqa: E402
from src.q2 import config as C  # noqa: E402


def main():
    full = pd.read_csv(C.Q2_DIR / "full_controls_results.csv")
    n_cand = int(pd.read_csv(C.Q2_DIR / "graph_nodes.csv")["is_candidate"].sum())
    rows = []
    for k, name in METHODS:
        rows.append(
            {
                "对照": name.replace("\n", ""),
                "化合物层": full[f"jaccard_{k}"].astype(float).mean(),
                "模块层": full[f"jaccard_mod_{k}"].astype(float).mean(),
                "Top-1 相同比例": full[f"{k}_top1_match"].astype(float).mean(),
                "Top-1 相同维度数": int(full[f"{k}_top1_match"].astype(bool).sum()),
            }
        )
    rows.append(
        {
            "对照": "随机基线",
            "化合物层": random_jaccard(n_cand),
            "模块层": np.nan,
            "Top-1 相同比例": np.nan,
            "Top-1 相同维度数": np.nan,
        }
    )
    df = pd.DataFrame(rows)
    write_table(
        df,
        C.FIGURES,
        "tab_q2_jaccard",
        caption=(
            f"主方法与对照方法的 Jaccard 系数（{len(full)} 维平均；"
            f"随机基线为 {n_cand} 个候选中两个随机 3 元集合的期望）"
        ),
        label="tab:q2_jaccard",
        tex_df=df.drop(columns=["Top-1 相同维度数"]),
        colfmt="lccc",
    )
    print(C.FIGURES / "tab_q2_jaccard.csv")


if __name__ == "__main__":
    main()
