import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.data import DIM_GROUPS  # noqa: E402
from src.q1.tables import write_table  # noqa: E402
from src.q2 import config as C  # noqa: E402

SCORE_LABELS = {"spearman": "ρ", "partial_spearman": "偏 ρ"}


def main():
    final = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    m = final[final["method"] == "main"].copy()
    group_of = {d: g for g, members in DIM_GROUPS.items() for d in members}
    order = [d for members in DIM_GROUPS.values() for d in members if d in set(m["dimension"])]
    m["_o"] = m["dimension"].map({d: i for i, d in enumerate(order)})
    m = m.sort_values(["_o", "rank"])
    df = pd.DataFrame(
        {
            "组": m["dimension"].map(group_of),
            "维度": m["dimension"],
            "位置": m["rank"],
            "化合物": m["compound"],
            "得分类型": m["score_type"].map(SCORE_LABELS),
            "得分": m["score"],
            "95% CI 下限": m["r_ci_lower"],
            "95% CI 上限": m["r_ci_upper"],
            "置换门槛 q_d": m["q_d"],
            "显著": m["is_significant"].map({True: "是", False: "否"}),
            "π^c": m["pi_c"],
            "π^m": m["pi_m"],
            "稳定性": m["stability"],
            "维度等级": m["grade"],
            "共变标记": m["covariation"].map({True: "是", False: "否"}),
        }
    )
    tex = pd.DataFrame(
        {
            "维度": df["维度"].where(df["位置"] == 1, ""),
            "位": df["位置"],
            "化合物": df["化合物"],
            "得分": [f"{s:+.3f}" for s in df["得分"]],
            "95% CI": [
                f"[{a:+.2f}, {b:+.2f}]" for a, b in zip(df["95% CI 下限"], df["95% CI 上限"])
            ],
            "显著": df["显著"],
            "入选频率": [f"{v:.2f}" for v in df["π^c"]],
            "稳定性": df["稳定性"],
            "等级": df["维度等级"].where(df["位置"] == 1, ""),
        }
    )
    write_table(
        df,
        C.FIGURES,
        "tab_q2_top3",
        caption=(
            f"全部 {m['dimension'].nunique()} 维 Top-3 完整结果"
            "（第 1 位为 Spearman，第 2、3 位为偏 Spearman；"
            f"区间与 $\\pi^c$ 来自 {int(m['n_bootstrap'].iloc[0])} 次日期组 bootstrap）"
        ),
        label="tab:q2_top3",
        tex_df=tex,
        colfmt="llllcccll",
        longtable=True,
    )
    print(C.FIGURES / "tab_q2_top3.csv", len(df))


if __name__ == "__main__":
    main()
