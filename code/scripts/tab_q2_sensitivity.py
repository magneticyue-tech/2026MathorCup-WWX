import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.tables import write_table  # noqa: E402
from src.q2 import config as C  # noqa: E402


def main():
    s = pd.read_csv(C.SENS_DIR / "sensitivity_summary.csv")
    df = pd.DataFrame(
        {
            "方案": s["arm"],
            "设置": s["description"],
            "化合物层 Jaccard": s["mean_jaccard_compound"],
            "模块层 Jaccard": s["mean_jaccard_module"],
            "Top-1 改变维度数": s["n_top1_changed"],
            "等级改变维度数": s["n_grade_changed"],
            "等级改变明细": s["grade_changed_dims"].fillna(""),
            "π^c 第 1 位": s["pi_c_pos1"],
            "π^c 第 2 位": s["pi_c_pos2"],
            "π^c 第 3 位": s["pi_c_pos3"],
            "A/B/C 维度数": [
                "" if pd.isna(a) else f"{int(a)}/{int(b)}/{int(c)}"
                for a, b, c in zip(s["n_A"], s["n_B"], s["n_C"])
            ],
            "备注": s["not_applicable"].fillna(""),
        }
    )
    tex = df[
        [
            "设置",
            "化合物层 Jaccard",
            "模块层 Jaccard",
            "Top-1 改变维度数",
            "等级改变维度数",
            "π^c 第 1 位",
            "A/B/C 维度数",
        ]
    ].copy()
    tex = tex.rename(columns={"π^c 第 1 位": "第 1 位入选频率"})
    for c in ("Top-1 改变维度数", "等级改变维度数"):
        tex[c] = pd.Series([None if pd.isna(v) else int(v) for v in tex[c]], dtype=object)
    n_runs = int((s["arm"] != "baseline").sum())
    write_table(
        df,
        C.FIGURES,
        "tab_q2_sensitivity",
        caption=(
            f"敏感性运行结果汇总（{n_runs} 组设置相对正式基线；逐对删除只作边际相关诊断，"
            "选择类指标不适用）"
        ),
        label="tab:q2_sensitivity",
        tex_df=tex,
        colfmt="p{6.2cm}cccccc",
    )
    print(C.FIGURES / "tab_q2_sensitivity.csv", len(df))


if __name__ == "__main__":
    main()
