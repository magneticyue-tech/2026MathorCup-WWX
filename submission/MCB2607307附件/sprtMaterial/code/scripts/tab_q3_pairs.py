import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.paths import FIG_DIR, PROCESSED_DIR  # noqa: E402
from src.q3.tables import to_tex  # noqa: E402

FRACS = ["frac_ref", "frac_improved", "frac_unchanged", "frac_no_improve"]


def main():
    p = pd.read_csv(PROCESSED_DIR / "q3_transfer_pairs.csv", encoding="utf-8-sig")
    assert len(p) == 462 and not p.duplicated(["src", "tgt"]).any() and (p["src"] != p["tgt"]).all()
    rows = []
    for s, g in sorted(p.groupby("src"), key=lambda x: x[0]):
        assert g["tgt"].nunique() == 21 and g["n_src"].nunique() == 1
        rows.append(
            {
                "src": s,
                "n_src": int(g["n_src"].iloc[0]),
                "alpha_star_mean_median": g["alpha_star_mean"].median(),
                "p_main_median_across_targets": g["p_main_median"].median(),
                "frac_t0_chem_viol_mean": g["frac_t0_chem_viol"].mean(),
                "frac_alpha0_mean": g["frac_alpha0"].mean(),
                **{f"{c}_mean": g[c].mean() for c in FRACS},
            }
        )
    df = pd.DataFrame(rows)
    assert len(df) == 22
    s = df[[f"{c}_mean" for c in FRACS]].sum(1)
    assert np.abs(s - 1).max() <= 1e-9
    df.to_csv(FIG_DIR / "tab_q3_pairs.csv", index=False, encoding="utf-8-sig")
    t1 = df.iloc[:, :6].rename(
        columns={
            "src": "源风格",
            "n_src": "源啤酒数",
            "alpha_star_mean_median": "α* 均值（跨目标风格中位数）",
            "p_main_median_across_targets": "迁移后 P_t 中位数（跨目标风格中位数）",
            "frac_t0_chem_viol_mean": "T0 化学违约比例（跨目标风格平均比例）",
            "frac_alpha0_mean": "α*=0 比例（跨目标风格平均比例）",
        }
    )
    t2 = df[["src"] + [f"{c}_mean" for c in FRACS]].rename(
        columns={
            "src": "源风格",
            "frac_ref_mean": "达到参照范围",
            "frac_improved_mean": "改善",
            "frac_unchanged_mean": "保持原谱",
            "frac_no_improve_mean": "未改善",
        }
    )
    n1 = (
        "每个源风格汇总其 21 个目标风格，21 个目标等权；连续位置量取 21 个逐对汇总值的中位数，"
        "比例取 21 个逐对比例的算术平均；风格对不视为独立啤酒。"
        "逐对明细见 q3\\_transfer\\_pairs.csv（462 行）。"
    )
    n2 = (
        "四类按方法论 §6.4 的先后顺序互斥判定；每格为该源风格 21 个目标风格逐对比例的算术平均，"
        "分母为该源风格的啤酒数，四列之和为 1。"
    )
    tex = to_tex(t1, "跨风格迁移按源风格汇总（一）", "tab:q3_pairs_a", n1)
    tex += "\n" + to_tex(
        t2, "跨风格迁移按源风格汇总（二）：结论类别（跨目标风格平均比例）", "tab:q3_pairs_b", n2
    )
    (FIG_DIR / "tab_q3_pairs.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
