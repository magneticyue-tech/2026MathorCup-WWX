import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.paths import FIG_DIR, PROCESSED_DIR  # noqa: E402
from src.q3.tables import to_tex  # noqa: E402

CAT = {"reach": "达到参照范围", "improved": "改善", "kept": "保持原谱", "not_improved": "未改善"}


def main():
    t = pd.read_csv(PROCESSED_DIR / "q3_transfer_pils_tripel.csv", encoding="utf-8-sig")
    out = pd.DataFrame(
        {
            "beer": t["beer"],
            "α*": t["alpha_star"],
            "α_fit": t["alpha_fit"],
            "有效约束数": t["n_active"],
            "起约束感官维度": t["binding_sens_dims"].fillna("—"),
            "P 源": t["p_src"],
            "P T0": t["p_t0"],
            "P T1": t["p_t1"],
            "P T5": t["p_t5"],
            "P 主方法": t["p_main"],
            "P T2": t["p_t2"],
            "d² 主方法": t["d2_main"],
            "保留度": t["retention"],
            "结论类别": t["category"].map(CAT),
            "双乙酰源缺失": t["src_missing_diacetyl"],
        }
    )
    out.to_csv(FIG_DIR / "tab_q3_transfer.csv", index=False, encoding="utf-8-sig")
    note = (
        "P 为交叉拟合判别器给出的 P(Tripel)；"
        f"真实 Tripel 折外 P 的 25\\% 分位为 {t['ref_q25_tripel'].iloc[0]:.3f}。"
        "T2 无满足两层约束的真实 Tripel 时记为 —。"
        "双乙酰源缺失的啤酒，化学层结论为“除双乙酰（源缺失）外未恶化”。"
    )
    (FIG_DIR / "tab_q3_transfer.tex").write_text(
        to_tex(out, "Pils/Lager → Tripel 11 款逐款迁移结果", "tab:q3_transfer", note),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
