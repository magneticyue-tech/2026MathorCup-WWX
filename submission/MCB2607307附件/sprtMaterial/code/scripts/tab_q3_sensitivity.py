import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.paths import FIG_DIR, PROCESSED_DIR  # noqa: E402
from src.q3.tables import to_tex  # noqa: E402


def _ci(r, k):
    d, lo, hi = r.get(f"{k}.diff"), r.get(f"{k}.diff_lo"), r.get(f"{k}.diff_hi")
    if pd.isna(d):
        return "—"
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _d(r, k):
    v = r.get(k)
    return "—" if pd.isna(v) else f"{v:+.3f}"


def main():
    df = pd.read_csv(PROCESSED_DIR / "q3_sensitivity.csv", encoding="utf-8-sig")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "臂": r["arm"],
                "改动": r["change"],
                "Δ 合并 AUC": _ci(r, "main.auc_marg"),
                "Δ TSTR": _ci(r, "main.tstr"),
                "Δ 模块内 ρ": _ci(r, "main.module_rho"),
                "Δ α* 中位（11 款）": _d(r, "pils_tripel.alpha_median.diff"),
                "Δ P 中位（11 款）": _d(r, "pils_tripel.p_median.diff"),
                "Δ α* 中位（462 对）": _d(r, "all_pairs.alpha_median.diff"),
                "Δ 约束起作用比例（462 对）": _d(r, "all_pairs.active_frac.diff"),
                "备注": "" if pd.isna(r.get("note")) else r.get("note"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(FIG_DIR / "tab_q3_sensitivity.csv", index=False, encoding="utf-8-sig")
    note = (
        "生成器臂的差值区间为与基线共用啤酒重抽索引的 2,000 次配对差 95\\% 区间；迁移臂报告点差。"
    )
    (FIG_DIR / "tab_q3_sensitivity.tex").write_text(
        to_tex(out, "问题 3 敏感性分析汇总（臂 − 基线）", "tab:q3_sensitivity", note),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
