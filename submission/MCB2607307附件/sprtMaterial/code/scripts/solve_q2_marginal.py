"""主方法 Top-3 的合并边际 Spearman，与风格内系数并列 -> q2_marginal.csv。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "q2"))
import config as C  # noqa: E402
import core  # noqa: E402

from data import load_q2_dataset  # noqa: E402


def main():
    ds = load_q2_dataset()
    a = core.prepare_arm(ds, core.ArmSpec("main"))
    _, _, _, Cxy = core.select_full(a)
    fin = pd.read_csv(C.Q2_DIR / "final_top3_with_rationality.csv")
    m = fin[fin["method"] == "main"].copy()
    ci = {c: i for i, c in enumerate(a.cand)}
    di = {d: i for i, d in enumerate(a.dims)}
    m["r_pooled"] = [Cxy[ci[c], di[d]] for c, d in zip(m["compound"], m["dimension"])]
    out = m[
        [
            "dimension",
            "rank",
            "compound",
            "r_pooled",
            "r_style",
            "r_style_ci_lower",
            "r_style_ci_upper",
            "flag_style_ci0",
        ]
    ]
    path = C.Q2_DIR / "q2_marginal.csv"
    out.to_csv(path, index=False)
    same = (out["r_pooled"] * out["r_style"] > 0).mean()
    ratio = (out["r_style"] / out["r_pooled"]).median()
    print(
        f"同号 {same:.2%}；比值中位 {ratio:.2f}；|r_style|<0.1 {int((out['r_style'].abs() < 0.1).sum())} 项"
    )
    print(f"写出 {path}")


if __name__ == "__main__":
    main()
