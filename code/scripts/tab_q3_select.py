import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.paths import FIG_DIR, PROCESSED_DIR  # noqa: E402
from src.q3.tables import to_tex  # noqa: E402


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    z = np.load(PROCESSED_DIR / "q3_select.npz", allow_pickle=False)
    cfgs = [tuple(json.loads(c)) for c in z["configs"]]
    final = json.loads(str(z["final_config"]))
    L, Ls = z["final_inner_L"], z["final_small_L"]
    outer = Counter(
        (str(k), int(a), float(b))
        for k, a, b in zip(z["chosen_kind"], z["chosen_k"], z["chosen_kappa"])
    )
    rows = []
    for i, (kind, k, kap) in enumerate(cfgs):
        rows.append(
            {
                "协方差": kind,
                "k": "—" if k is None else int(k),
                "κ": kap,
                "全数据内层 L": L[i],
                "小样本风格内层 L": Ls[i],
                "外层选中次数": outer.get((kind, k, kap), 0),
                "全数据选定": kind == final["kind"] and k == final["k"] and kap == final["kappa"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / "tab_q3_select.csv", index=False, encoding="utf-8-sig")
    show = df[(df["κ"] == final["kappa"]) | (df["外层选中次数"] > 0) | df["全数据选定"]]
    show = show.sort_values("全数据内层 L", ascending=False).head(20)
    tex = to_tex(
        show,
        "生成器嵌套选参：全数据内层留出潜空间评分（nat/款）",
        "tab:q3_select",
        f"全数据选定 {final['kind']}、k={final['k']}、κ={final['kappa']:g}；"
        "列出 κ 取选定值的设定与外层被选中的设定；"
        "完整 96 个设定见 tab\\_q3\\_select.csv。小样本风格 L 只报告，不参与选参。",
        nd=2,
    )
    (FIG_DIR / "tab_q3_select.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
