import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.tables import write_table  # noqa: E402
from src.q2 import config as C  # noqa: E402

ATTR_LABELS = {
    "Esters": "Esters",
    "Fullness (body)": "Fullness",
    "Alcohol (warming)": "Alcohol warming",
    "Sweetness": "Sweetness",
    "Hops": "Hops",
    "Bitterness": "Bitterness",
    "Malt": "Malt",
    "Overall appreciation": "Overall",
}
SET_LABELS = {"Best predictors": "含乙醇", "Best predictors - no ethanol": "无乙醇"}


def main():
    d = pd.read_csv(C.Q2_DIR / "q2_spike.csv")
    d["组"] = [f"{st} {SET_LABELS[s]} n={n}" for st, s, n in zip(d["style"], d["set"], d["n"])]
    d["单元"] = [f"{m:.2f} ({t:.1f})" for m, t in zip(d["mean_diff"], d["t"])]
    order = list(dict.fromkeys(d["组"]))
    wide = d.pivot(index="attribute", columns="组", values="单元").loc[list(ATTR_LABELS), order]
    wide.index = [ATTR_LABELS[a] for a in wide.index]
    wide = wide.reset_index().rename(columns={"index": "属性"})
    write_table(
        wide,
        C.FIGURES,
        "tab_q2_spike",
        caption="加标杯$-$对照杯配对差（括号内为 $t$ 值；同一品评员同日配对）",
        label="tab:q2_spike",
        colfmt="l" + "c" * len(order),
    )
    print(C.FIGURES / "tab_q2_spike.csv")


if __name__ == "__main__":
    main()
