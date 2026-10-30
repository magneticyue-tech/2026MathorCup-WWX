import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.tables import write_table  # noqa: E402
from src.q2 import config as C  # noqa: E402

METHODS = (("M1", "边际选择"), ("main", "前向选择"))


def triple(row, key, level):
    return ", ".join(f"{row[f'{key}_{level}_recovery_{k}']:.2f}" for k in (1, 2, 3))


def main():
    sim = pd.read_csv(C.Q2_DIR / "simulation_recovery_results.csv")
    rows = []
    for _, r in sim.iterrows():
        rec = {
            "植入权重": r["weights"],
            "设定 r": r["target_r"],
            "实现首位 |r| 中位": round(float(r["observed_r1_median"]), 3),
            "重复次数": int(r["n_sim"]),
        }
        for key, name in METHODS:
            rec[f"{name}·化合物层"] = triple(r, key, "comp")
            rec[f"{name}·模块层"] = triple(r, key, "mod")
            rec[f"{name}·非植入模块占比"] = round(float(r[f"{key}_nonplanted_module_share"]), 3)
        rows.append(rec)
    df = pd.DataFrame(rows)
    tex = df.drop(columns=["重复次数"] + [f"{n}·非植入模块占比" for _, n in METHODS])
    write_table(
        df,
        C.FIGURES,
        "tab_q2_recovery",
        caption=(
            "植入--还原实验找回率（按植入顺序第 1、2、3 位；"
            f"每设置 {int(sim['n_sim'].iloc[0])} 次，"
            f"种子 {int(sim['seed'].iloc[0])}）"
        ),
        label="tab:q2_recovery",
        tex_df=tex,
        colfmt="ccc" + "cc" * len(METHODS),
        float_fmt="{:.2f}",
    )
    print(C.FIGURES / "tab_q2_recovery.csv")


if __name__ == "__main__":
    main()
