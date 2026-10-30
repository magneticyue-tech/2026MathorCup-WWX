import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1 import plotstyle  # noqa: E402
from src.q1.data import DIM_GROUPS, DIMS  # noqa: E402
from src.q1.paths import FIG_DIR  # noqa: E402
from src.q1.report_beers import REPORT_ROWS  # noqa: E402
from src.q1.tables import latex_table  # noqa: E402

CSV_COLS = (
    "beer_row",
    "beer",
    "style",
    "n_ratings",
    "dim",
    "G_hat",
    "lo",
    "hi",
    "G_obs",
    "score_cell",
)
SOURCE = {"main": "化学主模型", "chemstyle": "化学 + 风格回退模型"}


def build(src):
    path = Path(src) / "q1_scores_long.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    long = pd.read_csv(path, encoding="utf-8-sig")
    n_ratings = plotstyle.load_npz(Path(src) / "q1_data.npz")["n_ratings"]
    sub = long[long["beer_row"].isin(REPORT_ROWS) & long["dim"].isin(DIMS)].copy()
    sub["n_ratings"] = n_ratings[sub["beer_row"].to_numpy() - 1]
    sub["score_cell"] = (
        sub["G_hat"].astype(str)
        + "["
        + sub["lo"].astype(str)
        + ","
        + sub["hi"].astype(str)
        + "]/"
        + sub["G_obs"].astype(str)
    )
    order = {d: i for i, d in enumerate(DIMS)}
    sub = sub.sort_values(["beer_row", "dim"], key=lambda s: s.map(order) if s.name == "dim" else s)
    return sub[list(CSV_COLS)].reset_index(drop=True)


def tex_tables(df, model_key):
    beers = df.drop_duplicates("beer_row")[["beer_row", "beer", "style", "n_ratings"]]
    note = (
        "单元格为预测分布中位数 $\\hat G$ 及 90\\% 等级区间 [lo,hi]，斜杠后为实测等级 $G$；"
        f"加粗表示预测与实测不同。交付来源：{SOURCE[model_key]}（model\\_key={model_key}）。"
    )
    mapping = latex_table(
        beers.rename(
            columns={"beer_row": "行号", "beer": "beer", "style": "风格", "n_ratings": "评分条数"}
        ),
        "rlll",
        "正文报告啤酒的编号对照",
        "tab:q1_report_beers_map",
    ).replace("\\begin{table}[htbp]", "\\begin{table}[h]")
    parts = []
    for g, dims in DIM_GROUPS.items():
        dims = [d for d in DIMS if d in dims]
        cells = df.pivot(index="dim", columns="beer_row", values="score_cell").loc[dims]
        diff = (
            df.assign(d=df["G_hat"] != df["G_obs"])
            .pivot(index="dim", columns="beer_row", values="d")
            .loc[dims]
        )
        cols = list(cells.columns)
        code = dict(zip(beers["beer_row"], beers["beer"]))
        tab = cells.reset_index()
        tab.columns = ["维度", *[str(code[c]) for c in cols]]
        mask = [[False, *diff.loc[d, cols].tolist()] for d in dims]
        body = latex_table(
            tab,
            "l" + "c" * len(cols),
            f"正文报告啤酒的逐维评分（{g} 组）。{note}",
            f"tab:q1_report_beers_{g.lower()}",
            longtable=True,
            bold_mask=mask,
        )
        parts.append(
            "\\begin{landscape}\n\\scriptsize\n"
            + mapping.replace("tab:q1_report_beers_map", f"tab:q1_report_beers_map_{g.lower()}")
            + body
            + "\\end{landscape}\n"
        )
    return "\n".join(parts)


def main():
    args = plotstyle.script_args(FIG_DIR)
    df = build(args.src)
    model_key = plotstyle.load_json(Path(args.src) / "q1_delivery_scores.json")["model_key"]
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "tab_q1_report_beers.csv", index=False, encoding="utf-8-sig")
    (args.out / "tab_q1_report_beers.tex").write_text(tex_tables(df, model_key), encoding="utf-8")
    ok = len(df) == len(REPORT_ROWS) * len(DIMS) and not df.duplicated(["beer_row", "dim"]).any()
    print(f"{args.out / 'tab_q1_report_beers.csv'}：{len(df)} 行，{'通过' if ok else '不通过'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
