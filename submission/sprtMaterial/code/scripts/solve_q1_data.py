import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.data import (  # noqa: E402
    DIMS,
    ZERO_HEAVY_DIMS,
    ChemNAEncoder,
    attendance_share,
    barnyard_check,
    check_id_mapping,
    floor_indicators,
    load_q1,
    load_raw_ratings,
    panel_date_nesting,
    reconstruct_attachment4,
    split_half_reliability,
)
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.paths import PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402


def _py(v):
    if isinstance(v, (set, frozenset)):
        return sorted(_py(x) for x in v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_py(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _close(o, e, tol) -> bool:
    if tol is None:
        return o == e
    return bool(np.all(np.abs(np.asarray(o, float) - np.asarray(e, float)) <= tol))


def make_check(name, observed, expected, tol=None) -> dict:
    o, e = _py(observed), _py(expected)
    if isinstance(tol, list):
        passed = len(o) == len(e) == len(tol) and all(map(_close, o, e, tol))
    else:
        passed = _close(o, e, tol)
    return {"name": name, "observed": o, "expected": e, "tol": tol, "passed": bool(passed)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    require_sklearn()
    ensure_dirs()
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    n_splits = 2 if args.smoke else 20

    data = load_q1()
    ratings = load_raw_ratings()
    t = data.targets
    a4 = data.Z.assign(beer=data.beer.values)

    mapping = check_id_mapping(ratings, t.xbar, a4)
    ok, n_const, n_const_dims = reconstruct_attachment4(ratings, a4)
    half = split_half_reliability(ratings, n_splits=n_splits, seed=SEED)
    att = attendance_share(ratings, t.xbar)
    barn = barnyard_check(ratings, t.xbar)
    nest = panel_date_nesting(data.Z, t.date_group)

    rel = pd.Series(half["reliability"], index=DIMS)
    att_s = pd.Series(att, index=DIMS)
    g_counts = np.bincount(t.G.ravel(), minlength=7)
    zero_heavy = (t.G == 0).mean(0) >= 0.5
    group_sizes = pd.Series(t.date_group).value_counts()
    floor = floor_indicators(data.X)
    encoded = ChemNAEncoder("col_mean").fit(data.X).transform(data.X)
    unrepro_dims = set(np.array(DIMS)[(~ok).any(0)])
    const_dims = set(np.array(DIMS)[(ratings.groupby("taster")[DIMS].std() == 0).any().values])

    checks = [
        make_check(
            "评分表形状",
            [len(ratings), ratings["taster"].nunique(), ratings["date"].nunique(), len(t.xbar)],
            [2197, 16, 37, 250],
        ),
        make_check("编号对应 Spearman（按行号）", mapping["spearman_row"], 0.969, 0.002),
        make_check("编号对应 Spearman（按 beer 列）", mapping["spearman_beer"], -0.010, 0.002),
        make_check("轮廓指派等于行号", mapping["assignment_match"], 250),
        make_check("附件 4 重建一致单元", int(ok.sum()), 11274),
        make_check("恒值品评员 × 维度组合 / 所在维度数", [n_const, n_const_dims], [15, 12]),
        make_check("未复现单元全部落在这 12 个维度", unrepro_dims <= const_dims, True),
        make_check("G 等级计数", g_counts, [4674, 3579, 2057, 1545, 524, 112, 9]),
        make_check(
            "零级占比 ≥ 0.5 的维度数，且等于 ZERO_HEAVY_DIMS",
            [int(zero_heavy.sum()), set(np.array(DIMS)[zero_heavy]) == set(ZERO_HEAVY_DIMS)],
            [22, True],
        ),
        make_check(
            "日期组个数 / 最小 / 最大",
            [group_sizes.size, group_sizes.min(), group_sizes.max()],
            [25, 5, 24],
        ),
        make_check("多日期品评啤酒数", (ratings.groupby("row")["date"].nunique() > 1).sum(), 12),
        make_check(
            "品评团（k=3）个数 / 覆盖啤酒 / 跨日期组数",
            [nest["n_multi"], nest["n_covered"], nest["n_cross_date"]],
            [41, 228, 0],
        ),
        make_check("信度中位数", rel.median(), 0.53, 0.01),
        make_check("信度最小 / 最大", [rel.min(), rel.max()], [0.09, 0.88], 0.02),
        make_check(
            "信度最低的四个维度",
            set(rel.nsmallest(4).index),
            {"A_hops_woody", "F_malt_grain", "t2n", "A_malt_grain"},
        ),
        make_check(
            "半组等级完全一致率 / QWK",
            [half["half_grade_exact"], half["half_grade_qwk"]],
            [0.497, 0.374],
            0.01,
        ),
        make_check("出席偏移占比中位数", att_s.median(), 0.022, 0.003),
        make_check(
            "出席偏移占比最大维度 / 值",
            [att_s.idxmax(), att_s.max()],
            ["A_malt_grain", 0.152],
            [None, 0.005],
        ),
        make_check(
            "barnyard 与 stale_hops 评分不同的记录数 / Spearman",
            [barn["n_differ"], barn["spearman"]],
            [14, 0.938],
            [None, 0.002],
        ),
        make_check("附件 1 的 NA 总数", int(data.X.isna().sum().sum()), 73),
        make_check("col_mean 编码后化学表的 NaN 个数", int(encoded.isna().sum().sum()), 0),
        make_check(
            "下限值单元数 / 所在列数", [int(floor.values.sum()), floor.shape[1]], [6807, 142]
        ),
    ]

    out = out_dir / "q1_data.npz"
    np.savez_compressed(
        out,
        beer_row=np.arange(1, len(t.xbar) + 1),
        beer=data.beer.values,
        style=data.style.values.astype(str),
        xbar=t.xbar,
        G=t.G,
        n_ratings=t.n_ratings,
        date_group=t.date_group,
        dims=np.array(DIMS),
        reconstruct_ok=ok,
        reliability=half["reliability"],
        half_grade_exact=half["half_grade_exact"],
        half_grade_qwk=half["half_grade_qwk"],
        attendance_share=att,
        zero_heavy=zero_heavy,
        g_counts=g_counts,
        barnyard_identical_share=barn["identical_share"],
        barnyard_spearman=barn["spearman"],
        map_spearman_row=mapping["spearman_row"],
        map_spearman_beer=mapping["spearman_beer"],
        map_assignment_match=mapping["assignment_match"],
        n_panels_k3=nest["n_multi"],
        panels_cross_date=nest["n_cross_date"],
        checks=json.dumps(checks, ensure_ascii=False),
        seed=SEED,
        meta=run_meta(),
    )

    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    print(f"写出 {out}")

    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
