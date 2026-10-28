import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q1_data import make_check  # noqa: E402
from solve_q1_metrics import has_string_numbers, sanitize  # noqa: E402
from src.q1.data import load_q1  # noqa: E402
from src.q1.external import (  # noqa: E402
    BASE_STYLES,
    EXPECTED_BASE_CODES,
    EXPECTED_CONTROL_N,
    EXPECTED_MD5,
    check_md5,
    compare_rows,
    control_means,
    load_control_ratings,
    match_base_beers,
    resolve_workbook,
    summarize,
)
from src.q1.meta import require_sklearn, run_meta  # noqa: E402
from src.q1.paths import PROCESSED_DIR, RAW_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402


def load(name):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    require_sklearn()
    ensure_dirs()
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR

    try:
        path = resolve_workbook()
        md5 = check_md5(path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"外部检查无法进行：{e}")
        sys.exit(2)

    cal = pd.read_excel(RAW_DIR / "8.xlsx", sheet_name="Calibrated chemical dataset")
    match = match_base_beers(path, cal)
    for style in BASE_STYLES:
        assert match[style]["code"] == EXPECTED_BASE_CODES[style], (style, match[style])
    beer = load_q1().beer.to_numpy()
    rows = {}
    for style in BASE_STYLES:
        r = np.where(beer == match[style]["code"])[0]
        assert len(r) == 1, f"beer {match[style]['code']} 匹配到 {len(r)} 行"
        rows[style] = int(r[0])

    with open(load("q1_metrics.json"), encoding="utf-8") as f:
        success = json.load(f)["success"]
    if success:
        model_key = "main"
        s = np.load(load("q1_scores.npz"), allow_pickle=False)
        G_hat, G_hat_Q, p_final = s["G_hat"], s["G_hat_Q"], s["p_final"]
    else:
        model_key = "chemstyle"
        b = np.load(load("q1_baselines.npz"), allow_pickle=False)
        G_hat = b["chemstyle_final_G_hat"]
        G_hat_Q = b["chemstyle_final_G_hat_Q"]
        p_final = b["chemstyle_final_p"]
    z = np.load(load("q1_data.npz"), allow_pickle=False)
    G, xbar, dims = z["G"], z["xbar"], [str(d) for d in z["dims"]]

    ctrl, counts = load_control_ratings(path)
    tables, frames, ctrl_n = {}, [], {}
    for style in BASE_STYLES:
        t = control_means(ctrl, style)
        tables[style] = t
        ctrl_n[style] = int(t["n"].iloc[0])
        frames.append(
            compare_rows(
                match[style]["code"], [rows[style]], t, G_hat, G_hat_Q, G, xbar, p_final, dims
            )
        )
    df = pd.concat(frames, ignore_index=True)
    summary = summarize(df)

    n_styles = len(BASE_STYLES)
    by = {s: df[df["beer"] == match[s]["code"]] for s in BASE_STYLES}
    arr = {
        k: np.stack([by[s][k].to_numpy() for s in BASE_STYLES])
        for k in ("ctrl_mean", "ctrl_se", "ctrl_grade", "G_hat", "G_hat_Q", "G_obs", "xbar")
    }
    ext_dims = by[BASE_STYLES[0]]["dim"].tolist()
    p_sel = np.stack([p_final[rows[s], [dims.index(d) for d in ext_dims]] for s in BASE_STYLES])

    out_json = {
        "model_key": model_key,
        "success": success,
        "match": match,
        "beer_rows": {s: rows[s] for s in BASE_STYLES},
        "ctrl_n": ctrl_n,
        "n_all": counts["n_all"],
        "n_dedup": counts["n_dedup"],
        "summary": summary,
        "rows": df.to_dict(orient="records"),
        "seed": SEED,
        "meta": json.loads(run_meta()),
    }
    out_json = sanitize(out_json)
    checks = [
        make_check("外部检查：工作簿 md5", md5, EXPECTED_MD5),
        make_check(
            "外部检查：匹配到的底酒 beer 列取值（blond、NABLAB）",
            [match[s]["code"] for s in BASE_STYLES],
            [EXPECTED_BASE_CODES[s] for s in BASE_STYLES],
        ),
        make_check(
            "外部检查：对照杯评分条数（blond、NABLAB）",
            [ctrl_n[s] for s in BASE_STYLES],
            [EXPECTED_CONTROL_N[s] for s in BASE_STYLES],
        ),
        make_check("外部检查：每款啤酒匹配到的行恰为 1 行", len(rows) == n_styles, True),
        make_check("外部检查：对照杯表的维度数", len(ext_dims), 11),
        make_check(
            "所有结果 JSON 中没有字符串形式的数值或状态（除 _reason 字段）",
            not has_string_numbers(
                {k: v for k, v in out_json.items() if k not in ("rows", "summary", "match")}
            )
            and not has_string_numbers(out_json["summary"]),
            True,
        ),
    ]
    out_json["checks"] = checks
    with open(out_dir / "q1_external.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=1, allow_nan=False)
    np.savez_compressed(
        out_dir / "q1_external.npz",
        model_key=model_key,
        beer_codes=np.array([match[s]["code"] for s in BASE_STYLES]),
        beer_rows=np.array([rows[s] for s in BASE_STYLES]),
        match_dist=np.array([match[s]["dist"] for s in BASE_STYLES]),
        match_dist2=np.array([match[s]["dist2"] for s in BASE_STYLES]),
        dims=np.array(ext_dims),
        ctrl_n=np.array([ctrl_n[s] for s in BASE_STYLES]),
        p_final=p_sel,
        summary=json.dumps(summary),
        md5=md5,
        n_all=counts["n_all"],
        n_dedup=counts["n_dedup"],
        checks=json.dumps(checks, ensure_ascii=False),
        seed=SEED,
        meta=run_meta(),
        **arr,
    )
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {c['name']}: observed={c['observed']} expected={c['expected']}")
    print(f"model_key={model_key}；summary={json.dumps(summary)}")
    print(f"match={json.dumps(match)}")
    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        print("失败项：" + "；".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
