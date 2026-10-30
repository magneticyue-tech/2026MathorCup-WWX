import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q1.export import build_long, build_wide  # noqa: E402
from src.q1.paths import PROCESSED_DIR, SEED  # noqa: E402


def need(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sha12(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def meta_json():
    return json.dumps(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    src, out = args.src, args.out
    out.mkdir(parents=True, exist_ok=True)

    with open(need(src / "q1_metrics.json"), encoding="utf-8") as f:
        success = json.load(f)["success"]
    if not isinstance(success, bool):
        raise KeyError("q1_metrics.json 的 success 不是布尔值")
    d = np.load(need(src / "q1_data.npz"), allow_pickle=False)
    beer_row, beer, style = d["beer_row"], d["beer"], d["style"]
    G, xbar, dims = d["G"], d["xbar"], [str(x) for x in d["dims"]]
    assert beer_row.tolist() == list(range(1, len(beer_row) + 1)), "beer_row 不是 1..250"
    for a in (beer, style, G, xbar, d["n_ratings"]):
        assert len(a) == len(beer_row), "数组首轴与 beer_row 不对齐"

    if success:
        model_key = "main"
        s_path, m_path = need(src / "q1_scores.npz"), need(src / "q1_main_oof.npz")
        s = np.load(s_path, allow_pickle=False)
        G_hat, G_hat_Q, p_final, lo, hi = (
            s[k] for k in ("G_hat", "G_hat_Q", "p_final", "lo", "hi")
        )
        mu_hat = np.load(m_path, allow_pickle=False)["oof_mu"].mean(0)
        inputs = {"q1_scores.npz": sha12(s_path), "q1_main_oof.npz": sha12(m_path)}
    else:
        model_key = "chemstyle"
        b_path = need(src / "q1_baselines.npz")
        b = np.load(b_path, allow_pickle=False)
        G_hat, G_hat_Q, p_final, lo, hi, mu_hat = (
            b[f"chemstyle_final_{k}"] for k in ("G_hat", "G_hat_Q", "p", "lo", "hi", "mu_hat")
        )
        inputs = {"q1_baselines.npz": sha12(b_path)}
    inputs["q1_data.npz"] = sha12(src / "q1_data.npz")
    inputs["q1_metrics.json"] = sha12(src / "q1_metrics.json")
    for a in (G_hat, G_hat_Q, p_final, lo, hi, mu_hat):
        assert len(a) == len(beer_row), "交付数组首轴与 beer_row 不对齐"

    np.savez_compressed(
        out / "q1_delivery_scores.npz",
        model_key=model_key,
        G_hat=G_hat,
        G_hat_Q=G_hat_Q,
        p_final=p_final,
        lo=lo,
        hi=hi,
        mu_hat=mu_hat,
        seed=SEED,
        meta=meta_json(),
    )
    with open(out / "q1_delivery_scores.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": model_key,
                "selection_basis": {"success": success},
                "inputs_sha256_12": inputs,
                "seed": SEED,
                "meta": json.loads(meta_json()),
            },
            f,
            ensure_ascii=False,
            indent=1,
        )

    dv = np.load(out / "q1_delivery_scores.npz", allow_pickle=False)
    wide = build_wide(beer_row, beer, dv["G_hat"], dims)
    long = build_long(
        beer_row,
        beer,
        style,
        dims,
        dv["G_hat"],
        dv["G_hat_Q"],
        dv["p_final"],
        dv["lo"],
        dv["hi"],
        G,
        xbar,
        dv["mu_hat"],
    )
    wide.to_csv(out / "q1_final_scores.csv", index=False, encoding="utf-8-sig")
    long.to_csv(out / "q1_scores_long.csv", index=False, encoding="utf-8-sig")
    ok_wide = wide.shape == (len(beer_row), len(dims) + 2)
    ok_long = len(long) == len(beer_row) * len(dims)
    print(
        f"model_key={model_key}；宽表 {wide.shape} {'通过' if ok_wide else '不通过'}；"
        f"长表 {len(long)} 行 {'通过' if ok_long else '不通过'}"
    )
    if not (ok_wide and ok_long):
        sys.exit(1)


if __name__ == "__main__":
    main()
