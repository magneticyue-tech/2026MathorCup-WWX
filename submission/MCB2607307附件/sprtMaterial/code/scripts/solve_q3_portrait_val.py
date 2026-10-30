import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import N_BOOT, N_SYNTH  # noqa: E402
from src.q3.data import load_chem  # noqa: E402
from src.q3.meta import dump_json, run_meta  # noqa: E402
from src.q3.paths import (  # noqa: E402
    CHUNK_DIR,
    PROCESSED_DIR,
    Q1_DIR,
    SEED,
    SMOKE_DIR,
    ensure_dirs,
)  # noqa: E402
from src.q3.portrait_val import METRIC_NAMES, cell_metrics, fold_unit  # noqa: E402
from src.q3.q1_bridge import load_main_oof  # noqa: E402

BOOT_SEED = SEED


def run(path, chem, xbar, G, oof, rep, fold, n):
    u = fold_unit(chem, xbar, G, oof, rep, fold, n, SEED)
    with open(path, "wb") as f:
        pickle.dump(u, f, protocol=4)


def size_group(n):
    return "n<=5" if n <= 5 else ("6-15" if n <= 15 else ">15")


def style_metrics(units, styles, w=None):
    out = {}
    for st in styles:
        vals = []
        for u in units:
            c = u["cells"].get(st)
            if c is None:
                continue
            m = cell_metrics(c, None if w is None else w[c["test_rows"]])
            if m is not None:
                vals.append(m)
        out[st] = np.nanmean(vals, 0) if vals else np.full(len(METRIC_NAMES), np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ensure_dirs()
    t0 = time.time()
    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    chunk = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / "portrait_val"
    chunk.mkdir(parents=True, exist_ok=True)
    chem = load_chem()
    z = np.load(Q1_DIR / "q1_data.npz", allow_pickle=False)
    xbar, G = z["xbar"], z["G"]
    assert (z["style"] == chem.style).all()
    oof = load_main_oof()
    R, K = oof["fold_id"].shape[0], int(oof["fold_id"].max()) + 1
    cells = [(0, 0), (0, 1)] if args.smoke else [(r, k) for r in range(R) for k in range(K)]
    n = 50 if args.smoke else N_SYNTH
    jobs = [
        delayed(run)(chunk / f"r{r}_k{k}.pkl", chem, xbar, G, oof, r, k, n)
        for r, k in cells
        if not (chunk / f"r{r}_k{k}.pkl").exists()
    ]
    print(f"待计算 {len(jobs)}", flush=True)
    Parallel(n_jobs=args.workers, verbose=5)(jobs)
    units = []
    for r, k in cells:
        with open(chunk / f"r{r}_k{k}.pkl", "rb") as f:
            units.append(pickle.load(f))

    styles = chem.styles
    n_style = {st: int((chem.style == st).sum()) for st in styles}
    weights = np.array([n_style[st] for st in styles], float)
    weights /= weights.sum()
    est = style_metrics(units, styles)
    E = np.array([est[st] for st in styles])
    n_eval = {st: sum(st in u["cells"] for u in units) for st in styles}
    ok = np.isfinite(E[:, 0])

    def agg(Em):
        okm = np.isfinite(Em[:, 0])
        macro = np.nanmean(Em[okm], 0)
        w = weights[okm] / weights[okm].sum()
        return macro, np.nansum(Em[okm] * w[:, None], 0)

    macro, weighted = agg(E)
    nb = 20 if args.smoke else N_BOOT
    rng = np.random.default_rng(BOOT_SEED)
    bm, bw = [], []
    for _ in range(nb):
        w = np.bincount(
            rng.integers(0, len(chem.style), len(chem.style)), minlength=len(chem.style)
        )
        eb = style_metrics(units, styles, w)
        m1, m2 = agg(np.array([eb[st] for st in styles]))
        bm.append(m1)
        bw.append(m2)
    bm, bw = np.array(bm), np.array(bw)
    groups = {}
    for gname in ("n<=5", "6-15", ">15"):
        sel = np.array([size_group(n_style[st]) == gname for st in styles]) & ok
        groups[gname] = {
            "n_styles": int(sel.sum()),
            **{m: float(np.nanmean(E[sel, i])) for i, m in enumerate(METRIC_NAMES)},
        }
    np.savez_compressed(
        out_dir / "q3_portrait_val.npz",
        styles=np.array(styles),
        metric_names=np.array(METRIC_NAMES),
        style_metrics=E,
        n_eval_folds=np.array([n_eval[st] for st in styles]),
        n_style=np.array([n_style[st] for st in styles]),
        macro=macro,
        weighted=weighted,
        boot_macro=bm,
        boot_weighted=bw,
        seed=SEED,
        meta=run_meta(),
    )
    js = {
        "n_evaluable_styles": int(ok.sum()),
        "eval_folds_by_style": n_eval,
        "folds_style_missing_in_train": {
            f"{u['rep']}_{u['fold']}": u["styles_missing_train"] for u in units
        },
        "folds_style_one_in_train": {
            f"{u['rep']}_{u['fold']}": u["styles_one_train"] for u in units
        },
        "fold_configs": {f"{u['rep']}_{u['fold']}": u["config"] for u in units},
        "inner_excluded": {f"{u['rep']}_{u['fold']}": u["inner_excluded"] for u in units},
        "summary": {
            m: {
                "macro": float(macro[i]),
                "macro_ci95": np.nanpercentile(bm[:, i], [2.5, 97.5]).tolist(),
                "weighted": float(weighted[i]),
                "weighted_ci95": np.nanpercentile(bw[:, i], [2.5, 97.5]).tolist(),
            }
            for i, m in enumerate(METRIC_NAMES)
        },
        "by_size_group": groups,
        "by_style": {
            st: {m: float(E[si, i]) for i, m in enumerate(METRIC_NAMES)}
            for si, st in enumerate(styles)
        },
        "wall_time_s": time.time() - t0,
    }
    dump_json(js, out_dir / "q3_portrait_val.json")
    print({m: round(float(macro[i]), 3) for i, m in enumerate(METRIC_NAMES)})
    print("evaluable", int(ok.sum()), "wall", round(time.time() - t0))


if __name__ == "__main__":
    main()
