import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.data import load_chem  # noqa: E402
from src.q3.generator import StyleGen  # noqa: E402
from src.q3.meta import dump_json, make_check, run_meta  # noqa: E402
from src.q3.paths import CHUNK_DIR, PROCESSED_DIR, SEED, SMOKE_DIR, ensure_dirs  # noqa: E402
from src.q3.selection import (  # noqa: E402
    CONFIGS,
    _skf,
    inner_scores,
    inner_splits,
    outer_splits,
    select_config,
)


def run_cell(path, X, S, styles, tr, rep):
    sc, small, excl = inner_scores(X, S, tr, rep, styles=styles)
    np.savez(path, scores=sc, small=small, excluded=excl)
    return path


def prerun(X, S, styles, out_dir):
    sets = {
        "famod k=15 κ=2": dict(kind="famod", k=15, kappa=2.0),
        "fa k=15 κ=2": dict(kind="fa", k=15, kappa=2.0),
        "lw κ=2": dict(kind="lw", k=None, kappa=2.0),
        "diag κ=2": dict(kind="diag", k=None, kappa=2.0),
        "global fa k=15": dict(kind="fa", k=15, kappa=2.0, global_only=True),
    }
    expect = {
        "famod k=15 κ=2": -240.7,
        "fa k=15 κ=2": -244.9,
        "lw κ=2": -267.1,
        "diag κ=2": -297.3,
        "global fa k=15": -241.9,
    }
    res = {k: [] for k in sets}
    for rep in range(2):
        for tr, te in _skf(5, SEED + rep, S):
            for name, cfg in sets.items():
                g = StyleGen(**cfg).fit(X[tr], S[tr], styles)
                res[name].append(float(np.nanmean(g.score(X[te], S[te]))))
    out = {k: float(np.mean(v)) for k, v in res.items()}
    checks = [make_check(f"prerun {k}", out[k], expect[k], 1.0) for k in sets]
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["name"], round(c["observed"], 2), c["expected"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json({"mean_L": out, "checks": checks}, out_dir / "q3_select_prerun.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preset", choices=["final", "prerun"], default="final")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ensure_dirs()
    chem = load_chem()
    X, S, styles = chem.XG, chem.style, chem.styles
    if args.preset == "prerun":
        prerun(X, S, styles, SMOKE_DIR)
        return

    out_dir = SMOKE_DIR if args.smoke else PROCESSED_DIR
    chunk = (SMOKE_DIR / "_chunks" if args.smoke else CHUNK_DIR) / "select"
    chunk.mkdir(parents=True, exist_ok=True)
    splits = outer_splits(S)
    if args.smoke:
        splits = [c for c in splits if c[0] == 0 and c[1] < 2]
    t0 = time.time()
    jobs = []
    for rep, fold, tr, _ in splits:
        p = chunk / f"r{rep}_k{fold}.npz"
        if not p.exists():
            jobs.append(delayed(run_cell)(p, X, S, styles, tr, rep))
    pf = chunk / "full.npz"
    if not pf.exists():
        jobs.append(delayed(run_cell)(pf, X, S, styles, np.arange(len(S)), 0))
    print(f"待计算单元 {len(jobs)}", flush=True)
    Parallel(n_jobs=args.workers, verbose=5)(jobs)

    L, Ls, ex, chosen = [], [], [], []
    for rep, fold, _, _ in splits:
        z = np.load(chunk / f"r{rep}_k{fold}.npz")
        L.append(z["scores"])
        Ls.append(z["small"])
        ex.append(int(z["excluded"]))
        chosen.append(select_config(z["scores"]))
        print(rep, fold, chosen[-1], int(z["excluded"]))
    zf = np.load(pf)
    final = select_config(zf["scores"])
    print("final_config", final)

    # 划分核对
    faro_out, brut_out, faro_in, brut_in, n_faro1 = [], [], [], [], 0
    for rep, fold, tr, _ in outer_splits(S):
        faro_out.append(int((S[tr] == "Faro").sum()))
        brut_out.append(int((S[tr] == "Brut").sum()))
        for itr, _ in inner_splits(tr, S, rep):
            nf = int((S[itr] == "Faro").sum())
            faro_in.append(nf)
            brut_in.append(int((S[itr] == "Brut").sum()))
            n_faro1 += nf == 1
    checks = [
        make_check("外层训练集 Faro 款数范围", [min(faro_out), max(faro_out)], [2, 3]),
        make_check("外层训练集 Brut 款数范围", [min(brut_out), max(brut_out)], [3, 4]),
        make_check(
            "内层训练集 Faro ≥1 且 Brut ≥2", bool(min(faro_in) >= 1 and min(brut_in) >= 2), True
        ),
        make_check("内层训练集恰 1 款 Faro 的个数", n_faro1, 30),
        make_check("inner_excluded 全为 0", bool(sum(ex) == 0), True),
    ]
    wall = time.time() - t0
    freq = Counter(json.dumps(c, sort_keys=True) for c in chosen)
    np.savez_compressed(
        out_dir / "q3_select.npz",
        outer_rep=np.array([c[0] for c in splits]),
        outer_fold=np.array([c[1] for c in splits]),
        chosen_kind=np.array([c["kind"] for c in chosen]),
        chosen_k=np.array([c["k"] for c in chosen]),
        chosen_kappa=np.array([c["kappa"] for c in chosen]),
        configs=np.array([json.dumps(c) for c in CONFIGS]),
        inner_L=np.array(L),
        inner_small_L=np.array(Ls),
        inner_excluded=np.array(ex),
        final_config=json.dumps(final),
        final_inner_L=zf["scores"],
        final_small_L=zf["small"],
        seed=SEED,
        meta=run_meta(),
    )
    dump_json(
        {
            "final_config": final,
            "chosen_freq": dict(freq),
            "k_range": [min(c["k"] for c in chosen), max(c["k"] for c in chosen)],
            "kappa_range": [min(c["kappa"] for c in chosen), max(c["kappa"] for c in chosen)],
            "small_L_vs_kappa_final": {
                str(cfg): float(zf["small"][i])
                for i, cfg in enumerate(CONFIGS)
                if cfg[0] == final["kind"] and cfg[1] == final["k"]
            },
            "inner_excluded": ex,
            "checks": checks,
            "wall_time_s": wall,
        },
        out_dir / "q3_select.json",
    )
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["name"], c["observed"], c["expected"])
    print(f"wall {wall:.0f}s")


if __name__ == "__main__":
    main()
