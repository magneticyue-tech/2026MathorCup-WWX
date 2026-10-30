"""主模型调参网格边界检查：向更少迭代、更浅深度扩展网格，看内层最优点是否落在网格内部。

只跑重复 0 的 5 个外层折，单调约束关闭；与正式运行 q1_main_oof.npz 的重复 0 比较外层折外指标。
用法（仓库根目录）：PYTHONUTF8=1 $PY experimentalCode/q1_raw_target/exp_grid_boundary.py
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from src.q1 import model as M  # noqa: E402
from src.q1.data import DIM_GROUPS, DIMS, load_q1  # noqa: E402
from src.q1.paths import PROCESSED_DIR, SEED  # noqa: E402
from src.q1.splits import outer_splits  # noqa: E402

ITERS = (10, 25, 50, 100, 200)
GRID = [
    {
        "max_depth": d,
        "min_samples_leaf": leaf,
        "learning_rate": 0.05,
        "l2_regularization": l2,
        "max_iter": it,
    }
    for d in (1, 2, 3)
    for leaf in (10, 20, 40)
    for l2 in (0.0, 1.0)
    for it in ITERS
]
OUT = ROOT / "experimentalCode" / "results" / "q1_raw_target" / "grid_boundary.json"


def run(X, Y, groups, style, tr, te, inner_seed):
    M.MONO_OPTIONS = (False,)
    return M.fit_outer_fold(X, Y, DIMS, DIM_GROUPS, groups, style, tr, te, inner_seed, GRID)


def summarize(oof, xbar):
    mae = float(np.mean(np.abs(oof - xbar)))
    ss_res = ((xbar - oof) ** 2).sum(0)
    ss_tot = ((xbar - xbar.mean(0)) ** 2).sum(0)
    rho = [spearmanr(oof[:, j], xbar[:, j])[0] for j in range(xbar.shape[1])]
    return {"mae": mae, "r2": float(np.mean(1 - ss_res / ss_tot)), "rho": float(np.mean(rho))}


def main():
    z = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=False)
    xbar, date_group = z["xbar"], z["date_group"]
    data = load_q1()
    splits = outer_splits(date_group, data.style)
    units = Parallel(n_jobs=5, verbose=5)(
        delayed(run)(data.X, xbar, date_group, data.style, tr, te, SEED + 100)
        for tr, te in splits[0]
    )
    oof = np.full_like(xbar, np.nan)
    by_iter = {g: {it: [] for it in ITERS} for g in DIM_GROUPS}
    by_depth = {g: {d: [] for d in (1, 2, 3)} for g in DIM_GROUPS}
    chosen = {g: [] for g in DIM_GROUPS}
    for u in units:
        oof[u["test_idx"]] = u["oof"]
        for g, sc in u["scores"].items():
            s = sc[0]
            for it in ITERS:
                by_iter[g][it].append(
                    float(np.nanmin([s[i] for i, c in enumerate(GRID) if c["max_iter"] == it]))
                )
            for d in (1, 2, 3):
                by_depth[g][d].append(
                    float(np.nanmin([s[i] for i, c in enumerate(GRID) if c["max_depth"] == d]))
                )
            chosen[g].append(u["chosen"][g])

    main_oof = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=False)["oof_mu"][0]
    out = {
        "grid": {"max_depth": [1, 2, 3], "min_samples_leaf": [10, 20, 40], "max_iter": ITERS},
        "best_by_iter": {
            g: {it: float(np.mean(v)) for it, v in d.items()} for g, d in by_iter.items()
        },
        "best_by_depth": {
            g: {dd: float(np.mean(v)) for dd, v in d.items()} for g, d in by_depth.items()
        },
        "chosen": chosen,
        "outer_extended": summarize(oof, xbar),
        "outer_main_r0": summarize(main_oof, xbar),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("best_by_iter", "outer_extended", "outer_main_r0")}))


if __name__ == "__main__":
    main()
