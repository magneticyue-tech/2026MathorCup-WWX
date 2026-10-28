# code/src/q2/simulation.py
"""Q2 §7.2 模拟还原：在真实化学秩上植入 3 个不同模块的化合物，比较主方法与 M1 的找回率。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd

import config as C
import core
from data import load_q2_dataset


def simulate(ds, n_sim, seed):
    a = core.prepare_arm(ds, core.ArmSpec("main"))
    X, mod, n = a.Zx, a.mod, a.Zx.shape[0]
    Cxx = X.T @ X / n
    valid = np.ones(X.shape[1], bool)
    pan = np.unique(ds.panels, return_inverse=True)[1]
    rows = []
    for w in C.SIM_WEIGHTS:
        w = np.asarray(w)
        for target in C.SIM_TARGET_R:
            rng = np.random.default_rng(seed)
            hit = {m: np.zeros((n_sim, 3), bool) for m in ("main", "M1")}
            mhit = {m: np.zeros((n_sim, 3), bool) for m in ("main", "M1")}
            other = {m: np.zeros(n_sim) for m in ("main", "M1")}
            r_obs = np.zeros(n_sim)
            for i in range(n_sim):
                while True:
                    js = rng.choice(X.shape[1], 3, replace=False)
                    if len(set(mod[js])) == 3:
                        break
                S = X[:, js] @ w
                S = (S - S.mean()) / S.std()
                k = min(0.99, target / np.corrcoef(S, X[:, js[0]])[0, 1])
                e = rng.normal(0, 1, n) + rng.normal(0, C.SIM_PANEL_OFFSET_SD, pan.max() + 1)[pan]
                e = (e - e.mean()) / e.std()
                ry, _ = core.zrank((k * S + np.sqrt(1 - k**2) * e)[:, None])
                Cxy = X.T @ ry / n
                r_obs[i] = Cxy[js[0], 0]
                main_sel, _ = core.forward_select(Cxx, Cxy, mod, valid)
                m1_sel = core.ranked_select(Cxy, mod, valid)
                for m, s in (("main", main_sel[0]), ("M1", m1_sel[0])):
                    hit[m][i] = [j in s for j in js]
                    mhit[m][i] = [mod[j] in set(mod[s]) for j in js]
                    other[m][i] = np.mean([mod[j] not in set(mod[js]) for j in s])
            row = {"weights": ",".join(f"{x:g}" for x in w), "target_r": target, "n_sim": n_sim,
                   "seed": seed, "observed_r1_median": float(np.median(r_obs))}
            for m in ("main", "M1"):
                for k in range(3):
                    row[f"{m}_comp_recovery_{k + 1}"] = hit[m][:, k].mean()
                for k in range(3):
                    row[f"{m}_mod_recovery_{k + 1}"] = mhit[m][:, k].mean()
                row[f"{m}_nonplanted_module_share"] = other[m].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main(n_sim=C.N_SIMULATION, seed=C.SEED, out_dir=C.Q2_DIR):
    ds = load_q2_dataset()
    df = simulate(ds, n_sim, seed)
    n_settings = len(df)
    if n_settings != 6 or (df["n_sim"] != n_sim).any():
        raise AssertionError("模拟设置应为 6 行，每种重复数一致")
    cols = [c for c in df.columns if "recovery" in c or "share" in c]
    if not ((df[cols] >= 0) & (df[cols] <= 1)).all().all():
        raise AssertionError("比例越界")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "simulation_recovery_results.csv", index=False)
    print(f"模拟 {n_settings} 种设置 × {n_sim} 次（seed {seed}）；找回率分母为模拟重复数")
    show = df[["weights", "target_r"] + [c for c in cols if "comp" in c or "share" in c]]
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        main(n_sim=10, out_dir=C.SMOKE_DIR)
    else:
        main()
