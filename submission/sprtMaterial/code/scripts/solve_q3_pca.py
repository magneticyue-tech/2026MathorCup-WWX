"""潜空间 PCA：真实啤酒、各风格生成样本与 Pils/Lager → Tripel 迁移轨迹的投影 -> q3_pca.npz。"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import SRC_STYLE, TGT_STYLE  # noqa: E402
from src.q3.fullmodel import fit_full, load_final_config, variant_data  # noqa: E402
from src.q3.meta import dump_json, make_check, run_meta  # noqa: E402
from src.q3.paths import PROCESSED_DIR, SEED, ensure_dirs  # noqa: E402
from src.q3.sens_search import path_points  # noqa: E402

N_COMP = 2
N_SHOW = 200


def main():
    ensure_dirs()
    chem, _ = variant_data("base")
    gen, _ = fit_full(chem, load_final_config(PROCESSED_DIR / "q3_select.json"))
    styles = chem.styles
    syn = np.load(PROCESSED_DIR / "q3_synthetic_profiles.npz", allow_pickle=False)
    cz = np.load(PROCESSED_DIR / "q3_transfer_chem.npz", allow_pickle=False)
    tr = np.load(PROCESSED_DIR / "q3_transfer.npz", allow_pickle=False)
    paths = np.load(PROCESSED_DIR / "q3_transfer_pils_paths.npz", allow_pickle=False)
    n_syn = syn["profiles"].shape[1]

    U = gen.U
    own = np.array([styles.index(s) for s in chem.style])
    same_u = bool(np.allclose(cz["u_star"][np.arange(len(U)), own], U))
    lat, ok_profiles = [], True
    for si, st in enumerate(styles):
        u = gen.sample_latent(st, n_syn, np.random.default_rng([SEED, si]))
        x = gen.mg.inverse(u).astype(np.float32)
        ok_profiles &= bool(np.allclose(x, syn["profiles"][si], equal_nan=True))
        lat.append(u[:N_SHOW])
    lat = np.stack(lat)

    center = U.mean(0)
    _, sv, vt = np.linalg.svd(U - center, full_matrices=False)
    comps = vt[:N_COMP]
    evr = (sv**2 / (sv**2).sum())[:N_COMP]

    def proj(a):
        return (np.asarray(a) - center) @ comps.T

    ti = styles.index(TGT_STYLE)
    rows = paths["rows"].astype(int)
    alphas = paths["alphas"]
    traj = np.stack([path_points(U[i], cz["u_star"][i, ti], alphas) for i in rows])
    a_star = tr["alpha_star"][rows, ti]
    end = U[rows] + a_star[:, None] * (cz["u_star"][rows, ti] - U[rows])

    checks = [
        make_check("真实潜变量与化学层产物一致", same_u, True),
        make_check("重抽潜变量反变换与 q3_synthetic_profiles 一致", ok_profiles, True),
        make_check(
            "路径 α* 与 q3_transfer 一致", bool(np.allclose(a_star, paths["alpha_star"])), True
        ),
    ]
    out = PROCESSED_DIR / "q3_pca.npz"
    np.savez_compressed(
        out,
        styles=np.array(styles),
        style=np.asarray(chem.style, str),
        beer=np.asarray(chem.beer),
        evr=evr,
        real=proj(U),
        synth=proj(lat.reshape(-1, U.shape[1])).reshape(len(styles), -1, N_COMP),
        style_mean=proj(np.stack([gen.mu[s] for s in styles])),
        src_rows=rows,
        alphas=alphas,
        traj=proj(traj.reshape(-1, U.shape[1])).reshape(len(rows), len(alphas), N_COMP),
        feasible=paths["feasible"],
        end=proj(end),
        alpha_star=a_star,
        src_style=SRC_STYLE,
        tgt_style=TGT_STYLE,
        n_show=N_SHOW,
        seed=SEED,
        meta=run_meta(),
    )
    dump_json({"checks": checks, "evr": evr.tolist()}, PROCESSED_DIR / "q3_pca.json")
    for c in checks:
        print("PASS" if c["pass"] else "FAIL", c["name"])
    print(f"前两主成分解释方差 {evr.round(3).tolist()}；写出 {out}")
    if not all(c["pass"] for c in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
