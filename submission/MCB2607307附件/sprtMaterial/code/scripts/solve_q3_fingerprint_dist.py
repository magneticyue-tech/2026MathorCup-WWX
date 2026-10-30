"""化学指纹图数据：三个命名风格的前 10 项在标准化潜空间中的生成样本与真实啤酒 -> q3_fingerprint_dist.npz。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.config import NAMED_STYLES  # noqa: E402
from src.q3.fullmodel import fit_full, load_final_config, variant_data  # noqa: E402
from src.q3.meta import run_meta  # noqa: E402
from src.q3.paths import PROCESSED_DIR, SEED  # noqa: E402

TOP = 10


def main():
    chem, _ = variant_data("base")
    gen, _ = fit_full(chem, load_final_config(PROCESSED_DIR / "q3_select.json"))
    syn = np.load(PROCESSED_DIR / "q3_synthetic_profiles.npz", allow_pickle=False)
    fp = pd.read_csv(PROCESSED_DIR / "q3_fingerprint.csv")
    styles = [str(s) for s in syn["styles"]]
    cols = list(chem.gen_cols)
    sd = np.sqrt(np.diag(gen.C))
    out = {}
    for k, st in enumerate(NAMED_STYLES):
        f = fp[(fp["style"] == st) & (fp["rank"] <= TOP)].sort_values("rank")
        j = np.array([cols.index(c) for c in f["compound"]])
        u_syn = gen.latent(syn["profiles"][styles.index(st)].astype(float))
        z_syn = (u_syn[:, j] - gen.g[j]) / sd[j]
        z_real = (gen.U[np.asarray(chem.style) == st][:, j] - gen.g[j]) / sd[j]
        out[f"s{k}_compound"] = f["compound"].to_numpy(str)
        out[f"s{k}_delta"] = f["delta"].to_numpy()
        out[f"s{k}_lo"] = f["delta_lo"].to_numpy()
        out[f"s{k}_hi"] = f["delta_hi"].to_numpy()
        out[f"s{k}_syn"] = z_syn
        out[f"s{k}_real"] = z_real
    path = PROCESSED_DIR / "q3_fingerprint_dist.npz"
    np.savez_compressed(path, styles=np.array(NAMED_STYLES), seed=SEED, meta=run_meta(), **out)
    print(f"写出 {path}")


if __name__ == "__main__":
    main()
