import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.paths import PROCESSED_DIR  # noqa: E402


def main():
    z = np.load(PROCESSED_DIR / "q3_portraits.npz", allow_pickle=False)
    styles, dims = list(z["styles"]), list(z["dims"])
    rows = []
    for si, st in enumerate(styles):
        for j, d in enumerate(dims):
            r = {
                "style": st,
                "dim": d,
                "mu_q10": z["mu_q10"][si, j],
                "mu_q50": z["mu_q50"][si, j],
                "mu_q90": z["mu_q90"][si, j],
                "mode": int(z["mode"][si, j]),
            }
            r.update({f"pi{k}": z["pi"][si, j, k] for k in range(7)})
            r.update({f"pi_tilde{k}": z["pi_tilde"][si, j, k] for k in range(7)})
            r["weak_dim"] = bool(z["weak_dim"][j])
            r["barnyard_flag"] = "同 stale_hops" if d == "barnyard" else ""
            rows.append(r)
    pd.DataFrame(rows).to_csv(
        PROCESSED_DIR / "q3_portraits_long.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
