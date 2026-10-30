import numpy as np
import pandas as pd

BARNYARD_FLAG = "疑似 stale_hops 副本"


def build_wide(beer_row, beer, G_hat, dims) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(G_hat).astype(int), columns=list(dims))
    df.insert(0, "beer", np.asarray(beer))
    df.insert(0, "beer_row", np.asarray(beer_row))
    return df


def build_long(
    beer_row, beer, style, dims, G_hat, G_hat_Q, p_final, lo, hi, G_obs, xbar, mu_hat
) -> pd.DataFrame:
    dims = list(dims)
    n, D = np.asarray(G_hat).shape
    rep = lambda a: np.repeat(np.asarray(a), D)  # noqa: E731
    flat = lambda a: np.asarray(a).reshape(n * D)  # noqa: E731
    df = pd.DataFrame(
        {
            "beer_row": rep(beer_row),
            "beer": rep(beer),
            "style": rep(style),
            "dim": np.tile(dims, n),
            "G_hat": flat(G_hat).astype(int),
            "G_hat_Q": flat(G_hat_Q).astype(int),
        }
    )
    p = np.asarray(p_final).reshape(n * D, 7)
    for k in range(7):
        df[f"p{k}"] = p[:, k]
    df["lo"] = flat(lo).astype(int)
    df["hi"] = flat(hi).astype(int)
    df["G_obs"] = flat(G_obs).astype(int)
    df["xbar_obs"] = flat(xbar)
    df["mu_hat"] = flat(mu_hat)
    df["barnyard_flag"] = np.where(df["dim"] == "barnyard", BARNYARD_FLAG, "")
    return df
