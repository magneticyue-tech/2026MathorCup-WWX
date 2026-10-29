import numpy as np
from scipy.stats import norm

TIE_SCHEMES = ("cond_mean", "mean_rank")


class Marg:
    """逐列秩变换与原子感知的逆经验分布（方法论 §2.2）。"""

    def __init__(self, tie_scheme: str = "cond_mean"):
        if tie_scheme not in TIE_SCHEMES:
            raise ValueError(f"未知并列方案: {tie_scheme}")
        self.tie_scheme = tie_scheme

    def fit(self, M):
        M = np.asarray(M, float)
        self.blocks = []
        for j in range(M.shape[1]):
            x = M[:, j][~np.isnan(M[:, j])]
            if len(x) == 0:
                raise ValueError(f"第 {j} 列没有非缺失观测")
            vals, cnt = np.unique(x, return_counts=True)
            n = len(x)
            hi = np.cumsum(cnt) / n
            lo = hi - cnt / n
            if self.tie_scheme == "cond_mean":
                e = (norm.pdf(norm.ppf(lo)) - norm.pdf(norm.ppf(hi))) / (hi - lo)
            else:
                avg_rank = (lo * n + 1 + hi * n) / 2
                e = norm.ppf((avg_rank - 0.5) / n)
            kp, kv = [], []
            for v, c, lk, hk, ek in zip(vals, cnt, lo, hi, e):
                if c >= 2:
                    kp += [lk + 1e-9, hk - 1e-9]
                    kv += [v, v]
                else:
                    kp.append(norm.cdf(ek))
                    kv.append(v)
            self.blocks.append((vals, e, np.array(kp), np.array(kv)))
        return self

    def transform(self, M):
        M = np.asarray(M, float)
        if np.isnan(M).any():
            raise ValueError("transform 的输入含 NaN，须先填充")
        return np.column_stack([np.interp(M[:, j], b[0], b[1]) for j, b in enumerate(self.blocks)])

    def inverse(self, U):
        P = norm.cdf(np.atleast_2d(np.asarray(U, float)))
        return np.column_stack([np.interp(P[:, j], b[2], b[3]) for j, b in enumerate(self.blocks)])

    def inverse_column(self, j, v):
        b = self.blocks[j]
        return np.interp(norm.cdf(np.asarray(v, float)), b[2], b[3])

    def col_min(self):
        return np.array([b[0][0] for b in self.blocks])

    def col_max(self):
        return np.array([b[0][-1] for b in self.blocks])
