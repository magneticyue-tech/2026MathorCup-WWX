import numpy as np


def _style_rows(s_tr, style):
    idx = np.where(np.asarray(s_tr) == style)[0]
    if len(idx) == 0:
        raise ValueError(f"训练集中没有风格 {style}")
    return idx


def draw_g1(Xf_tr, s_tr, style, n, rng):
    idx = _style_rows(s_tr, style)
    return Xf_tr[rng.choice(idx, n)]


def draw_g2(Xf_tr, s_tr, style, n, rng):
    idx = _style_rows(s_tr, style)
    return np.column_stack([Xf_tr[rng.choice(idx, n), j] for j in range(Xf_tr.shape[1])])


def draw_g3(gen, s_tr, style, n, rng):
    idx = _style_rows(s_tr, style)
    a, b = rng.choice(idx, n), rng.choice(idx, n)
    lam = rng.uniform(size=(n, 1))
    return gen.mg.inverse(lam * gen.U[a] + (1 - lam) * gen.U[b])
