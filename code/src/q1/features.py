MONOTONE_PAIRS = {
    "alcohol": ("ethanol..v.v.", 1),
    "bitternes": ("bitterness", 1),
    "acidity": ("pH", -1),
    "lactic": ("lactic_acid.mg.L.", 1),
    "acetic": ("acetic_acid.g.L.", 1),
    "X4vg": ("X4VG", 1),
}


def assert_pair_features(feature_names: list[str]) -> None:
    missing = [f for f, _ in MONOTONE_PAIRS.values() if f not in feature_names]
    if missing:
        raise KeyError(f"缺少单调配对化合物列: {missing}")


def monotone_cst(feature_names: list, dim: str, enabled: bool) -> list[int]:
    cst = [0] * len(feature_names)
    if enabled and dim in MONOTONE_PAIRS:
        feature, direction = MONOTONE_PAIRS[dim]
        cst[feature_names.index(feature)] = direction
    return cst
