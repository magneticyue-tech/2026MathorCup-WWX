import hashlib
import json
import platform

import numpy
import pandas
import scipy
import sklearn


def run_meta(**extra) -> str:
    return json.dumps(
        {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            **extra,
        },
        ensure_ascii=False,
    )


def sha12(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def to_py(v):
    if isinstance(v, numpy.generic):
        return v.item()
    if isinstance(v, numpy.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [to_py(x) for x in v]
    if isinstance(v, dict):
        return {str(k): to_py(x) for k, x in v.items()}
    return v


def make_check(name, observed, expected, tol=None) -> dict:
    o, e = to_py(observed), to_py(expected)
    if tol is None:
        passed = o == e
    elif isinstance(o, list):
        passed = len(o) == len(e) and all(abs(a - b) <= tol for a, b in zip(o, e))
    else:
        passed = abs(o - e) <= tol
    return {"name": name, "expected": e, "observed": o, "tol": tol, "pass": bool(passed)}


def dump_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_py(obj), f, ensure_ascii=False, indent=1, allow_nan=True)
