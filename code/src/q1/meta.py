import json
import platform
import sys

import numpy
import pandas
import scipy
import sklearn


def run_meta() -> str:
    return json.dumps(
        {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        }
    )


def require_sklearn(version: str = "1.7.2") -> None:
    if sklearn.__version__ != version:
        raise RuntimeError(
            f"需要 scikit-learn {version}，实际为 {sklearn.__version__}；PY = {sys.executable}"
        )
