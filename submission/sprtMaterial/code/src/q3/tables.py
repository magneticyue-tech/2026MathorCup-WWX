import numpy as np
import pandas as pd


def _fmt(v, nd=3):
    if isinstance(v, (bool, np.bool_)):
        return "是" if v else "否"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        return f"{v:.{nd}f}"
    return str(v).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def to_tex(df: pd.DataFrame, caption: str, label: str, note: str = "", nd=3, size="small") -> str:
    cols = list(df.columns)
    head = " & ".join(_fmt(c) for c in cols) + r" \\"
    body = "\n".join(
        " & ".join(_fmt(v, nd) for v in row) + r" \\" for row in df.itertuples(index=False)
    )
    out = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\{size}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}",
        r"\hline",
        head,
        r"\hline",
        body,
        r"\hline",
        r"\end{tabular}}",
    ]
    if note:
        out.append(f"\\par\\footnotesize 注：{note}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"
