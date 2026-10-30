import math
from pathlib import Path

import numpy as np

_ESC = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}


def escape(s) -> str:
    return "".join(_ESC.get(ch, ch) for ch in str(s))


def _is_missing(v) -> bool:
    return (
        v is None
        or (isinstance(v, float) and math.isnan(v))
        or (isinstance(v, np.floating) and np.isnan(v))
    )


def _cell(v, float_fmt) -> str:
    if _is_missing(v):
        return "--"
    if isinstance(v, (bool, np.bool_)):
        return escape(str(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return float_fmt.format(float(v))
    return escape(v)


def latex_table(
    df,
    colfmt,
    caption,
    label,
    float_fmt="{:.3f}",
    longtable=False,
    bold_mask=None,
    escape_cells=True,
) -> str:
    header = " & ".join(escape(c) for c in df.columns) + r" \\"
    body = []
    vals = df.to_numpy(object)
    for i, row in enumerate(vals):
        cells = []
        for j, v in enumerate(row):
            c = _cell(v, float_fmt) if escape_cells or not isinstance(v, str) else v
            if bold_mask is not None and bool(bold_mask[i][j]):
                c = r"\textbf{" + c + "}"
            cells.append(c)
        body.append(" & ".join(cells) + r" \\")
    if longtable:
        lines = [
            r"\begin{longtable}{" + colfmt + "}",
            r"\caption{" + caption + r"}\label{" + label + r"}\\",
            r"\toprule",
            header,
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            header,
            r"\midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endfoot",
            *body,
            r"\end{longtable}",
        ]
    else:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + caption + r"}\label{" + label + "}",
            r"\begin{tabular}{" + colfmt + "}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    return "\n".join(lines) + "\n"


def write_table(df, out_dir, stem, caption, label, tex_df=None, colfmt=None, **kw) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig", na_rep="")
    tdf = df if tex_df is None else tex_df
    colfmt = colfmt or "l" * tdf.shape[1]
    text = latex_table(tdf, colfmt, caption, label, **kw)
    (out_dir / f"{stem}.tex").write_text(text, encoding="utf-8")
