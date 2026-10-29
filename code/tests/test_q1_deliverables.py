import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
from src.q1.data import DIM_GROUPS, DIMS
from src.q1.export import BARNYARD_FLAG, build_long, build_wide
from src.q1.plotstyle import PALETTE, significance_color
from src.q1.report_beers import REPORT_ROWS
from src.q1.tables import latex_table

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FIG_SCRIPTS = sorted(
    p for p in SCRIPTS.glob("fig_q1_*.py") if "methodology_flowchart" not in p.name
)
TAB_SCRIPTS = sorted(SCRIPTS.glob("tab_q1_*.py"))
EXPORT_SCRIPTS = sorted(SCRIPTS.glob("export_q1_*.py"))
ALLOWED_SRC = {"paths", "plotstyle", "tables", "export", "report_beers", "data"}
ALLOWED_DATA = {"DIMS", "DIM_GROUPS", "ZERO_HEAVY_DIMS", "load_raw_ratings"}
STAT_KEYS = ("mae", "exact", "within1", "qwk", "rps", "logloss", "cont_mae")
LONG_COLS = [
    "beer_row",
    "beer",
    "style",
    "dim",
    "G_hat",
    "G_hat_Q",
    *[f"p{k}" for k in range(7)],
    "lo",
    "hi",
    "G_obs",
    "xbar_obs",
    "mu_hat",
    "barnyard_flag",
]


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_latex_table():
    df = pd.DataFrame({"a_b": ["x%y", "p&q#"], "v": [1.23456, np.nan]})
    tex = latex_table(df, "lr", "cap", "lab", bold_mask=[[False, True], [False, False]])
    assert r"a\_b" in tex and r"x\%y" in tex and r"p\&q\#" in tex
    assert "--" in tex and r"\textbf{1.235}" in tex
    assert r"\begin{table}" in tex and "longtable" not in tex
    lt = latex_table(df, "lr", "cap", "lab", longtable=True)
    assert r"\begin{longtable}" in lt and r"\endhead" in lt and r"\endfirsthead" in lt


def test_build_wide():
    G = np.arange(6).reshape(3, 2) % 7
    w = build_wide([1, 2, 3], [10, 20, 30], G, ["d0", "d1"])
    assert list(w.columns) == ["beer_row", "beer", "d0", "d1"] and w.shape == (3, 4)
    assert (w["beer_row"] != w["beer"]).all()


def test_build_long():
    n, dims = 2, ["d0", "barnyard", "d2"]
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(7), size=(n, 3))
    G = rng.integers(0, 7, (n, 3))
    df = build_long([1, 2], [50, 60], ["s", "t"], dims, G, G, p, G, G, G, G * 1.0, G * 1.0)
    assert len(df) == 6 and list(df.columns) == LONG_COLS
    np.testing.assert_array_equal(df[[f"p{k}" for k in range(7)]].to_numpy(), p.reshape(6, 7))
    flagged = df[df["barnyard_flag"] != ""]
    assert (flagged["dim"] == "barnyard").all() and (
        flagged["barnyard_flag"] == BARNYARD_FLAG
    ).all()
    assert len(flagged) == n


def test_significance_color():
    assert significance_color(0.1, 0.2) == PALETTE["worse"]
    assert significance_color(-0.2, -0.1) == PALETTE["better"]
    assert significance_color(-0.1, 0.1) == PALETTE["neutral"]


def _pair(a, b, rng):
    pt = {k: float(rng.normal(0, 0.02)) for k in STAT_KEYS}
    return {
        "a": a,
        "b": b,
        "point": pt,
        "lo": {k: v - 0.01 for k, v in pt.items()},
        "hi": {k: v + 0.01 for k, v in pt.items()},
        "wins_a": int(rng.integers(0, 50)),
    }


@pytest.fixture(scope="module")
def fake_src(tmp_path_factory):
    rng = np.random.default_rng(0)
    src = tmp_path_factory.mktemp("src")
    rules = ("const", "round", "median", "quant", "qwk_cut", "median_knn")
    models = {}

    def entry():
        e = {k: float(rng.uniform(0.2, 0.6)) for k in ("mae", "exact", "within1", "qwk")}
        e.update(balanced_acc=0.3, share56=0.01, rps=0.3, logloss=1.0, cont_mae=0.48)
        e.update(cont_r2=0.1, cont_spearman=0.3)
        e["zero_heavy"] = {"n_dims": 22, "mae": 0.4, "exact": 0.6, "qwk": 0.2, "qwk_undefined": 0}
        e["other"] = {"n_dims": 28, "mae": 0.5, "exact": 0.5, "qwk": 0.3, "qwk_undefined": 0}
        return e

    for m in ("main", "b0", "b1", "lin10", "enet", "rf", "chemstyle", "c2"):
        for r in rules:
            models[f"{m}.{r}"] = entry()
    models["k2.median"] = entry()
    models["c1.sel"] = entry()
    ctrls = ("b0", "b1", "lin10", "enet", "rf", "chemstyle", "k2", "c1", "c2")
    paired = [_pair("main.median", "b1.median", rng)]
    paired += [_pair(f"{c}.{'sel' if c == 'c1' else 'median'}", "main.median", rng) for c in ctrls]
    ref = {
        k: {"cells_per_rep": 100.0, "exact": 0.6, "within1": 0.9, "mae": 0.4, "bias": 0.3}
        for k in ("all", "A", "L")
    }
    metrics = {
        "models": models,
        "climatology": {"rps": 0.4, "logloss": 1.06},
        "paired": paired,
        "reference_error": ref,
        "reliability": rng.uniform(0.1, 0.9, 50).tolist(),
        "r2_main": rng.uniform(0, 0.5, 50).tolist(),
        "leak_dims": [],
        "grade_freq": {"truth": (np.ones(7) / 7).tolist()},
        "success": True,
    }
    (src / "q1_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    cm = rng.integers(0, 100, (7, 7)).astype(float)
    np.savez(src / "q1_metrics.npz", conf_main_median=cm, conf_k2_median=cm, conf_c1_sel=cm)
    arms = {
        a: {**_pair(a, "main", rng), "changed_delivered": 0.05}
        for a in ("na_median_ind", "drop_sum", "alpha")
    }
    (src / "q1_sensitivity.json").write_text(json.dumps({"arms": arms}), encoding="utf-8")
    per_style = [
        {
            "style": f"S{i}",
            "n": int(n),
            "small": bool(n <= 5),
            "mae_median": 0.5,
            "b0_mae": 0.6,
            "b1_mae": 0.55,
        }
        for i, n in enumerate((3, 8, 20))
    ]
    (src / "q1_stress.json").write_text(json.dumps({"per_style": per_style}), encoding="utf-8")
    np.savez(src / "q1_baselines.npz", b2_r2=rng.uniform(0, 0.3, 50))
    ext_rows = [
        {
            "beer": b,
            "dim": d,
            "ctrl_mean": 2.0,
            "ctrl_se": 0.2,
            "ctrl_grade": 2,
            "G_hat": 2,
            "G_obs": 2,
        }
        for b in (64, 223)
        for d in DIMS[:11]
    ]
    s = {
        "hat_mae": 0.5,
        "hat_exact": 5,
        "hat_bias": -0.1,
        "obs_mae": 0.3,
        "obs_exact": 7,
        "obs_bias": 0.0,
    }
    ext = {
        "model_key": "main",
        "rows": ext_rows,
        "summary": {"64": s, "223": s, "all": s},
        "match": {"blond": {"code": 64}, "NABLAB": {"code": 223}},
        "ctrl_n": {"blond": 33, "NABLAB": 13},
    }
    (src / "q1_external.json").write_text(json.dumps(ext), encoding="utf-8")
    beer = np.arange(250) + 1000
    G = rng.integers(0, 7, (250, 50))
    p = rng.dirichlet(np.ones(7), size=(250, 50))
    long = build_long(
        np.arange(1, 251),
        beer,
        ["sty"] * 250,
        DIMS,
        G,
        G,
        p,
        np.maximum(G - 1, 0),
        np.minimum(G + 1, 6),
        np.roll(G, 1, 0),
        G * 1.0,
        G * 1.0,
    )
    long.to_csv(src / "q1_scores_long.csv", index=False, encoding="utf-8-sig")
    np.savez(src / "q1_data.npz", n_ratings=rng.integers(5, 12, 250))
    (src / "q1_delivery_scores.json").write_text(
        json.dumps({"model_key": "main"}), encoding="utf-8"
    )
    return src


@pytest.mark.parametrize("script", FIG_SCRIPTS, ids=lambda p: p.stem)
def test_fig_scripts_run(script, fake_src, tmp_path):
    subprocess.run(
        [sys.executable, str(script), "--src", str(fake_src), "--out", str(tmp_path)], check=True
    )
    pdf = tmp_path / f"{script.stem}.pdf"
    assert pdf.exists() and pdf.stat().st_size > 0 and pdf.read_bytes()[:4] == b"%PDF"


@pytest.mark.parametrize("script", TAB_SCRIPTS, ids=lambda p: p.stem)
def test_tab_scripts_run(script, fake_src, tmp_path):
    subprocess.run(
        [sys.executable, str(script), "--src", str(fake_src), "--out", str(tmp_path)], check=True
    )
    assert (tmp_path / f"{script.stem}.csv").exists() and (tmp_path / f"{script.stem}.tex").exists()


def test_fig_scripts_missing_input(tmp_path):
    r = subprocess.run(
        [sys.executable, str(FIG_SCRIPTS[0]), "--src", str(tmp_path), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "FileNotFoundError" in r.stderr
    assert not list(tmp_path.glob("*.pdf"))


def test_report_table_structure(fake_src, tmp_path):
    mod = load_script(SCRIPTS / "tab_q1_report_beers.py")
    df = mod.build(fake_src)
    assert len(df) == 600 and list(df.columns) == list(mod.CSV_COLS)
    assert not df.duplicated(["beer_row", "dim"]).any()
    nr = np.load(fake_src / "q1_data.npz")["n_ratings"]
    assert (df["n_ratings"].to_numpy() == nr[df["beer_row"].to_numpy() - 1]).all()
    r0 = df.iloc[0]
    assert r0["score_cell"] == f"{r0['G_hat']}[{r0['lo']},{r0['hi']}]/{r0['G_obs']}"
    assert not df["score_cell"].str.contains(r"\\", regex=False).any()
    groups = [set(v) for v in DIM_GROUPS.values()]
    assert sum(len(g) for g in groups) == len(set().union(*groups)) == len(DIMS)
    tex = mod.tex_tables(df, "main")
    assert tex.count(r"\begin{longtable}") == 6 and tex.count(r"\begin{landscape}") == 6
    assert tex.count(r"\endhead") == 6 and "model\\_key=main" in tex
    beers = df.drop_duplicates("beer_row")
    assert set(beers["beer_row"]) == set(REPORT_ROWS)
    for b in beers["beer"]:
        assert str(b) in tex
    diff = df[df["G_hat"] != df["G_obs"]].iloc[0]
    assert r"\textbf{" + diff["score_cell"] + "}" in tex
    same = df[df["G_hat"] == df["G_obs"]]
    if len(same):
        assert r"\textbf{" + same.iloc[0]["score_cell"] + "}" not in tex


@pytest.mark.skipif(shutil.which("xelatex") is None, reason="未安装 xelatex")
def test_report_table_compiles(fake_src, tmp_path):
    mod = load_script(SCRIPTS / "tab_q1_report_beers.py")
    tex = mod.tex_tables(mod.build(fake_src), "main")
    doc = (
        "\\documentclass[12pt,a4paper]{article}\n\\usepackage{ctex}\n\\usepackage{booktabs}\n"
        "\\usepackage{longtable}\n\\usepackage{pdflscape}\n\\usepackage[margin=2.5cm]{geometry}\n"
        "\\begin{document}\n" + tex + "\\end{document}\n"
    )
    (tmp_path / "t.tex").write_text(doc, encoding="utf-8")
    for _ in range(2):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    log = (tmp_path / "t.log").read_text(encoding="utf-8", errors="ignore")
    assert "Missing character" not in log
    assert "Overfull \\hbox" not in log and "Overfull \\vbox" not in log
    subprocess.run(["pdftoppm", "-r", "30", "-png", "t.pdf", "pg"], cwd=tmp_path, check=True)
    pages = sorted(tmp_path.glob("pg-*.png"))
    assert pages
    for pg in pages:
        img = mpimg.imread(pg)
        assert img.std() > 0.01, f"{pg.name} 是空白页"
    landscape = [pg for pg in pages if mpimg.imread(pg).shape[1] > mpimg.imread(pg).shape[0]]
    assert landscape, "没有横向页面"


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods, calls = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
            mods.append((node.module, [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("src"):
                    mods.append((a.name, []))
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            calls.append(name)
    return mods, calls


@pytest.mark.parametrize("script", FIG_SCRIPTS + TAB_SCRIPTS + EXPORT_SCRIPTS, ids=lambda p: p.stem)
def test_script_imports(script):
    mods, calls = _imports(script)
    for mod, names in mods:
        if mod == "src.q1":
            assert set(names) <= ALLOWED_SRC, (script.name, names)
            continue
        assert mod.startswith("src.q1."), (script.name, mod)
        sub = mod.split(".")[2]
        assert sub in ALLOWED_SRC, (script.name, mod)
        if sub == "data":
            assert set(names) <= ALLOWED_DATA, (script.name, names)
    assert not {"fit", "apply_rules", "nested_cv_predict"} & set(calls), script.name
