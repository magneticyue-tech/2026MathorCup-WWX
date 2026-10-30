import json

import numpy as np
import pandas as pd
import pytest
from src.q1.data import load_raw_ratings
from src.q1.paths import FIG_DIR, PROCESSED_DIR
from src.q1.report_beers import REPORT_ROWS, multi_date_rows

KEYS = ("G_hat", "G_hat_Q", "p_final", "lo", "hi", "mu_hat")


def need(path):
    assert path.exists(), f"产物缺失: {path}"
    return path


@pytest.fixture(scope="module")
def delivery():
    return np.load(need(PROCESSED_DIR / "q1_delivery_scores.npz"), allow_pickle=False)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(need(PROCESSED_DIR / "q1_metrics.json").read_text(encoding="utf-8"))


def test_delivery_source(delivery, metrics):
    key = str(delivery["model_key"])
    assert key == ("main" if metrics["success"] else "chemstyle")
    if key == "main":
        s = np.load(PROCESSED_DIR / "q1_scores.npz", allow_pickle=False)
        oof = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=False)["oof_mu"]
        ref = {k: s[k] for k in KEYS[:5]} | {"mu_hat": oof.mean(0)}
    else:
        b = np.load(PROCESSED_DIR / "q1_baselines.npz", allow_pickle=False)
        names = {"p_final": "p"}
        ref = {k: b[f"chemstyle_final_{names.get(k, k)}"] for k in KEYS}
    for k in KEYS:
        np.testing.assert_array_equal(delivery[k], ref[k])


def test_external_source(delivery):
    key = str(delivery["model_key"])
    ej = json.loads(need(PROCESSED_DIR / "q1_external.json").read_text(encoding="utf-8"))
    en = np.load(need(PROCESSED_DIR / "q1_external.npz"), allow_pickle=False)
    assert ej["model_key"] == key and str(en["model_key"]) == key
    tex = need(FIG_DIR / "tab_q1_external.tex").read_text(encoding="utf-8")
    assert f"model\\_key={key}" in tex


def test_wide(delivery):
    w = pd.read_csv(need(PROCESSED_DIR / "q1_final_scores.csv"), encoding="utf-8-sig")
    assert w.shape == (250, 52)
    assert w["beer_row"].tolist() == list(range(1, 251))
    np.testing.assert_array_equal(w.iloc[:, 2:].to_numpy(), delivery["G_hat"])


def test_long(delivery):
    lg = pd.read_csv(need(PROCESSED_DIR / "q1_scores_long.csv"), encoding="utf-8-sig")
    w = pd.read_csv(PROCESSED_DIR / "q1_final_scores.csv", encoding="utf-8-sig")
    assert len(lg) == 12500
    p = lg[[f"p{k}" for k in range(7)]].to_numpy()
    assert np.abs(p.sum(1) - 1).max() <= 1e-12
    assert ((lg["lo"] <= lg["G_hat"]) & (lg["G_hat"] <= lg["hi"])).all()
    dims = list(w.columns[2:])
    assert (lg["G_hat"].to_numpy().reshape(250, 50) == w[dims].to_numpy()).all()
    assert (np.floor(np.clip(lg["xbar_obs"], 0, 6) + 0.5).astype(int) == lg["G_obs"]).all()


def test_report_rows():
    assert list(REPORT_ROWS) == multi_date_rows(load_raw_ratings())
    assert len(REPORT_ROWS) == 12


def test_metrics_table_matches_json(metrics):
    t = pd.read_csv(need(FIG_DIR / "tab_q1_metrics.csv"), encoding="utf-8-sig")
    fields = [c for c in t.columns if c not in ("key", "模型")]
    for _, row in t.iterrows():
        if row["key"] in ("b2", "climatology"):
            continue
        m = metrics["models"][row["key"]]
        for f in fields:
            v = m.get(f)
            if v is None:
                assert pd.isna(row[f])
            else:
                assert abs(row[f] - v) <= 1e-9, (row["key"], f)
    clim = t[t["key"] == "climatology"].iloc[0]
    assert abs(clim["rps"] - metrics["climatology"]["rps"]) <= 1e-9
    assert abs(clim["logloss"] - metrics["climatology"]["logloss"]) <= 1e-9


def test_paired_rows(metrics):
    t = pd.read_csv(need(FIG_DIR / "tab_q1_paired.csv"), encoding="utf-8-sig")
    assert len(t) == len(metrics["paired"])
