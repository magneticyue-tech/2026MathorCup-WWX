import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.gen_metrics import GEN_NAMES, SCALARS  # noqa: E402
from src.q3.paths import FIG_DIR, PROCESSED_DIR  # noqa: E402
from src.q3.plotting import GEN_LABELS  # noqa: E402
from src.q3.tables import to_tex  # noqa: E402

LABELS = {
    "auc_marg": "合并 C2ST AUC",
    "auc_cond": "条件 C2ST AUC",
    "dcr_ratio": "DCR 比",
    "near_copy": "近复制率",
    "trts": "TRTS",
    "tstr": "TSTR",
    "floor_err": "下限比例误差",
    "floor_row": "每款下限列数",
    "corr_err": "相关误差",
    "wcorr_err": "风格内相关误差",
    "module_rho": "模块内 ρ",
}


def main():
    with open(PROCESSED_DIR / "q3_genval.json", encoding="utf-8") as f:
        js = json.load(f)
    m = js["metrics"]
    rows = []
    for k in SCALARS:
        r = {"指标": LABELS[k]}
        for g in GEN_NAMES:
            e = m[f"{g}.{k}"]
            r[GEN_LABELS[g]] = f"{e['estimate']:.3f} [{e['ci95']['lo']:.3f}, {e['ci95']['hi']:.3f}]"
        rows.append(r)
    real = {"指标": "真实参照"}
    real[GEN_LABELS["main"]] = (
        f"TRTR {m['real.trtr']['estimate']:.3f}；"
        f"模块内 ρ {m['real.real_module_rho']['estimate']:.3f}；"
        f"每款下限列数 {m['real.real_floor_row']['estimate']:.1f}；真实对真实 AUC "
        f"{m['real.rr_auc_marg']['estimate']:.3f}/{m['real.rr_auc_cond']['estimate']:.3f}"
    )
    rows.append(real)
    df = pd.DataFrame(rows).fillna("")
    acc = js["acceptance"]
    basis = js["acceptance_basis"]
    df.to_csv(FIG_DIR / "tab_q3_genval.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"项": k, "通过": v} for k, v in acc.items()]).to_csv(
        FIG_DIR / "tab_q3_genval_acceptance.csv", index=False, encoding="utf-8-sig"
    )
    note = (
        "每格为 5 次重复均值 [按啤酒 2,000 次重抽 95\\% 区间]，区间条件于已拟合模型与划分。"
        "G1、G3 生成池含近重复样本，其 AUC 与 TRTS 不可比。验收："
        + "；".join(f"({k}) {'通过' if v else '未通过'}" for k, v in acc.items())
        + f"。主模型合并/条件 AUC {basis['main']['auc_marg']:.3f}/{basis['main']['auc_cond']:.3f}。"
    )
    (FIG_DIR / "tab_q3_genval.tex").write_text(
        to_tex(df, "生成器验证指标（方法论 §6.2）", "tab:q3_genval", note), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
