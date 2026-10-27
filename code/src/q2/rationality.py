# code/src/q2/rationality.py
"""Q2 合理性层：49维输出，Fisher CI 真实计算与验证。"""
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path: 
    sys.path.insert(0, str(CURRENT_DIR))

import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
import config as C
from data import load_q2_dataset


def _norm_name(name: str) -> str:
    n = str(name).lower()
    n = re.sub(r"\.v\.v\.|\.g\.l\.|\.mg\.l\.|\.psi\.", "", n)
    return re.sub(r"[^a-z0-9]+", "", n)


def get_thresholds():
    try:
        df_thresh = pd.read_excel(C.THRESHOLD_PATH, sheet_name="Sensory thresholds")
        name_col = [c for c in df_thresh.columns if "name" in str(c).lower() or "compound" in str(c).lower()][0]
        thresh_col = [c for c in df_thresh.columns if "threshold_mean" in str(c).lower()][0]
        thresh_dict = {}
        for _, row in df_thresh.iterrows():
            raw_name = str(row[name_col]).strip()
            val = pd.to_numeric(row[thresh_col], errors="coerce")
            if pd.notna(val) and val > 0: 
                thresh_dict[_norm_name(raw_name)] = val
        return thresh_dict
    except Exception: 
        return {}


def attachment8_labels(compounds: list[str]) -> pd.DataFrame:
    if not C.CHEMICAL_DICT_PATH.exists(): 
        return pd.DataFrame({"compound": compounds})
    df_dict = pd.read_csv(C.CHEMICAL_DICT_PATH).rename(columns={"csv1_field": "compound"})
    cols = ["compound", "chemical_name", "origin_paper", "off_flavors", "aging", "transform_interpretation"]
    return df_dict[[c for c in cols if c in df_dict.columns]].set_index("compound").reindex(compounds).reset_index()


def calc_style_ci(X_resid, Y_resid, top_compounds):
    n = len(X_resid)
    ci_dict = {}
    for dim, comp in top_compounds:
        if dim not in Y_resid.columns or comp not in X_resid.columns:
            ci_dict[(dim, comp)] = np.nan
            continue
        x, y = X_resid[comp].values, Y_resid[dim].values
        r = np.corrcoef(x, y)[0, 1]
        if np.isnan(r) or abs(r) >= 1.0:
            ci_dict[(dim, comp)] = np.nan
            continue
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = norm.ppf(0.975)
        lower = np.tanh(z - z_crit * se)
        upper = np.tanh(z + z_crit * se)
        # 返回是否包含 0（True 表示包含 0，即风格内不显著）
        ci_dict[(dim, comp)] = bool(lower <= 0 <= upper)
    return ci_dict


def run_rationality():
    print("生成合理性证据表 (49维, 恢复 Fisher CI)...")
    ds = load_q2_dataset()
    
    df_base = pd.read_csv(C.Q2_DIR / "final_top3_with_stability.csv")
    df_base = df_base[df_base["dimension"] != "barnyard"].copy()
    assert len(df_base) == 147, f"主方法必须是 147 行 (49x3)，实际 {len(df_base)} 行"

    comps = df_base["compound"].unique().tolist()
    df_base = df_base.merge(attachment8_labels(comps), on="compound", how="left")
    
    thresh_dict = get_thresholds()
    trans_map = dict(zip(df_base["compound"], df_base["transform_interpretation"])) if "transform_interpretation" in df_base.columns else {}
    
    oav_ratios, thresholds = {}, {}
    for c in comps:
        norm_c = _norm_name(c)
        if norm_c in thresh_dict and c in ds.chem_raw.columns:
            thresh = thresh_dict[norm_c]
            thresholds[c] = thresh
            conc = ds.chem_raw[c]
            if "log10" in str(trans_map.get(c, "")).lower(): 
                conc = 10 ** conc
            oav_ratios[c] = ((conc / thresh) >= 1).mean()
        else: 
            thresholds[c], oav_ratios[c] = np.nan, np.nan

    df_base["threshold_mean"] = df_base["compound"].map(thresholds)
    df_base["oav_ge_1_ratio"] = df_base["compound"].map(oav_ratios)

    # 风格内 CI 计算
    chem = ds.chem_raw
    if 'tasting_category_fine' in chem.columns:
        # 【修复】：使用 drop_first=True 避免完全共线
        dummies = pd.get_dummies(chem['tasting_category_fine'], drop_first=True).values
        style_dummies = np.column_stack([np.ones(len(chem)), dummies])
        
        X_mat_raw, Y_mat_raw = ds.X_rank.values, ds.Y_rank.values
        beta_X = np.linalg.lstsq(style_dummies, X_mat_raw, rcond=None)[0]
        beta_Y = np.linalg.lstsq(style_dummies, Y_mat_raw, rcond=None)[0]
        
        df_X_resid = pd.DataFrame(X_mat_raw - style_dummies @ beta_X, columns=ds.candidate_cols)
        df_Y_resid = pd.DataFrame(Y_mat_raw - style_dummies @ beta_Y, columns=ds.Y_rank.columns)
        
        top_pairs = list(zip(df_base["dimension"], df_base["compound"]))
        ci_dict = calc_style_ci(df_X_resid, df_Y_resid, top_pairs)
        df_base["flag_style_ci0"] = [ci_dict.get((d, c), np.nan) for d, c in top_pairs]
    else:
        df_base["flag_style_ci0"] = np.nan

    # breadth 计算（严格只算 main）
    counts = df_base["compound"].value_counts()
    df_base["n_dims_top3"] = df_base["compound"].map(counts)
    
    df_base["flag_breadth5"] = df_base["n_dims_top3"] >= 5
    df_base["flag_oav_lt1"] = df_base["oav_ge_1_ratio"].notna() & (df_base["oav_ge_1_ratio"] < 1)
    
    # 生成 covariation：三者逻辑或
    df_base["covariation"] = df_base["flag_breadth5"] | df_base["flag_oav_lt1"]
    df_base.loc[df_base["flag_style_ci0"] == True, "covariation"] = True

    # 等级计算
    def get_grade(row):
        if row["rank"] == 1:
            if not row["is_significant"]: return "C"
            return "A" if row["pi_c"] >= 0.6 else "B"
        return ""
        
    df_base["grade"] = df_base.apply(get_grade, axis=1)
    df_base["grade"] = df_base["dimension"].map(df_base[df_base["rank"] == 1].set_index("dimension")["grade"].to_dict())
    
    df_base["sign"] = np.where(df_base["score"] >= 0, "+", "-")
    df_base["interpretation"] = ""
    df_base["method"] = "main"
    
    # 严格标注
    df_base.loc[df_base["dimension"] == "stale_hops", "interpretation"] = "同 barnyard"

    # ==========================================
    # 缺失的 M0~M3 标识列及 M3 追加行的占位（等待 full_controls.py 接入）
    # ==========================================
    # 如果 full_controls 已经生成了对应结果，应该在这里做合并，例如：
    # for m in ["M0", "M1", "M2", "M3"]:
    #     df_ctrl = pd.read_csv(C.Q2_DIR / f"{m}_top3.csv")
    #     # 构建标识列
    #     df_base[f"in_{m}"] = df_base["compound"].isin(df_ctrl["compound"])
    #
    # # 追加 M3 中主方法未选中的 Top-1
    # df_m3_top1 = pd.read_csv(C.Q2_DIR / "M3_top3.csv")
    # df_m3_top1 = df_m3_top1[df_m3_top1["rank"] == 1]
    # merged = df_base[df_base["rank"] == 1].merge(df_m3_top1, on="dimension", suffixes=("_main", "_M3"))
    # m3_diff = merged[merged["compound_main"] != merged["compound_M3"]]
    # ... 写入额外行 ...
    
    out_path = C.Q2_DIR / "final_top3_with_rationality.csv"
    df_base.to_csv(out_path, index=False)
    print(f"✅ 合理性证据表已生成！共 {len(df_base)} 行 (49x3，剔除了 barnyard 独立行)。")

if __name__ == "__main__":
    run_rationality()