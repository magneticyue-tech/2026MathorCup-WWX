# code/src/q2/rationality.py
"""Q2 合理性层：预期表匹配、命中率计算、最佳预期化合物排名与 D/M/S 分类。"""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd

import config as C
from data import load_q2_dataset

# ==============================================================================
# 1. 预期表定义 (基于 q2_methodology.md §6.1)
# 格式: { 维度名: {"D": [直接化合物...], "M": [机理化合物...]} }
# 注意：列名必须和 1.csv 完全一致，比如 isopentyl_acetate 不是 isoamyl_acetate
# ==============================================================================
EXPECTED_TABLE = {
    "A_malt_all": {"D": [], "M": ["color.EBC."]},
    "A_malt_grain": {"D": ["X3.methylbutanal"], "M": []},
    "A_malt_bread": {"D": ["X3.methylbutanal", "X2.acetylpyrrole", "isomaltol"], "M": []},
    "A_malt_cara": {"D": ["isomaltol", "furfural", "X5.methylfurfural", "hydroxy.methyl.furfural", "X2.acetylfuran"], "M": ["color.EBC."]},
    "A_malt_burn": {"D": ["methylpyrazine", "X2.6.dimethyl.pyrazine", "X2.ethyl.3.methylpyrazine", "X2.furanmethanol"], "M": ["color.EBC."]},
    "A_hops_all": {"D": ["linalool", "geraniol", "alpha.humulene", "E.caryophyllene"], "M": []},
    "A_hops_citrus": {"D": ["linalool", "geraniol", "citronellol", "limonene", "nerol"], "M": []},
    "A_hops_tropical": {"D": ["geraniol", "linalool", "ethyl_2.methyl_butyrate"], "M": []},
    "A_hops_noble": {"D": ["alpha.humulene", "E.caryophyllene", "humulene.epoxide.II", "caryophyllene.oxide", "X.E..beta.farnesene", "linalool"], "M": []},
    "A_hops_woody": {"D": ["alpha.humulene", "E.caryophyllene", "humulene.epoxide.II", "caryophyllene.oxide", "germacrene.D", "alpha.calacorene", "cadalene", "tau.cadinol"], "M": []},
    "A_esters_all": {"D": ["isopentyl_acetate", "ethyl_acetate", "ethyl_hexanoate", "ethyl_octanoate", "phenethyl_acetate", "isobutyl_acetate"], "M": []},
    "A_esters_fruity": {"D": ["ethyl_hexanoate", "ethyl_octanoate", "ethyl.butyrate", "ethyl_decanoate", "isopentyl_acetate"], "M": []},
    "A_esters_flower": {"D": ["phenethyl_acetate", "phenethyl_alcohol", "geraniol", "linalool"], "M": []},
    "A_esters_isoaa": {"D": ["isopentyl_acetate"], "M": ["isoamyl_alcohol"]},
    "A_esters_ethac": {"D": ["ethyl_acetate"], "M": ["ethanol..v.v.", "acetic_acid.g.L."]},
    "F_malt_all": {"D": [], "M": ["color.EBC."]},
    "F_malt_grain": {"D": ["X3.methylbutanal"], "M": []},
    "F_malt_bread": {"D": ["X3.methylbutanal", "X2.acetylpyrrole", "isomaltol"], "M": []},
    "F_malt_cara": {"D": ["isomaltol", "furfural", "X5.methylfurfural", "hydroxy.methyl.furfural", "X2.acetylfuran"], "M": ["color.EBC."]},
    "F_malt_burn": {"D": ["methylpyrazine", "X2.6.dimethyl.pyrazine", "X2.ethyl.3.methylpyrazine", "X2.furanmethanol"], "M": ["color.EBC."]},
    "F_hops_all": {"D": ["linalool", "geraniol", "alpha.humulene", "E.caryophyllene"], "M": ["bitterness"]},
    "F_hops_citrus": {"D": ["linalool", "geraniol", "citronellol", "limonene", "nerol"], "M": []},
    "F_hops_tropical": {"D": ["geraniol", "linalool", "ethyl_2.methyl_butyrate"], "M": []},
    "F_hops_noble": {"D": ["alpha.humulene", "E.caryophyllene", "humulene.epoxide.II", "caryophyllene.oxide", "X.E..beta.farnesene", "linalool"], "M": []},
    "F_hops_woody": {"D": ["alpha.humulene", "E.caryophyllene", "humulene.epoxide.II", "caryophyllene.oxide", "germacrene.D", "alpha.calacorene", "cadalene", "tau.cadinol"], "M": []},
    "F_esters_all": {"D": ["isopentyl_acetate", "ethyl_acetate", "ethyl_hexanoate", "ethyl_octanoate", "phenethyl_acetate", "isobutyl_acetate"], "M": []},
    "F_esters_fruity": {"D": ["ethyl_hexanoate", "ethyl_octanoate", "ethyl.butyrate", "ethyl_decanoate", "isopentyl_acetate"], "M": []},
    "F_esters_flower": {"D": ["phenethyl_acetate", "phenethyl_alcohol", "geraniol", "linalool"], "M": []},
    "F_esters_isoaa": {"D": ["isopentyl_acetate"], "M": ["isoamyl_alcohol"]},
    "F_esters_ethac": {"D": ["ethyl_acetate"], "M": ["ethanol..v.v.", "acetic_acid.g.L."]},
    "dms": {"D": ["DMS"], "M": []},
    "metallic": {"D": ["iron"], "M": []},
    "X4vg": {"D": ["X4VG"], "M": []},
    "t2n": {"D": ["X.E..2.nonenal", "X.E.E..2.4.decadienal"], "M": []},
    "diacetyl": {"D": ["X2.3.butanedione"], "M": []},
    "stale_hops": {"D": ["humulene.epoxide.II", "caryophyllene.oxide"], "M": []},
    "orange": {"D": ["limonene", "valencene", "linalool"], "M": []},
    "coriander": {"D": ["linalool", "camphor", "geraniol"], "M": []},
    "clove": {"D": ["eugenol", "X4VG", "E.methyl.isoeugenol", "E.caryophyllene"], "M": []},
    "lactic": {"D": ["lactic_acid.mg.L."], "M": ["ethyl.lactate", "pH"]},
    "acetic": {"D": ["acetic_acid.g.L."], "M": ["ethyl_acetate", "pH"]},
    "body": {"D": [], "M": ["ethanol..v.v.", "glycerol", "GFS", "unfermentables", "protein.g.L.", "betaglucan.mg.L."]},
    "alcohol": {"D": ["ethanol..v.v.", "isoamyl_alcohol", "isobutanol", "X1.propanol", "X2.methyl.1.butanol"], "M": []},
    "bitternes": {"D": ["bitterness"], "M": []},
    "sweetness": {"D": ["GFS"], "M": ["unfermentables", "glycerol"]},
    "co2": {"D": ["CO2.PSI."], "M": []},
    "acidity": {"D": ["pH", "lactic_acid.mg.L.", "acetic_acid.g.L."], "M": []}
    # aftertaste, overall, barnyard (同 stale_hops) 在计算中特殊处理
}

def run_rationality():
    print("="*50)
    print("开始化学合理性验证")
    print("="*50)

    ds = load_q2_dataset()
    df_final = pd.read_csv(C.Q2_DIR / "final_top3_with_stability.csv")
    df_nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    
    # 映射模块
    node_to_module = dict(zip(df_nodes["node"], df_nodes["module_id"]))
    module_to_nodes = {}
    for node, m_id in node_to_module.items():
        module_to_nodes.setdefault(m_id, []).append(node)

    # 统计每个化合物进入 Top-3 的维度数（用于 S 类判定）
    compound_dim_counts = df_final["compound"].value_counts().to_dict()

    # 边际相关的排名计算（针对未命中）
    X_mat = ds.X_rank.values
    n = len(ds.sample_ids)
    marginal_ranks = {}

    for dim_idx, dim_name in enumerate(ds.Y_rank.columns):
        y = ds.Y_rank.values[:, dim_idx]
        r_all = np.abs((X_mat.T @ y) / n)
        # 排序得到 225 个候选的排名 (1 是最高)
        sorted_idx = np.argsort(-r_all)
        ranks = {ds.candidate_cols[idx]: rank+1 for rank, idx in enumerate(sorted_idx)}
        marginal_ranks[dim_name] = ranks

    # 开始逐行判定
    print("正在匹配预期表并分类...")
    results = []
    
    for _, row in df_final.iterrows():
        dim = row["dimension"]
        comp = row["compound"]
        m_id = row["module_id"]
        
        # 处理 barnyard 别名
        target_dim = "stale_hops" if dim == "barnyard" else dim
        
        exp = EXPECTED_TABLE.get(target_dim, {"D": [], "M": []})
        
        is_strict_hit = False
        is_module_hit = False
        category = "U" # 默认未解释
        
        # 1. 严格命中判定 (D 或 M)
        if comp in exp["D"]:
            is_strict_hit = True
            category = "D"
        elif comp in exp["M"]:
            is_strict_hit = True
            category = "M"
        else:
            # 2. 模块命中判定
            for c_d in exp["D"]:
                if node_to_module.get(c_d) == m_id:
                    is_module_hit = True
                    category = "D"
                    break
            if not is_module_hit:
                for c_m in exp["M"]:
                    if node_to_module.get(c_m) == m_id:
                        is_module_hit = True
                        category = "M"
                        break
                        
        # 3. 如果都不是，判定为 S（共变标记）或 X（非酿造）或 U（未解释）
        if category == "U":
            # X 类：非酿造来源
            x_list = ["butylated.hydroxytoluene", "X2.ethylhexylsalicylate", "isopropyl.laurate", "isopropyl.tetradecanoate", "isopropyl.hexadecanoate"]
            if comp in x_list:
                category = "X"
            # S 类：进入 >=5 个维度 Top3
            elif compound_dim_counts.get(comp, 0) >= 5:
                category = "S"
                
        results.append({
            "dimension": dim,
            "rank": row["rank"],
            "compound": comp,
            "module_id": m_id,
            "pi_c": row["pi_c"],
            "stability": row["stability"],
            "category": category,
            "strict_hit": is_strict_hit,
            "module_hit": is_module_hit,
            "exp_D": ",".join(exp["D"]),
            "exp_M": ",".join(exp["M"])
        })

    df_eval = pd.DataFrame(results)

    # 计算总体命中率
    # 对于每个维度，只要 Top-3 里有至少一个严格命中，就算该维度严格命中
    dim_strict_hits = df_eval.groupby("dimension")["strict_hit"].any()
    dim_module_hits = df_eval.groupby("dimension")["module_hit"].any()

    valid_dims = [d for d in dim_strict_hits.index if d not in C.NO_PRIOR_DIMS and d != "barnyard"]
    n_valid = len(valid_dims)

    strict_rate = dim_strict_hits[valid_dims].sum() / n_valid
    module_rate = dim_module_hits[valid_dims].sum() / n_valid

    print(f"\n✅ 合理性验证完成！")
    print(f"有效计算维度数: {n_valid}")
    print(f"严格命中率: {strict_rate:.3f} (期望参考: 0.47)")
    print(f"模块命中率: {module_rate:.3f} (期望参考: 0.57)")

    # 找最佳预期化合物排名
    print("\n【已知未命中检查】(最佳预期化合物在 225 个候选中的排名):")
    for dim in ["A_malt_grain", "dms", "diacetyl", "stale_hops"]:
        if dim in EXPECTED_TABLE:
            # 找出预期表里所有的 D 和 M
            all_exp = EXPECTED_TABLE[dim]["D"] + EXPECTED_TABLE[dim]["M"]
            if not all_exp:
                continue
            
            # 找这些化合物里边际排名最靠前的
            best_rank = 999
            best_comp = None
            for c in all_exp:
                if c in marginal_ranks.get(dim, {}):
                    r = marginal_ranks[dim][c]
                    if r < best_rank:
                        best_rank = r
                        best_comp = c
            
            if best_comp:
                print(f"  - {dim:<15}: 最佳预期化合物 {best_comp:<20} 排名 {best_rank}")

    # 保存输出
    out_file = C.Q2_DIR / "final_top3_with_rationality.csv"
    df_eval.to_csv(out_file, index=False)
    print(f"\n最终结果已保存至: {out_file}")
    
    print("\n【预览】部分维度的分类结果:")
    for dim in ["A_malt_all", "lactic", "acetic", "t2n"]:
        sub = df_eval[df_eval["dimension"] == dim].sort_values("rank")
        if len(sub) == 0: continue
        print(f"\n--- {dim} ---")
        for _, r in sub.iterrows():
            print(f"Top {r['rank']}: {r['compound']:<25} 类别={r['category']}")

if __name__ == "__main__":
    run_rationality()