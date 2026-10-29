# code/src/q2/acceptance.py
"""Q2 最终验收（结构化读取 CSV）、附录 147 行数据视图与向 Modeling 的移交清单。"""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parents[1]
for _p in (CURRENT_DIR, CODE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

import config as C
import core
from check_attachment6 import check_att6
from data import input_checks, load_q2_dataset
from labels import attachment8_labels, check_against_dictionary
from src.q1.data import DIM_GROUPS

Q = C.Q2_DIR
FIGS = ["network_graph.png", "method_comparison.png", "stability_pi_c_hist.png", "module_sizes.png",
        "top1_heatmap.png"]


def _ok(cond, msg, log):
    log.append(("通过" if cond else "失败", msg))
    if not cond:
        raise AssertionError(msg)


def appendix_view(fin, ds):
    main = fin[fin["method"] == "main"].copy()
    order = []
    for g, dims in DIM_GROUPS.items():
        for d in sorted([d for d in dims if d in ds.delivery_sensory_cols], key=ds.sensory_cols.index):
            order.append((g, d))
    grp = {d: g for g, d in order}
    pos = {d: i for i, (_, d) in enumerate(order)}
    main["group"] = main["dimension"].map(grp)
    main["_o"] = main["dimension"].map(pos)
    view = main.sort_values(["_o", "rank"])[
        ["group", "dimension", "rank", "compound", "sign", "pi_c", "stability", "grade"]]
    return view.reset_index(drop=True)


def handoff_md(fin, full, sens):
    main = fin[fin["method"] == "main"].copy()
    main["is_significant"] = main["is_significant"].astype(bool)
    r1 = main[main["rank"] == 1]
    L = ["# Q2 结果移交（Coding → Modeling）", "",
         "数值来源：`code/data/processed/q2/final_top3_with_rationality.csv`。`interpretation` 列留空，由 Modeling 撰写。", ""]
    for g in "ABC":
        dims = r1.loc[r1["grade"] == g, "dimension"].tolist()
        L.append(f"- {g} 级 {len(dims)} 个：{'、'.join(dims) if dims else '无'}")
    L += ["", "## 主方法每维 Top-3", "", "| 维度 | 等级 | 第 1 位 | 第 2 位 | 第 3 位 |", "|---|---|---|---|---|"]
    for dim, g in main.groupby("dimension", sort=False):
        g = g.sort_values("rank")
        cells = [f"{r.compound}（{r.sign}，π^c={r.pi_c:.2f}，{'显著' if r.is_significant else '不显著'}）"
                 for r in g.itertuples()]
        L.append(f"| {dim} | {g['grade'].iloc[0]} | " + " | ".join(cells) + " |")
    bad = main[(~main["is_significant"]) | (main["stability"] == "不稳定")]
    L += ["", f"## 不显著或低稳定项（{len(bad)} 项）", ""]
    L += [f"- {r.dimension} 第 {r.rank} 位 {r.compound}：{'显著' if r.is_significant else '不显著'}，π^c={r.pi_c:.3f}（{r.stability}）"
          for r in bad.itertuples()]
    diff = full[~full["M3_top1_match"]]
    L += ["", f"## 主方法与 M3 Top-1 不同（{len(diff)} 维）", ""]
    L += [f"- {r.dimension}：主方法 {r.main_top3.split('|')[0]}，M3 {r.M3_top3.split('|')[0]}" for r in diff.itertuples()]
    cov = main[main["covariation"] == True]  # noqa: E712
    L += ["", f"## 共变标记（{len(cov)} 项）", ""]
    for r in cov.itertuples():
        why = [n for n, f in (("风格内 CI 含 0", r.flag_style_ci0), ("≥5 维", r.flag_breadth5),
                              ("全部啤酒 OAV<1", r.flag_oav_lt1)) if f == True]  # noqa: E712
        L.append(f"- {r.dimension} 第 {r.rank} 位 {r.compound}：{'；'.join(why)}")
    thr = main[main["oav_ge_1_ratio"].notna()].drop_duplicates("compound")
    L += ["", f"## 阈值证据（有阈值且已校准的 Top-3 化合物 {len(thr)} 个，OAV≥1 啤酒比例）", ""]
    L += [f"- {r.compound}：{r.oav_ge_1_ratio:.2f}" for r in thr.sort_values("oav_ge_1_ratio", ascending=False).itertuples()]
    L += ["", "## 同模块成员与模块可分检验", ""]
    for r in main[main["same_module_compounds"].fillna("") != ""].itertuples():
        extra = f"；超过 q_d 的成员：{r.module_split_members}" if isinstance(r.module_split_members, str) and r.module_split_members else ""
        L.append(f"- {r.dimension} 第 {r.rank} 位 {r.compound}：{r.same_module_compounds}{extra}")
    L += ["", "## 标注", "", f"- stale_hops 行 `note` 列：{C.STALE_HOPS_NOTE}；barnyard 不单独建模。", "",
          "## 未解释项", "", "- 全部 147 项的化学解释待 Modeling 撰写（Coding 不生成机理解释）。", "",
          "## 敏感性汇总", "", "数值见 `code/data/processed/q2/sensitivity/sensitivity_summary.csv`。", "", "```",
          sens.drop(columns=["description", "not_applicable", "grade_changed_dims"]).round(3).to_string(index=False),
          "```", ""]
    for r in sens.itertuples():
        if isinstance(r.grade_changed_dims, str) and r.grade_changed_dims:
            L.append(f"- {r.arm} 等级改变：{r.grade_changed_dims}")
        if isinstance(r.not_applicable, str) and r.not_applicable:
            L.append(f"- {r.arm}：{r.not_applicable}")
    return "\n".join(L) + "\n"


def main():
    log = []
    ds = load_q2_dataset()
    for f in ("top3_results.csv", "top3_results_with_sig.csv", "final_top3_with_stability.csv"):
        df = pd.read_csv(Q / f)
        core.check_147(df, f)
        log.append(("通过", f"{f}：147 行、49 维、每维 3 行、无 barnyard"))
    fin = pd.read_csv(Q / "final_top3_with_rationality.csv")
    full = pd.read_csv(Q / "full_controls_results.csv")
    main = fin[fin["method"] == "main"]
    core.check_147(main, "最终表 main")
    m3 = fin[fin["method"] == "M3"]
    n_diff = int((~full["M3_top1_match"]).sum())
    _ok(len(m3) == n_diff and set(m3["dimension"]) == set(full.loc[~full["M3_top1_match"], "dimension"]),
        f"M3 行 {len(m3)} = M3 与主方法 Top-1 不同的维度数 {n_diff}", log)
    _ok(m3["pi_c"].isna().all() and m3["grade"].isna().all() and m3["not_applicable"].notna().all(),
        "M3 行的主方法统计量为缺失并标注不适用", log)
    _ok(set(fin["method"]) <= {"main", "M3"}, "最终表 method 只有 main/M3", log)

    nodes, edges = pd.read_csv(Q / "graph_nodes.csv"), pd.read_csv(Q / "graph_edges.csv")
    n_cc = int((edges["edge_type"] == "candidate-candidate").sum())
    _ok(len(nodes) == 231 and n_cc == 245,
        f"图节点 {len(nodes)}，候选间边 {n_cc}，注释节点相关边 {int((edges['edge_type'] == 'annotation').sum())}", log)
    chk = input_checks()
    _ok(chk["xbar_z_median_spearman_by_row"] >= 0.95 and chk["xbar_z_median_spearman_by_beer"] <= 0.1,
        f"感官目标为 xbar 的秩；xbar 与 z 逐维 Spearman 中位数按行号 {chk['xbar_z_median_spearman_by_row']:.3f}，"
        f"按 beer 列 {chk['xbar_z_median_spearman_by_beer']:.3f}", log)
    sizes = chk["date_group_sizes"]
    _ok(chk["n_date_groups"] == C.N_DATE_GROUPS_EXPECTED and len(np.unique(ds.date_group)) == C.N_DATE_GROUPS_EXPECTED,
        f"日期组 {chk['n_date_groups']} 个，每组 {min(sizes.values())}–{max(sizes.values())} 款", log)
    _ok(np.allclose(ds.Y_rank.to_numpy(), core.zrank(ds.xbar[ds.delivery_sensory_cols].to_numpy())[0]),
        "主感官目标 Y_rank 与 xbar 的标准化秩逐元素一致", log)
    stab = pd.read_csv(Q / "final_top3_with_stability.csv")
    sig = pd.read_csv(Q / "top3_results_with_sig.csv")
    _ok((stab["n_bootstrap"] == C.N_BOOTSTRAP).all() and (sig["n_permutation"] == C.N_PERMUTATION).all(),
        f"主 bootstrap {C.N_BOOTSTRAP} 次、置换 {C.N_PERMUTATION} 次", log)
    boot = np.load(Q / "q2_bootstrap_main.npz")
    _ok(boot["boot_sel"].shape[0] == C.N_BOOTSTRAP, "bootstrap 结果数组含 1000 次重复", log)
    nd = boot["n_distinct_groups"]
    _ok(str(boot["boot_unit"]) == "date_group" and int(boot["n_groups_drawn"]) == C.N_DATE_GROUPS_EXPECTED
        and len(nd) == C.N_BOOTSTRAP,
        f"主 bootstrap 每次抽取 {int(boot['n_groups_drawn'])} 个日期组；抽到的不同日期组数 "
        f"{nd.min()}–{nd.max()}（中位 {np.median(nd):g}）", log)
    zi = pd.read_csv(Q / "zero_inflation_c_grade.csv")
    log.append(("记录", "零膨胀检查（C 级维度）：" + "；".join(
        f"{r.dimension} 零值占比 {r.zero_share:.3f}{'，已重算' if r.eligible else '，不适用'}" for r in zi.itertuples())
        if len(zi) else "零膨胀检查：无 C 级维度"))
    run_log = (Q / "run_log.txt").read_text(encoding="utf-8") if (Q / "run_log.txt").exists() else ""
    banned = ("q1_panels", "品评团", "legacy_z")
    _ok(not any(b in run_log for b in banned),
        "运行日志不含 q1_panels、推断团或 legacy_z", log)
    sim = pd.read_csv(Q / "simulation_recovery_results.csv")
    _ok(len(sim) == 6 and (sim["n_sim"] == C.N_SIMULATION).all(), "模拟 6 种设置各 1000 次", log)
    acc = check_att6(verbose=False)
    _ok(acc["attachment2_max_error"] <= 1e-12 and acc["attachment6_max_error"] <= 1e-12,
        f"附件 2 误差 {acc['attachment2_max_error']:.2e}，附件 6 误差 {acc['attachment6_max_error']:.2e}", log)

    r1 = main[main["rank"] == 1]
    regrade = [core.grade_from(bool(s), float(p)) for s, p in zip(r1["is_significant"], r1["pi_c"])]
    _ok(list(r1["grade"]) == regrade, "A/B/C 由 is_significant、π^c 独立复算一致", log)
    sig_re = (stab["score"].abs() > stab["q_d"]) & ((stab["r_ci_lower"] > 0) | (stab["r_ci_upper"] < 0))
    _ok((sig_re == stab["is_significant"]).all(), "is_significant = 超过门槛 且 CI 不含 0", log)
    abc = r1["grade"].value_counts().reindex(list("ABC")).fillna(0).astype(int).to_dict()
    log.append(("记录", f"等级 {abc}"))

    lab = attachment8_labels(ds.chem_cols)
    n_lab = check_against_dictionary(lab)
    for f in ("origin_paper", "off_flavors", "aging", "threshold_mean"):
        a = main[f]
        b = main["compound"].map(lab[f])
        same = (a.isna() & b.isna()) | (a.astype(str) == b.astype(str))
        _ok(bool(same.all()), f"最终表 {f} 与附件 8 逐项一致", log)
    log.append(("通过", f"附件 8 标签与 chemical_dictionary.csv 一致（{n_lab} 单元）"))
    _ok(main.loc[main["threshold_mean"].isna(), "oav_ge_1_ratio"].isna().all(), "无阈值项的 OAV 保持缺失", log)
    _ok(main["flag_style_ci_ordered"].fillna(True).all(), "风格内 CI 有序且在 [-1, 1]", log)
    _ok((main["interpretation"].fillna("") == "").all(), "interpretation 留空", log)
    _ok(set(main.loc[main["note"].fillna("") != "", "dimension"]) == {"stale_hops"}, "stale_hops 标注同 barnyard", log)

    sens = pd.read_csv(C.SENS_DIR / "sensitivity_summary.csv")
    arms = sens[sens["arm"] != "pairwise_deletion"]
    _ok(len(sens) == 11 and (sens["n_dims"] == 49).all(), f"敏感性 {len(sens) - 1} 个方案（含逐对删除）各 49 维", log)
    _ok((arms["n_permutation"] == C.N_PERMUTATION).all() and (arms["n_bootstrap"] == C.N_BOOTSTRAP).all(),
        "敏感性方案置换、bootstrap 均 1000 次", log)
    pw = sens[sens["arm"] == "pairwise_deletion"]
    _ok(pw["not_applicable"].str.contains("不适用").all() and pw["mean_jaccard_compound"].isna().all(),
        "逐对删除的选择指标标为不适用", log)
    text = sens.astype(str).to_string()
    _ok("PENDING" not in text and "PLACEHOLDER" not in text, "敏感性汇总无占位符", log)

    newest = max((Q / f).stat().st_mtime for f in ("final_top3_with_rationality.csv", "graph_nodes.csv"))
    for f in FIGS:
        p = C.FIGURES / f
        _ok(p.exists() and p.stat().st_mtime >= newest, f"图 {f} 基于本次结果重绘", log)

    view = appendix_view(fin, ds)
    _ok(len(view) == 147 and view.groupby("group")["dimension"].nunique().sum() == 49,
        f"附录视图 147 行，六组 {view.groupby('group')['dimension'].nunique().to_dict()}", log)
    view.to_csv(Q / "appendix_main_147.csv", index=False)
    (Q / "handoff_modeling.md").write_text(handoff_md(fin, full, sens), encoding="utf-8")
    pd.DataFrame(log, columns=["status", "check"]).to_csv(Q / "acceptance_log.csv", index=False)
    for s, m in log:
        print(f"[{s}] {m}")


if __name__ == "__main__":
    main()
