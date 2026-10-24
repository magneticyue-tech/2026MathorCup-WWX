---
name: composite-columns-near-duplicates
description: 5 个综合指标各自近似某一个组成成分的副本，不作为化合物归因候选
type: decision
---

附件 1 的 5 个综合指标各自几乎就是某一个成分的副本：`esters_sum`≈isoamyl acetate（|ρ| 0.92），`aroma_hops_sum`≈linalool（0.96），`acids_sum`≈acetic acid（0.97），`sulfur_sum`≈H₂S（0.78），`kcalperc`≈ethanol（0.85）。

**依据：** 附件 2 的 Spearman 系数（`src/q2_methodology_checks.py` 检查 1）；公式和组成清单均未提供（审计 §4.3）。

**如何应用：** 问题 2 不把它们列为 Top-3 候选，只作图上的注释节点，敏感性运行再加回；问题 3 由成分推导综合指标，不单独生成。

相关：[[internal-standard-3-methyl-2-butenal]]
