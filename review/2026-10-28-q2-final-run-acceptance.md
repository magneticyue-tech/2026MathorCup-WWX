---
id: 2026-10-28-q2-final-run-acceptance
from: modeling
to: coding
status: open
severity: major
refs:
  - review/2026-10-28-q2-complete-run.md
  - review/2026-10-28-q2-statistical-fixes.md
  - review/2026-10-28-q2-sensitivity-runs.md
  - model/q2_methodology.md
  - code/src/q2/config.py
  - code/data/processed/q2/
  - figures/q2/
---

## 前置条件

只有在 `q2-statistical-fixes` 通过其小样本验收，且 `q2-sensitivity-runs` 的 NA 决策已由 Modeling 确认、所有敏感性代码通过小样本验收后，才开始本工单。开始前重新读取当前方法文档和 Q2 活文档，检查 `git status`、原始附件、Q1 panel/参数产物和全部 Q2 输出路径。

## 正式运行顺序

使用仓库根目录作为工作目录；逐条运行并检查返回码。不得在一条命令中串行吞掉中间错误。

1. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/data.py`
2. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/graph.py`
3. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/association.py`
4. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/inference.py`
5. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/stability.py`
6. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/rationality.py`
7. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/controls_sensitivity.py`
8. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/full_controls.py`
9. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/simulation.py`
10. 运行新增的全部正式敏感性方案入口。
11. `$env:PYTHONIOENCODING='utf-8'; python code/src/q2/check_attachment6.py`
12. 运行修复后的 `advanced_audits.py` 检查及 Q2 图表入口，确认图表基于本次结果重绘。

如果代码入口名称因实现调整而变更，先更新本清单和运行记录，再执行。任何一步失败即停止其下游步骤，修复后从依赖该结果的最早步骤重跑。

## 最终验收

使用脚本或结构化读取 CSV 验收，不靠肉眼抽查：

- `top3_results.csv`、`top3_results_with_sig.csv`、`final_top3_with_stability.csv` 各有 147 行、49 个维度、每维 3 行；无 barnyard。
- 最终合理性表有 147 条 `main` 行；M3 行数等于 M3 与主方法 Top-1 不同的维度数。M3 行的维度、化合物、来源标签、区间和对照标记可追溯；缺少的统计量明确标为不适用，不写成 0。
- `graph_nodes.csv` 有 231 行；`graph_edges.csv` 中候选-候选边为 245 条，注释节点相关边单独统计。
- 63 个品评团；主稳定性和置换各按配置完成 1000 次。每个模拟设置恰为 1000 次。
- 附件 2 和附件 6 复算误差均 ≤ `1e-12`。
- A/B/C 只从 49 个主维度的 rank 1 行统计。由最终 `is_significant`、`pi_c` 独立复算等级，并逐项核对最终表。
- 附件 8 标签与 `chemical_dictionary.csv` 按化合物键逐项一致；空标签保留缺失，无阈值 OAV 保持缺失；breadth 不计 barnyard。
- 每个敏感性方案的维度数、有效重复次数、Jaccard、Top-1 变化、等级变化均有汇总；不存在待完成占位符。
- 更新 `figures/q2/`，并检查图中维度、Top-1 和图模块与本次 CSV 一致。
- 保存 `code/data/processed/q2/run_manifest.json` 和完整控制台日志。清单记录运行时间、Git HEAD、输入文件摘要、脚本命令、seed、bootstrap/permutation/simulation 次数、每个敏感性设置及输出 SHA-256。

## Coding 向 Modeling 移交

结果冻结后，列出 A/B/C 维度、主方法每维 Top-3、所有不显著/低稳定项、主方法与 M3 Top-1 不同项、共变标记、阈值证据、同模块成员及“同 barnyard”标记。`interpretation` 留空；不得由 Coding 根据名称自动生成机理解释。Modeling/Writing 完成解释后，再将 Q2 最终验收状态改为 resolved。

## 处理结果

待 Coding 完成后填写。
