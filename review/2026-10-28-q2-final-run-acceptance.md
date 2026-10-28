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

只有在 `q2-statistical-fixes` 通过其小样本验收，且 `q2-sensitivity-runs` 的全部敏感性代码通过小样本验收后，才开始本工单。逐对删除按 `model/q2_methodology.md` §2.2、§7.3 限定为边际相关诊断。开始前重新读取当前方法文档和 Q2 活文档，检查 `git status`、原始附件、Q1 的 `q1_data.npz` 与 `q1_main_oof.npz`（见下方“输入口径”）和全部 Q2 输出路径。

## 输入口径

感官目标与分组取自问题 1 的 W1 产物，口径见 `model/q2_methodology.md` §2.3、§2.4。

| 项 | 文件与键 | 形状与顺序 | 用途 |
|---|---|---|---|
| 感官目标 | `code/data/processed/q1/q1_data.npz` 的 `xbar` | (250, 50)；行序为 `beer_row`=1..250（原始评分文件 `beer_name` 的行号，不是附件的 `beer` 列）；列序为 `DIMS` | 取 49 个维度（去 barnyard）的秩，作为主感官目标 |
| 日期组 | 同文件的 `date_group` | (250,)；下标为 `beer_row-1`；取值 0..24 | 簇 bootstrap 的重抽单位；M2 的 ElasticNet 分组 CV；M3 经 `outer_splits(date_group, style)` 的外层划分；模拟中的日期组偏移 |
| 附件 4 的 z | `data/raw/4.csv` 的 `DIMS` 列 | (250, 50)；按行号与 `xbar` 对齐 | 只用于感官目标的敏感性臂 |

运行前断言：

- `beer_row` 恰为 1..250；`date_group` 取值恰为 0..24 共 25 个，每款啤酒只属于一组；各组大小写入运行记录，与方法文档预运行值（每组 5–24 款）不一致时只报告，不改分组。
- `xbar` 与 `z` 按行号对齐时逐维 Spearman 中位数不低于 0.95；按 `beer` 列对齐时不高于 0.1。后一条用于发现行号与 `beer` 列混用。
- Q2 正式运行在 Q1 的 W1、W2 完成后进行（方法 §8）。

缺项时的动作：`q1_data.npz` 缺失、键缺失或形状断言不符，停止依赖该输入的步骤，在本工单记录缺项并开复核记录。不得从原始评分文件自行重算分组，不得读取 `code/data/processed/q1/legacy_z/` 下的旧产物（含 `q1_panels.npz`），不得用推断品评团替代日期组。

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
- 向 Writing 移交主方法 147 行的紧凑附录表数据视图：从最终合理性表只取 `method=main`，按 Q1 六组维度与附件 4 维度顺序、组内 rank 1–3 排列，列为维度、名次、化合物、方向、`pi_c`、`stability`、`grade`；六组互不重叠且合计 147 行。完整合理性 CSV 与 M3 差异行分别保留，不能把宽表或 M3 行直接排入主结果附录表。表注与不适用值口径见 `model/q2_methodology.md` §1。
- 最终合理性表有 147 条 `main` 行；M3 行数等于 M3 与主方法 Top-1 不同的维度数。M3 行的维度、化合物、来源标签、区间和对照标记可追溯；缺少的统计量明确标为不适用，不写成 0。
- `graph_nodes.csv` 有 231 行；`graph_edges.csv` 中候选-候选边为 245 条，注释节点相关边单独统计。
- 25 个日期组：每次 bootstrap 重复抽取 25 个日期组，记录重复数和各次重复抽到的不同日期组数；主稳定性和置换各按配置完成 1000 次。每个模拟设置恰为 1000 次。
- 感官目标为 `xbar` 的秩，附件 4 的 `z` 只出现在敏感性臂；运行记录和输出中不得出现 `q1_panels` 或 63 个推断品评团。`run_manifest.json` 记录 `q1_data.npz` 的摘要和各日期组大小。
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
