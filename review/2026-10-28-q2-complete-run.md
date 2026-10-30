---
id: 2026-10-28-q2-complete-run
from: modeling
to: coding
status: ack
severity: major
refs:
  - model/q2_methodology.md
  - review/2026-10-28-q2-statistical-fixes.md
  - review/2026-10-28-q2-sensitivity-runs.md
  - review/2026-10-28-q2-final-run-acceptance.md
  - code/src/q2/
  - code/data/processed/q2/
  - figures/q2/
---

## 目标

按当前 `model/q2_methodology.md` 修正 Q2 实现，完整执行基线和敏感性实验，更新结果产物并逐项验收。以方法文档为唯一模型口径；代码与方法冲突时直接按方法修正代码并在回报中记录。只有方法自身缺定义或相互矛盾、无法确定目标量与计算规则时，才在问题所在方法段落提出具体修订并开复核记录；依赖该定义的步骤暂停，独立步骤继续。不得猜测或用旧输出填补。

本工单只负责实验及可复现产物。不得编写“维度 → 预期化合物”映射、命中率或事后类别。化学解释交由 Modeling 在结果冻结后完成。

## 执行顺序

按以下顺序完成，不得跳步：

1. 完成 `q2-statistical-fixes`。修复 M3、风格内区间、模拟指标和审计入口；运行其小样本测试。
2. 完成 `q2-sensitivity-runs`。按该工单和 `model/q2_methodology.md` 已定的 NA 敏感性范围实现全部方案；运行小样本测试。
3. 完成 `q2-final-run-acceptance`。按依赖顺序完整重跑基线、对照、模拟、敏感性和图表；通过数据和产物验收。
4. 完成结果冻结后，向 Modeling 移交最终表、逐维结果和未解释项清单。Coding 不写化学解释。

每张子工单必须在 `## 处理结果` 记录：完成状态、修改文件、运行命令、随机种子、实际重复次数、输出路径、验收指标和剩余问题。失败要保留错误信息并停止依赖该结果的后续步骤。

## 固定验收口径

- 输入：250 款啤酒、225 个候选化学列、6 个图注释列；交付 49 个感官维度，barnyard 不单独建模，stale_hops 标注“同 barnyard”。
- 感官目标与分组：感官目标为 `xbar` 的秩，bootstrap 重抽单位和分组 CV 单位为 25 个日期组；来源、键、形状和缺项处理见 `q2-final-run-acceptance` 的“输入口径”。附件 4 的 `z` 只用于感官目标的敏感性臂。
- 主表：49 个维度各有且仅有 3 行 `main`；仅在 M3 Top-1 与主方法不同时追加对应 `M3` 行。
- 图：231 个节点；阈值 0.6 的候选间边为 245 条，涉及注释节点的边另报。
- 正式置换与主 bootstrap：均使用 `config.py` 的 `SEED`、`N_PERMUTATION=1000`、`N_BOOTSTRAP=1000`。不得用预运行或旧产物代替。
- 附件 2 与附件 1 复算最大误差、附件 6 44 行复算最大误差均不超过 `1e-12`。
- 维度等级按方法 §4.5 由正式显著性和 `pi_c` 重算；最终报告 A/B/C 数量及维度名单。

## 当前已知执行注意

- 原始数据路径由 `code/src/q2/config.py` 指向 `data/raw`，Q2 结果目录为 `code/data/processed/q2`。每次运行前核实路径，不能套用旧工单中的目录。
- 现有 `code/src/q2/` 中读取 `q1_panels.npz`、附件 4 的 `z`、63 个推断品评团或 `at_zero` 掩码的位置（已知涉及 `config.py`、`data.py`、`stability.py`、`full_controls.py`、`advanced_audits.py`、`controls_sensitivity.py`），按方法 §2.3、§2.4、§5.1、§6.6 改为 `xbar` 的秩与 `date_group`，再运行任何步骤。
- Modeling/Writing 向结果文档写入前，重新读 `model/` 的活文档、`figures/` 和 `code/data/processed/` 最新结果。
- Coding 每次运行求解器前，重新读本工单、三个子工单和 `model/q2_methodology.md`。

## 处理结果

- 三张子工单已完成，结果与验收见各自 `## 处理结果`；正式产物在 `code/data/processed/q2/`、`figures/q2/`。
- 代码与旧实现的差异：选择、置换、bootstrap 统一由 `code/src/q2/core.py` 在相关矩阵上计算（与残差回归定义逐维一致）；模块只在候选上聚类（161 模块），注释节点不参与聚类；显著性为“超过置换门槛且 bootstrap CI 不含 0”；τ 敏感性从 `controls_sensitivity.py` 移到 `sensitivity_runs.py` 并完整重跑；删除了对信号下限的截断。
- 待 Modeling：化学解释、`model/q2_methodology.md` §4.6 区间定义、`model/sensitivity_analysis.md` Q2 行。

## 验收（Modeling，2026-10-29）

结论：不通过。三张子工单均退回 Coding，原因见各自 `## 验收`。根本原因是实现仍以附件 4 的 $z$ 与 63 个推断品评团为口径，未按方法 §2.3、§2.4 改为 `xbar` 的秩与 25 个日期组，本工单“当前已知执行注意”第 2 条未执行。

Modeling 本次完成：方法 §4.6 补写风格内区间定义；§2.2、§7.1 明确附件 6 复算使用 $z$，主分析与逐对删除诊断使用 `xbar` 的秩。化学解释与 `model/sensitivity_analysis.md` Q2 行在重跑验收通过后进行。

## 复验（Modeling，2026-10-29）

结论：`q2-statistical-fixes`、`q2-sensitivity-runs` 复验通过并关闭；`q2-final-run-acceptance` 数据验收通过，结果已冻结。剩余事项为 Modeling 依据 `handoff_modeling.md` 撰写化学解释（方法 §6.4），完成后关闭本线程与最终验收线程。Writing 可据冻结结果更新 `paper/main.tex` 中仍按 63 个品评团描述的问题二段落。
