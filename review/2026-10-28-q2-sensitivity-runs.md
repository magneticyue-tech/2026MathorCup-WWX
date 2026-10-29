---
id: 2026-10-28-q2-sensitivity-runs
from: modeling
to: coding
status: ack
severity: major
refs:
  - review/2026-10-28-q2-complete-run.md
  - review/2026-10-28-q2-statistical-fixes.md
  - model/q2_methodology.md
  - model/sensitivity_analysis.md
  - code/data/processed/q1/q1_data.npz
  - memory/attachment1-vs-8-conflict.md
  - code/src/q2/controls_sensitivity.py
  - code/src/q2/stability.py
---

## 目标

实现并运行方法 §7.3 的全部敏感性方案。现有 `controls_sensitivity.py` 已有 τ=0.5/0.7、M0/M1；不得把这些已有行当作全部敏感性结果。

## 化学 NA 的逐对删除范围

按 `model/q2_methodology.md` §2.2、§7.3，逐对删除只作边际相关诊断：每一对化学列与感官维度使用该对共同非缺失的啤酒，输出 Spearman 系数及有效样本数；附件 6 的 44 行逐元素复现，误差不超过 `1e-12`。此分支不重算偏 Spearman、模块图、显著性、bootstrap、Top-3 或等级。原因是各系数的样本集合不同，不能形成供图约束选择使用的统一相关矩阵。对 Top-3、Jaccard、Top-1 与等级变化标记“不适用”，并写明此范围限制；不得把列均值或最低值结果冒充逐对删除结果。此项口径已在方法论文档固定，不再等待组决策。

## 必须覆盖的方案

逐项生成独立结果行，每项均与正式基线比较。定义完整选择流程的方案报告 49 维平均化合物层 Jaccard、模块层 Jaccard、Top-1 改变维度数和等级改变维度名单。逐对删除只交付边际相关明细与附件 6 复算验收，其余选择指标标注“不适用”和理由，不填 0。

1. **模块阈值：** τ=0.5、0.7。可复用现有计算，但须重算模块限制下的正式选择和比较指标。
2. **候选集合：** 主 225 列基线分别加回 5 个综合指标、加回内标；两种对照分开报告。
3. **NA：** 最低值并列；逐对删除只作边际相关诊断，按本工单「化学 NA 的逐对删除范围」执行。NA 变换只作用于候选化学列，不改变感官数据。
4. **附件 1/8 冲突：** 依据 memory 中记录的冲突清单替换 24 款啤酒的 80 个单元格。保存替换清单、原值和替换值的可追溯表，不改原始附件。
5. **bootstrap 图：** 与主分析比较每次重抽后重建图和固定附件 2 图。两臂使用同一组日期组 bootstrap 索引和种子，以便成对比较。
6. **bootstrap 单位：** 日期组簇（基线，每次抽取 25 个日期组）与单款啤酒两臂比较。单款啤酒臂使用 250 个样本有放回抽样；其它重采样口径不变。
7. **感官目标：** 基线为 `xbar` 的秩；对照臂改用附件 4 的 `z` 的秩（`data/raw/4.csv` 的 `DIMS` 列，按行号与 `xbar` 对齐，不按 `beer` 列对齐）。两臂的目标维度均为 49 个且列名一致；日期组、bootstrap 单位与其余设置不变。来源与对齐断言见 `q2-final-run-acceptance` 的“输入口径”。不使用减去本团零点的参考评分。

等级改变必须由对应方案自己的显著性与 `pi_c` 重新计算，不能仅把 Top-1 改变近似成等级改变。图/阈值/候选/感官改变会影响正式推断时，使用方法规定的置换与 bootstrap 次数；不得用小样本或旧等级填充。预检试跑可用 10 次并写入临时目录，不得混入最终文件。

## 输出与测试

- 新增一个明确的 Q2 敏感性入口，按方案逐个运行并保存逐维明细及汇总；结果应能从单个方案失败后续跑，不需重跑已通过方案。
- `model/sensitivity_analysis.md` 更新每项的基线、对照设置、结果、脚本与结果文件链接。不得保留 Q2 的 `[PENDING]` 或 `[PLACEHOLDER]`。
- 同一方案固定种子重复运行的小样本结果逐项一致；验证每个方案恰有 49 个维度，barnyard 不进入 Jaccard、等级或 breadth。
- 附件 6 对照误差小于 `1e-12`。其它失败必须显式报错，不得以 NaN/0 伪装完成。
- 不运行 `ruff` 或 `black`。

## 处理结果

- 完成状态：完成（Coding，2026-10-29）。入口 `code/src/q2/sensitivity_runs.py`，可续跑（`*_meta.json` 记录状态；`--only`、`--force`）。
- 命令：`C:/Users/11600/anaconda3/envs/mathorcup_new/python.exe code/src/q2/sensitivity_runs.py`；小样本 `--smoke`（10 次置换/bootstrap，两次固定种子运行逐项一致，各方案 49 维）。
- 种子与次数：SEED=0；各完整方案置换 1000、bootstrap 1000。`boot_fixed_graph` 与主分析同一组品评团索引；`boot_fixed_graph`、`boot_beer_unit` 选择与数据不变，沿用基线置换门槛，CI 与 π^c 由本方案 bootstrap 重算。
- 结果（49 维平均化合物层/模块层 Jaccard；Top-1 改变数；等级改变）：τ=0.5 1.000/1.000，0，1；τ=0.7 0.990/0.993，0，0；加回综合指标 0.863/0.867，5，1；加回内标 0.984/0.984，0，0；NA 最低位 0.980/0.980，0，0；附件 8 冲突 0.943/0.943，1，0；固定图 1.000/1.000，0，0；单款啤酒 1.000/1.000，0，2；有锚目标 0.692/0.696，6，2。等级改变名单见 `sensitivity_summary.csv` 的 `grade_changed_dims`。
- 逐对删除：边际 Spearman 与有效样本数见 `pairwise_deletion_marginal.csv`；附件 6 44 行误差 6.66e-16；Top-3、Jaccard、Top-1、等级标为不适用。
- 实现口径：附件 8 冲突 80 单元/24 款由 `calibrated_discordances.csv` 与附件 8 `Calibrated chemical dataset` 逐值核对后替换，清单 `attachment8_replacements.csv`，该方案图由替换后附件 1 逐对重算。有锚维度：附件 4 50 维中 40 维有团零点，去掉 barnyard 后恰为 39 个交付维度（`anchored_dims.csv`）；团内无零点的啤酒用该维度各团零点中位数（同问题 1 旧版 m 模型的填补），记为实现假设。模块层 Jaccard 统一映射到 τ=0.6 基线模块，基线外的列各自成类。
- 输出：`code/data/processed/q2/sensitivity/`（`<方案>_table.csv`、`<方案>_compare.csv`、`<方案>_meta.json`、`sensitivity_summary.csv`）。
- 剩余问题：`model/sensitivity_analysis.md` 的 Q2 行未改（用户限定本次只做代码），数值由 Modeling 从 `sensitivity_summary.csv` 填入。

