---
id: 2026-10-28-q2-statistical-fixes
from: modeling
to: coding
status: open
severity: major
refs:
  - review/2026-10-28-q2-complete-run.md
  - model/q2_methodology.md
  - code/src/q1/splits.py
  - code/src/q1/model.py
  - code/src/q2/full_controls.py
  - code/src/q2/rationality.py
  - code/src/q2/simulation.py
  - code/src/q2/advanced_audits.py
---

## 目标

只修 Q2 的统计实现和小型验收入口。本工单完成并通过小样本检查后，才能开始正式长时间运行。

## 步骤 1：盘点 Q1 的 M3 参数来源

1. 阅读 `model/q2_methodology.md` §5.2、Q1 的 `code/src/q1/splits.py`、`code/src/q1/model.py`、`code/scripts/solve_q1_main.py`，以及 `code/data/processed/q1/q1_main_oof.npz` 中的 `chosen_params`、`fold_id`、`seed`。
2. 确认 Q1 中每个感官维度所属组、该组正式选出的 GBM 参数、单调约束和外层分组方式。将来源记录在本工单，不从 Q2 结果反向挑参数。
3. 修改 `code/src/q2/full_controls.py`：移除普通 `KFold`、Q2 自建调参网格和普通三折内层搜索；复用 Q1 记录的参数与 `outer_splits(..., n_splits=5, n_repeats=5)` 口径，在每个外层测试折计算 `n_repeats=5` 的置换重要度。
4. 输出每个维度的 25 个外层折重要度或等价明细、汇总 M3 Top-3、模块、Jaccard 和 Top-1 是否与主方法相同。若 Q1 产物缺少确定所需参数，停止并在此工单说明缺项，不得自造参数。

**小样本验收：** 用前 1 个重复的 5 折运行一个维度；每折训练/测试的品评团 ID 不相交；参数与记录的 Q1 产物逐项相同；有 5 折重要度记录。小样本文件不得被当作正式结果。

## 步骤 2：固定风格内关联区间口径

当前 §4.6 要求报告风格内关联及 95% CI，但没有规定 CI 算法。采用与主稳定性一致的品评团簇 bootstrap，并先补充 `model/q2_methodology.md` §4.6 的区间定义：

- 固定主方法 Top-3 化合物与维度；每次有放回抽取 63 个品评团。
- 在该重复样本内重新计算化学秩和感官秩，再分别对 22 个风格指示变量（含截距）回归取残差。
- 对固定化合物和维度的残差计算相关；重复 `config.N_BOOTSTRAP=1000` 次，取 2.5% 与 97.5% 分位作为区间。
- 记录估计值、区间端点和区间是否包含 0。单次重复中残差方差为零时记为缺失并报告次数，不填 0。
- 随机数使用 `config.SEED`；bootstrap 不改变主结果选择。

修改 `rationality.py`，让 `flag_style_ci0` 来自该区间，且 `covariation = flag_style_ci0 OR flag_breadth5 OR flag_oav_lt1`。缺少风格数据或区间时保留缺失并显式报错/报告，不得当作 False。

**小样本验收：** 运行 20 次 bootstrap；结果每个主 Top-3 项一行，CI 有序、落在 [-1, 1]；固定种子重复运行得到相同结果；验证区间含 0 的 flag 与端点一致。正式值仍须 1000 次。

## 步骤 3：补齐模拟与审计

1. `simulation.py` 从配置读取模拟次数和种子；设置值为每种权重/r 组合 1000 次。若新增配置键，提供默认值并在本工单说明。
2. 按方法 §7.2 输出主方法与 M1 的化合物找回率、模块找回率，以及 Top-3 中非植入模块所占比例。明确“找回率”分母为模拟重复数，所有 6 种设置均有 1000 次记录。
3. `advanced_audits.py` 中 `test_attachment6_replication()` 当前是空 `pass`。将它实现为可执行的附件 6 复算验收，或明确移除该重复入口并让主验收入口调用 `check_attachment6.py`；不得留下会显示成功但未检查的空函数。
4. 零膨胀检查必须通过 `load_q2_dataset()` 对齐样本 ID 和列名，使用方法规定的 `at_zero` 掩码，只检查 C 级维度。报告全样本/非零点样本的 Top-1、关联值、样本数；不能用原始全列数据误纳入 5 个综合指标或内标。

**小样本验收：** 每种模拟设置 10 次；结果六行齐全，比例在 [0,1]；`advanced_audits` 不允许空实现；零膨胀掩码与 250 个对齐样本逐行一致。

## 本工单验收

- 上述三个步骤的小样本检查通过，随机种子和试运行次数写入记录。
- 不改写附件原始文件，不在正式输出路径写入小样本结果。
- 不运行全量 bootstrap、正式敏感性或整套 Q2；这些由主工单第 3 步启动。
- 不运行 `ruff` 或 `black`。

## 处理结果

待 Coding 完成后填写。
