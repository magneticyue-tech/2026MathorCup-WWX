---
id: 2026-10-28-q2-statistical-fixes
from: modeling
to: coding
status: resolved
severity: major
refs:
  - review/2026-10-28-q2-complete-run.md
  - model/q2_methodology.md
  - code/src/q1/splits.py
  - code/data/processed/q1/q1_data.npz
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
3. 修改 `code/src/q2/full_controls.py`：移除普通 `KFold`、Q2 自建调参网格和普通三折内层搜索；复用 Q1 记录的参数与 `outer_splits(date_group, style, n_splits=5, n_repeats=5)` 口径（`date_group` 取自 `q1_data.npz`，见 `q2-final-run-acceptance` 的“输入口径”），在每个外层测试折计算 `n_repeats=5` 的置换重要度。
4. 输出每个维度的 25 个外层折重要度或等价明细、汇总 M3 Top-3、模块、Jaccard 和 Top-1 是否与主方法相同。若 Q1 产物缺少确定所需参数，停止并在此工单说明缺项，不得自造参数。

**小样本验收：** 用前 1 个重复的 5 折运行一个维度；每折训练/测试的日期组不相交；参数与记录的 Q1 产物逐项相同；有 5 折重要度记录。小样本文件不得被当作正式结果。

## 步骤 2：固定风格内关联区间口径

当前 §4.6 要求报告风格内关联及 95% CI，但没有规定 CI 算法。采用与主稳定性一致的日期组簇 bootstrap，并先补充 `model/q2_methodology.md` §4.6 的区间定义：

- 固定主方法 Top-3 化合物与维度；每次有放回抽取 25 个日期组。
- 在该重复样本内重新计算化学秩和感官秩（感官为 `xbar` 的秩），再分别对 22 个风格指示变量（含截距）回归取残差。
- 对固定化合物和维度的残差计算相关；重复 `config.N_BOOTSTRAP=1000` 次，取 2.5% 与 97.5% 分位作为区间。
- 记录估计值、区间端点和区间是否包含 0。单次重复中残差方差为零时记为缺失并报告次数，不填 0。
- 随机数使用 `config.SEED`；bootstrap 不改变主结果选择。

修改 `rationality.py`，让 `flag_style_ci0` 来自该区间，且 `covariation = flag_style_ci0 OR flag_breadth5 OR flag_oav_lt1`。缺少风格数据或区间时保留缺失并显式报错/报告，不得当作 False。

**小样本验收：** 运行 20 次 bootstrap；结果每个主 Top-3 项一行，CI 有序、落在 [-1, 1]；固定种子重复运行得到相同结果；验证区间含 0 的 flag 与端点一致。正式值仍须 1000 次。

## 步骤 3：补齐模拟与审计

1. `simulation.py` 从配置读取模拟次数和种子；设置值为每种权重/r 组合 1000 次。若新增配置键，提供默认值并在本工单说明。
2. 按方法 §7.2 输出主方法与 M1 的化合物找回率、模块找回率，以及 Top-3 中非植入模块所占比例。明确“找回率”分母为模拟重复数，所有 6 种设置均有 1000 次记录。模拟噪声按方法 §7.2 由日期组偏移和个体噪声两部分组成，日期组偏移的方差占比取 `xbar` 的秩在 25 个日期组上的组间方差占比，由 `date_group` 现算，不写死常数。
3. `advanced_audits.py` 中 `test_attachment6_replication()` 当前是空 `pass`。将它实现为可执行的附件 6 复算验收，或明确移除该重复入口并让主验收入口调用 `check_attachment6.py`；不得留下会显示成功但未检查的空函数。
4. 零膨胀检查必须通过 `load_q2_dataset()` 对齐样本 ID 和列名，以 `xbar` 等于 0 的啤酒（全部品评员均评 0）判定零膨胀，不使用 `at_zero` 掩码；按方法 §6.6，只检查 C 级维度中至少 30% 啤酒 `xbar` 为 0 的维度，在 `xbar` 大于 0 的啤酒上重算 Top-1 关联。报告全样本与 `xbar>0` 样本的 Top-1、关联值、样本数；不能用原始全列数据误纳入 5 个综合指标或内标。

**小样本验收：** 每种模拟设置 10 次；结果六行齐全，比例在 [0,1]；`advanced_audits` 不允许空实现；`xbar` 为 0 的掩码与 250 个对齐样本逐行一致。

## 本工单验收

- 上述三个步骤的小样本检查通过，随机种子和试运行次数写入记录。
- 不改写附件原始文件，不在正式输出路径写入小样本结果。
- 不运行全量 bootstrap、正式敏感性或整套 Q2；这些由主工单第 3 步启动。
- 不运行 `ruff` 或 `black`。

## 处理结果

- 完成状态：完成，小样本验收通过（Coding，2026-10-29）。
- 修改文件：`code/src/q2/config.py`、`data.py`、`full_controls.py`、`rationality.py`、`simulation.py`、`advanced_audits.py`、`check_attachment6.py`；新增 `core.py`、`labels.py`。
- M3 参数来源：`code/data/processed/q1/q1_main_oof.npz` 的 `chosen_params`（键 `r_k_组`，25 外层折 × 6 组，含 `mono`）；外层划分为 `src.q1.splits.outer_splits(date_group, style, 5, 5, seed=0)`，与 `fold_id` 逐折一致，各折训练/测试品评团不相交；单调约束用 `src.q1.features.monotone_cst`。特征为 225 个候选原值、训练折列均值填充；目标为附件 4 z 的标准化秩；置换重要度 `n_repeats=5`，`random_state=SEED+10r+k`。已移除 `KFold`、Q2 自建网格与三折内层搜索。
- 风格内区间：品评团簇 bootstrap，与主稳定性同一组索引（`core.bootstrap_indices`，SEED=0）；重抽样本内重算秩，对截距 + 21 个风格虚拟变量回归取残差后求相关；2.5%/97.5% 分位；`flag_style_ci0` 来自该区间，`covariation = flag_style_ci0 OR flag_breadth5 OR flag_oav_lt1`（缺失按三值逻辑保留）。`flag_oav_lt1` 改为“全部啤酒 OAV<1”（原代码为比例<1）。
- 模拟：配置新增 `N_SIMULATION=1000`、`SIM_WEIGHTS`、`SIM_TARGET_R`、`SIM_PANEL_OFFSET_SD=0.15`（同检查 7）；输出 6 行，主方法与 M1 的化合物层、模块层找回率和非植入模块比例，分母为模拟重复数。
- 审计：`advanced_audits.py` 调用 `check_attachment6.check_att6()`（误差超 1e-12 报错）；零膨胀经 `load_q2_dataset()` 对齐，`at_zero` 由 `src.q1.panels.at_zero_mask` 生成，只检查最终 C 级维度。
- 其他修正：`rank_standardize` 删除对下限 −1.81781854 的截断（原代码把 15 列共 1,172 个低于该值的单元并列为下限）。
- 小样本命令与结果：`C:/Users/11600/anaconda3/envs/mathorcup_new/python.exe code/src/q2/full_controls.py --smoke`（A_malt_all，重复 0 的 5 折，参数逐项一致）；风格内区间 20 次 ×2 次运行逐项相同、147 行、区间有序且在 [−1,1]、flag 与端点一致；`simulation.py --smoke` 每设置 10 次、6 行、比例在 [0,1]；`at_zero` 250×50 与样本逐行一致。小样本文件写入 `_smoke/` 后已删除。
- 剩余问题：`model/q2_methodology.md` §4.6 的区间定义补写未做（用户限定本次只做代码），由 Modeling 补入。

## 验收（Modeling，2026-10-29）

结论：不通过，退回 Coding。以下各项与方法及本工单不符，须在正式重跑前修正。

1. M3 目标为附件 4 $z$ 的秩（`full_controls.py:90` 取 `ds.Y_rank`，其来源为 `4.csv`），方法 §2.3、§5.2 规定为 `xbar` 的秩。`full_controls.py:57`、`:134` 用推断品评团核对训练/测试不相交，应改为 `date_group`；M2 的 `GroupKFold`（`full_controls.py:101`）同样改用 `date_group`。
2. 风格内区间的重抽单位为 63 个推断品评团（`rationality.py:56`），方法 §4.6（本次已补写区间定义）与 §5.1 规定为 25 个日期组。
3. 模拟噪声使用固定常数 `SIM_PANEL_OFFSET_SD=0.15` 和推断品评团（`simulation.py:26`、`:44`）。方法 §7.2 与本工单步骤 3 规定：日期组偏移的方差占比取 `xbar` 的秩在 25 个日期组上的组间方差占比，由 `date_group` 现算。删除该常数。
4. 零膨胀检查使用由 $z$ 推断的 `at_zero` 掩码（`advanced_audits.py:26`），且未执行“至少 30% 啤酒为 0”的筛选。方法 §6.6 与本工单步骤 4 规定以 `xbar == 0` 判定。按 `xbar`，预运行 C 级的 A_esters_flower、t2n 零值比例为 0.24、0.21，若正式 C 级仍为二者，均不满足筛选条件，应报告“无维度满足条件”，不输出重算结果。

保留项：`core.py` 的 Freedman–Lane 置换、`flag_oav_lt1` 改为“全部啤酒 OAV<1”、`covariation` 的三值逻辑、M3 超参数取自 `q1_main_oof.npz` 的 `chosen_params`。删除信号下限截断正确：低于 −1.81781854 的 1,172 个单元格分布在 15 列，这 15 列都没有下限单元格，原截断把真实低值并列为下限。

## 复验（Modeling，2026-10-29）

结论：通过。§“验收”四项已修正。

1. M2、M3 的感官目标为 `xbar` 的秩，分组为 `date_group`。M3 外层划分改为由 `q1_main_oof.npz` 的 `fold_id` 重建：`mathorcup` 环境（sklearn 1.9.1）调用 `outer_splits` 不能复现 Q1 在 sklearn 1.7.2 下的划分，Q1 产物是权威来源。小样本：5 折参数与 Q1 逐项一致，日期组不相交。
2. 风格内区间由日期组簇 bootstrap 计算，与主稳定性共用重抽索引（方法 §4.6）。正式：风格内 CI 含 0 的项 42/147。
3. 模拟噪声的日期组方差占比由 `xbar` 的秩现算，49 维中位数 0.29，删除 `SIM_PANEL_OFFSET_SD`；方法 §7.2 已写明公式。正式 6 种设置各 1000 次。
4. 零膨胀检查以 `xbar == 0` 判定并执行 30% 筛选。正式 C 级只有 t2n，零值占比 0.212，记为不适用（`zero_inflation_c_grade.csv`）。

命令：`MKL_THREADING_LAYER=SEQUENTIAL` 下以 `mathorcup` 环境运行 `code/src/q2/run_all.py`，15 步返回码均为 0；日志 `code/data/processed/q2/run_log.txt`，清单 `run_manifest.json`（含 `input_checks` 与各日期组大小）。SEED=0。
