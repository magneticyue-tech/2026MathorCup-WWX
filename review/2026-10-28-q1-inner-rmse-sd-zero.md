---
id: 2026-10-28-q1-inner-rmse-sd-zero
from: coding
to: modeling
status: open
severity: major
refs:
  - review/2026-10-28-q1-w2-main-model.md
  - review/2026-10-28-q1-w9-oav.md
  - model/q1_methodology.md
  - code/src/q1/model.py
  - code/scripts/solve_q1_main.py
---

## 问题

W2 选参标准 RMSE/SD（`sd = np.std(y_va, ddof=0)`）在部分内层验证折上分母为 0，方法论 §3.1 与 W2 工单都没有规定此时如何处理。
受影响的是 Sour 组的 lactic 与 acetic：验证行的 x̄ 全部为 0。
按工单原式计算，该单元的 RMSE/SD 为 inf，Sour 组全部配置的平均得分都为 inf，`nanargmin` 退回第 0 个配置，选参失效。
冒烟运行中全数据重拟合（`refit_full`）的 Sour 组选出的正是第 0 个配置，与上述推断一致。
coding 已采用 W9 工单对同一量的约定实现：分母为零的「内层折 × 维度」单元记为 NaN，从配置得分的平均中剔除，不写成 0。
该单元的分母只由验证行目标决定，与配置无关，因此每个配置剔除的是同一批单元，配置之间仍可比较。
请 modeling 确认该做法，或给出其他规定；若改变规定，受影响的 6 个外层折与全数据重拟合需要重算，其余中间文件可复用。

## 背景 / 复现

外层划分 `outer_splits(date_group, style)`，内层种子 `SEED + 100 + r`；全数据重拟合内层为 `inner_splits(np.arange(250), ..., SEED + 200, n_splits=5)`。

受影响的单元（外层折 `r_k`，内层折号，维度，验证行数）：

| 单元 | 内层折 | 维度 | 验证行数 |
|---|---|---|---|
| 0_3 | 1 | lactic | 53 |
| 2_2 | 2 | acetic | 53 |
| 2_4 | 2 | acetic | 34 |
| 3_2 | 1 | lactic | 46 |
| 3_4 | 2 | lactic、acetic | 51 |
| 4_3 | 1 | acetic | 45 |
| 全数据重拟合 | 2 | acetic | 47 |

复现：在 `code/` 下对每个单元调用 `inner_splits` 并检查 `np.std(xbar[b], axis=0) == 0`。
正式运行把剔除的单元写入 `q1_main_oof.npz` 的 `inner_undefined`（JSON）。
测试 `test_tune_group_excludes_constant_validation_cells` 断言剔除后的得分等于对有定义单元直接求平均。

W4（ElasticNet、RF 的选参，「调参标准与主模型一致」）与 W5（z 模型的嵌套 CV 复用 `fit_outer_fold`）沿用同一实现。

## 处理结果
