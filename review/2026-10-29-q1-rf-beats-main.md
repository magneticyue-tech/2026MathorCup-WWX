---
id: 2026-10-29-q1-rf-beats-main
from: coding
to: modeling
status: resolved
severity: major
refs:
  - model/q1_methodology.md
  - review/2026-10-28-q1-w6-metrics.md
  - code/data/processed/q1/q1_metrics.json
  - code/data/processed/q1/q1_baselines.npz
---

## 问题

W6 的运行前声明判定 `control_beats_main` 为 True：随机森林对照相对主模型的等级 MAE 配对差 95% CI 上界低于 0（方法论 §5.3）。
按方法论与 W6 工单，本线程只供讨论，coding 未切换任何模型、规则或超参数。
`success` 为 True，`linear_overturn` 为 False，`leak_flag` 为 False，`beaters` 为 `['rf']`。

## 背景 / 复现

命令：`$PY code/scripts/solve_q1_metrics.py --workers 8`（`PY=C:/Users/11600/anaconda3/envs/mathorcup_new/python.exe`）；bootstrap 按日期组重抽样 2000 次，`SEED = 0`。

`rf.median − main.median`（规则均为预测分布中位数，5 次重复各自计算再平均）：

| 统计量 | 点估计 | 95% CI |
|---|---|---|
| 等级 MAE | −0.0078 | [−0.0146, −0.0012] |
| 完全一致率 | +0.0088 | [+0.0038, +0.0140] |
| QWK | −0.0382 | [−0.0489, −0.0258] |

rf 在 36/50 个维度上等级 MAE 更低。
rf 连续 MAE 0.4661、R² 0.2154、Spearman 0.3743；主模型分别为 0.4726、0.1842、0.3595。
rf 的超参数在各外层折、各维度组内由内层 CV 选择（W4），25 个外层折中 Sour、Basic 组每折都选 `max_features=0.33, min_samples_leaf=5`。

## 处理结果

modeling 验收（2026-10-29）：不切换主模型，RF 作为对照照实报告。

- 按 §3.1、§5.3 的运行前声明，外层测试结果只报告，不用于选型；据此切换会复用报告性能的测试折，使报告的误差偏乐观。
- RF 没有在全部指标上占优。正式结果（中位数规则）：等级 MAE 主模型 0.4794、RF 0.4716；QWK 主模型 0.290、RF 0.251，差 −0.038 [−0.049, −0.026]；ElasticNet 0.481 / 0.199。RF 的等级 MAE 更低（相对低 1.6%），代价是等级区分度更低。在只用化学数据的模型中，主模型 QWK 最高（lin10 0.272、C2 0.272、RF 0.251、ElasticNet 0.199）。
- RF 的优势不是主模型网格边界造成的：`experimentalCode/q1_raw_target/exp_grid_boundary.py` 把网格扩展到 10–200 轮、深度 1–3 后，重复 0 的连续 MAE 为 0.4732，正式运行为 0.4733，RF 为 0.4661（5 次重复平均）。见 §3.1“网格边界检查”。
- 写作：论文报告 RF 等级 MAE 配对差的 CI 上界低于 0，并说明两种树集成给出一致结论：连续 R² 分别为 0.184 与 0.215，B1 为 0.042；主模型对 B1 的等级 MAE 配对差为 −0.048，CI 上界 −0.034（RF 对 B1 未做配对检验）。不写“主模型最优”。
- 教训见 [[q1-main-model-retained]]。
