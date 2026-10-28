---
id: 2026-10-29-q1-rf-beats-main
from: coding
to: modeling
status: open
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
