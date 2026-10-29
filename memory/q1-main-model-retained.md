---
name: q1-main-model-retained
description: Q1 主模型 HGB 在 RF 对照等级 MAE 更低、网格落在下边界时仍保留；用边界扩展检查代替全量重算
type: decision
---

Q1 主模型保持 `HistGradientBoostingRegressor` 与原调参网格，不按外层结果改选 RF，也不按扩展网格重算。

**依据：** 2026-10-29 验收。RF 等级 MAE 0.4716 低于主模型 0.4794（配对差 CI 上界 −0.0012），但 QWK 0.251 低于主模型 0.290（差 −0.038 [−0.049, −0.026]）；在只用化学数据的模型中主模型 QWK 最高。主网格 150 次选择全部取 `max_iter` 下边界 100；`experimentalCode/q1_raw_target/exp_grid_boundary.py` 扩展到 10–200 轮、深度 1–3 后最优点位于内部，重复 0 的外层连续 MAE 0.4732 对 0.4733，Spearman 反降 0.019。

**如何应用：** 对照模型胜出或超参数落在网格边界时，先用一个重复做边界扩展检查，量化对外层指标的影响；影响在重复间极差内则保留正式结果，把检查写入方法源文件与不足，不触发 Q1→Q2→Q3 的全量重算。改选模型须把模型族纳入内层选择后重新评价。

相关：[[method-decision-closure]]
