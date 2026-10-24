---
name: barnyard-stale-hops-duplicate
description: barnyard 很可能是 stale_hops 的副本；照常预测但标记，不用于化合物归因
type: decision
---

附件 4 的 barnyard 很可能是 stale_hops 的副本。

**依据：** Spearman 0.966，z 绝对差的中位数为 0.009，两者均由模板单元格 AG76 计算得到（`src/q1_methodology_checks.py` 检查 1）。

**如何应用：** barnyard 会被预测，但会被标记，且不用于化合物归因。

相关：[[attachment4-standardized-panel]]
