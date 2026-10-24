---
name: internal-standard-3-methyl-2-butenal
description: 附件 8 将 X3.methyl.2.butenal 标为内标（Internal standard solution），不作为关键化合物候选
type: decision
---

附件 1 的 `X3.methyl.2.butenal`（3-methyl-2-butenal）在附件 8 *Sensory thresholds* 中的来源为 *Internal standard solution*，即实验室加入的定量参照，不代表啤酒组成。

**依据：** `results/data_audit/chemical_dictionary.csv` 的 `origin_paper` 字段；该列在附件 1 中没有底值，最小值 4.81（`src/q2_methodology_checks.py` 检查 1）。

**如何应用：** 问题 2 将其排除在 Top-3 候选之外，敏感性运行再加回；问题 3 不把它解释为风味成分或生成目标。

相关：[[composite-columns-near-duplicates]]
