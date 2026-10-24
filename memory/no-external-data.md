---
name: no-external-data
description: 建模不使用外部数据；Zenodo 记录只用于核验附件和检验假设
type: constraint
---

建模不使用外部数据；Zenodo 记录仅用于核验附件和验证无法用附件检验的假设。

**依据：** Kaggle 数据集不含化学数据，也没有可关联到这些啤酒的键；Zenodo 中仅有的带外部测量值的啤酒（64、223）已在样本中。

**如何应用：** 训练、调参、特征和样本均仅来自附件；基于 Zenodo 的检查仅用于报告，其来源在论文中声明。

相关：[[data-provenance-zenodo]]
