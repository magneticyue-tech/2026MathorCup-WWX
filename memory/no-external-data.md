---
name: no-external-data
description: 原始评分只作 Q1 对照基线；其他外部数据不参与建模
type: constraint
---

建模不使用外部训练数据。Zenodo 仅用于核验附件、验证无法用附件检验的假设，以及为 Q1 提供作者后续公开的原始评分对照基线。

**依据：** Kaggle 数据集不含化学数据，也没有可关联到这些啤酒的键；Zenodo 旧版仅有的带外部测量值的啤酒（64、223）已在样本中。Zenodo 23624915 后续新增 2,197 条逐品评员原始评分，覆盖 250 款啤酒。

**如何应用：** Q1 主模型、调参、特征和样本范围仍按赛题附件和既定方法执行；`data/raw/raw_sensory_data_anonymized.csv` 只构造原始均分基线，在相同外层折评价，不参与主模型选型或映射选择。其他基于 Zenodo 的检查仅用于报告，并声明来源。

相关：[[data-provenance-zenodo]]
