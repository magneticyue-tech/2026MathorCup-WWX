---
name: data-provenance-zenodo
description: 附件 1–8 即 Schreurs et al. (2024) Zenodo 记录 10653704 的原始数据，导出后未改动
type: reference
---

附件 1–8 为 Schreurs et al. (2024) 已发表的数据（Zenodo 记录 10653704），组织者导出后未作改动。

**依据：** 单元格级比较的最大数值差为 8.9e-16，每个 CSV 中 NA 的位置完全相同；xlsx 文件除模板工作表 beer_1–beer_11 的 beer-number 单元格、附件 3 中多出的一个汇总工作表、附件 8 中多出的一个空工作表外均一致（`src/q1_supplement_checks.py` 代码块 A）。

**如何应用：** 论文方法部分可直接作为数据生成过程的描述；组织者未重新标准化、未删除样本、未添加噪声。`src/q1_supplement_checks.py` 需要该 Zenodo 工作簿。

相关：[[no-external-data]]、[[sample-size-250]]
