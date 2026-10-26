---
name: data-provenance-zenodo
description: 附件 1–8 来自 Zenodo 10653704；逐品评员原始感官评分由后续版本 23624915 发布
type: reference
---

附件 1–8 为 Schreurs et al. (2024) 已发表的数据（Zenodo 记录 10653704），组织者导出后未作改动。作者后续在同一记录的新版（Zenodo 23624915，DOI `10.5281/zenodo.23624915`）新增 `Raw sensory data anonymized.csv`，包含逐品评员、逐啤酒的未处理评分。用户提供的 accession `1065370493` 在 Zenodo API 返回 404；检索同题记录后确认实际发布原始评分的是 `23624915`。

**依据：** 单元格级比较的最大数值差为 8.9e-16，每个 CSV 中 NA 的位置完全相同；xlsx 文件除模板工作表 beer_1–beer_11 的 beer-number 单元格、附件 3 中多出的一个汇总工作表、附件 8 中多出的一个空工作表外均一致（`src/q1_supplement_checks.py` 代码块 A）。

**如何应用：** 论文方法部分可直接作为数据生成过程的描述；组织者未重新标准化、未删除样本、未添加噪声。`src/q1_supplement_checks.py` 需要该 Zenodo 工作簿。原始评分文件保存在 `data/raw/raw_sensory_data_anonymized.csv`，编码为 cp1252，其 `beer_name` 是附件 1/4 的行号而不是 `beer` 列（`model/q1_methodology.md` §1.1–§1.2）；它在 Q1 中的用途见 [[no-external-data]]。

相关：[[no-external-data]]、[[sample-size-250]]
