---
name: data-provenance-zenodo
description: 附件 1–8 来自 Schreurs et al. (2024, Nature Communications)；该论文的数据记录后续新增逐品评员原始评分，是 Q1、Q2、Q3 的感官数据来源
type: reference
---

附件 1–8 为 Schreurs et al. (2024)（*Nature Communications* 15:2368）已发表的数据（Zenodo 记录 10653704），组织者导出后未作改动。该论文的作者后来在同一数据记录的新版（Zenodo 23624915，DOI `10.5281/zenodo.23624915`）公布了逐品评员、逐啤酒的未处理评分 `Raw sensory data anonymized.csv`，覆盖同一 250 款啤酒（2,197 条记录）。用户提供的 accession `1065370493` 在 Zenodo API 返回 404；检索同题记录后确认实际发布原始评分的是 `23624915`。

**依据：** 单元格级比较的最大数值差为 8.9e-16，每个 CSV 中 NA 的位置完全相同；xlsx 文件除模板工作表 beer_1–beer_11 的 beer-number 单元格、附件 3 中多出的一个汇总工作表、附件 8 中多出的一个空工作表外均一致（`src/q1_supplement_checks.py` 代码块 A）。附件 4 是原始评分的确定性变换（复现 90.2% 单元，见 [[attachment4-standardized-panel]]）；只用附件把 $z$ 换算为 0–6 的构造参照与真实等级只有 62.9% 一致（`model/q1_methodology.md` §1.3、§5.3）。Kaggle 感官数据集不含化学数据，也没有可关联到这些啤酒的键，不使用。2026-10-26 经预运行比较后，由用户决定 Q1 改用原始评分；2026-10-28 决定 Q2 同样使用。

**如何应用：** 原始评分与附件同属该论文的数据发布，不按外部数据处理。Q1 以啤酒均分 $\bar x$ 为训练目标、取整等级为评价真值、品评日期为交叉验证分组；Q2 以 $\bar x$ 的秩为感官目标、日期组为 bootstrap 重抽单位（`model/q2_methodology.md` §2.3、§2.4）；Q3 的感官量经 Q1 模型与 $\bar x$ 取得。附件 4 在 Q1 只用于对照 C1、C2，在 Q2 只用于敏感性臂。论文须声明原始评分的来源，并写明它在赛题发布后才公布。同一数据记录中的 *Validation Results 1–3*（加标实验）只用于 Q1 的对照杯检查和 Q2 §6.5 的报告，不参与建模；该数据记录之外的数据不参与建模。

论文方法部分可直接作为数据生成过程的描述；组织者未重新标准化、未删除样本、未添加噪声。`src/q1_supplement_checks.py` 需要该 Zenodo 工作簿。原始评分文件保存在 `data/raw/raw_sensory_data_anonymized.csv`，编码为 cp1252，其 `beer_name` 是附件 1/4 的行号而不是 `beer` 列（`model/q1_methodology.md` §1.1–§1.2）。

相关：[[attachment4-standardized-panel]]、[[sample-size-250]]
