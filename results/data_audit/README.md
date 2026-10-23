# 数据审计复现与文件索引

正文为[啤酒风味数据审计与附件说明](啤酒风味数据审计与附件说明.md)。日期为2026年10月23日。

在项目根目录使用现有Python环境运行：

```powershell
python src/run_data_audit.py
```

依赖为numpy、pandas、openpyxl。本次运行版本分别为2.3.5、3.0.1、3.1.5，Python为3.12.14。脚本仅读取原始附件；诊断用的数值视图、临时候选匹配和相关复算均不写入data/processed。运行会更新本目录审计结果和正文，不会改变data/raw或原始赛题。

如需分步复现：

```powershell
python src/audit_data.py
python src/audit_deep.py
python src/audit_supplement.py
python src/build_audit_report.py
```

必须依此顺序运行。第一步生成基础字典，第三步补入语义、单位和解释。第四步组合定量结果与人工核实的来源解释。

优先查阅以下文件：

- chemical_dictionary.csv：231个化学、理化和综合特征的完整解释。
- sensory_dictionary.csv：50个品评目标、模板字段、原始公式和类别。
- forum_dictionary.csv：186个论坛描述符，含全缺失列与映射缺口。
- variable_dictionary.csv：六个CSV全部810个列位置的统计、角色与解释。
- transform_verification.csv：70个参考字段的尺度核查；nonpositive字段包括零和负参考值。37个log10字段之外，不把ln或log10误差大小直接解释成变换的证据。
- calibrated_discordances.csv：按原始CSV每个数字最后一位精度排除舍入后，仍有差异的80个单元格。该文件保留csv1_literal原文，查看时请按文本导入。
- calibrated_missing_comparison.csv：两个来源的缺失、零值对应。
- identifier_candidates.csv：仅供审阅的完整标识候选，尚未实施恢复。CSV导入软件可能再次将其转成数值，应指定标识列为文本。
- correlation_reproduction.csv：共29990个独立或交叉系数的Spearman、Pearson复算与有效样本数。
- raw_manifest.json：原始附件与数据说明的SHA-256。
- deep_findings.json与supplement_findings.json：机器可读的结论和原始赛题、草稿校验值。

1_columns.csv等文件保留完整逐列统计。3_xlsx_cells.csv与8_xlsx_cells.csv记录全部非空单元格、坐标、原始公式、缓存值、类型与注释，供定位来源。它们是审计证据，不是清洗后的训练集。矩阵的首列是行标签，模板版面空白、参考注释空白和测量NA应分别解释。

处理建议和待确认内容见正文第9节。当前没有实施填补、删除、极值修正、标识覆盖、逆变换落盘、样本合并或模型训练。
