# 协作规范

## 团队角色与职责归属
- Modeling——负责假设、推导、模型选择，以及评审结果解读。负责 `model/`。
- Coding——负责实现、求解器运行、图和表。负责 `code/`。
- Writing——负责论文草稿、排版和结果解读。负责 `paper/` 与 `model/result_interpretation.md`。

## model/

按问题编排的方法论文档，每个问题一份，命名为 `q<N>_methodology.md`
（`q1_methodology.md`、`q2_methodology.md`、……），由
`model/_METHODOLOGY_TEMPLATE.md` 创建。

持续维护的活文档：`q<N>_methodology.md`、`assumptions.md`、
`summary_sheet.md`、`sensitivity_analysis.md`、`result_interpretation.md`、
`strengths_and_weaknesses.md`。论文中对应的章节参考文档。

`assumptions.md` 只记录全局假设——即约束整个问题的数据、范围或环境的假设（例如单种群抽样、SI 单位、静风飞行）；它生成论文的 Assumptions 章节，并保存共享符号表。方法局部假设——与某一方法或方程绑定的函数形式、估计量或范围选择（例如状态的指数衰减、特定插补方案、不剔除异常值）——在相关的 `q<N>_methodology.md` 中使用处说明，并明确表述为假设，以便 `sensitivity_analysis.md` 和 `strengths_and_weaknesses.md` 引用；这些假设不列入 `assumptions.md`。引用按内容，不按编号。

## 跨团队同步
模型文件和求解器结果经常变化，基于过期副本行动会浪费工作。
开始任务前，以及再次向下游写入任何内容前：
- Modeling 和 Writing：重新阅读 `model/`（`assumptions.md`、
  `summary_sheet.md`、`sensitivity_analysis.md`、`result_interpretation.md`、
  `strengths_and_weaknesses.md`），以及 `figures/` 和 `code/data/processed/` 中最新的求解器输出。
- Coding：在运行求解器前重新阅读 `model/q<N>_methodology.md`，使实现与当前模型一致。

## 文档语言
中文

## 输出规范
报告语气：
- 用朴素的名词短语说明创建或更改了什么。不使用卖点式括号和为可读性而做的铺陈。
- 平实陈述结果。
- 使用 NASA NPR 1450.10DNASA Correspondence Management and Communications Standards and Style。