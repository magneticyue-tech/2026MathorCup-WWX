---
name: chemical-na-mean-imputation
description: 附件 1 的 73 个化学 NA 在主设定中按训练折列均值填充；低于检出限编码改为敏感性对照
type: decision
---

附件 1 的 9 个化学列共有 73 个 NA。主设定在训练折内按列均值填充，并将训练折均值应用于验证折和测试折。填充值只作计算占位，不解释为真实零或低于检出限。

**依据：** 同源 Nature Communications 论文在 Methods 中将相同数据的缺失化学测量报告为 0.1%，并按属性均值填充。附件 1 没有记录缺失原因或逐列检出限；附件 8 将 8 个单化合物的 60 个 NA 位置记为 0，但跨附件编码差异不能证明测量机制。`aroma_hops_sum` 的公式和组成未提供，也不能由现有字段重算。

**如何应用：** 不把“NA = 未检出”当作事实。中位数加缺失指示变量、HistGBR 原生 NaN，以及 8 个单化合物的最低位编码作为敏感性对照；具体规则见 `model/q1_methodology.md` §2.1 和 `model/sensitivity_analysis.md`。

相关：[[signal-floor-truncation]]、[[data-provenance-zenodo]]
