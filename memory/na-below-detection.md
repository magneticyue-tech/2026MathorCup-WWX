---
name: na-below-detection
description: 附件 1 单化合物 NA 主设定按低于检出限编码为最低值；该解读只得到部分支持，作为敏感性实验而非假设
type: decision
---

附件 1 中单化合物 NA 在主设定中按低于检出限处理，编码为最低值并加指示变量，不做均值插补。该解读是敏感性实验的对象，不是假设。

**依据：** 附件 8 校准数据在这 60 个位置上全部为 0，同时在另外 69 个单元格中保留 NA，所以 0 是作者有意的编码；但附件 1 和附件 8 出自同一作者，这只能说明作者怎么编码，不能说明测量机制。按相关化合物代理秩看，X2.3.butanedione、X3.methylbutanal、methylpyrazine、ethyl.nicotinate 与低于检出限一致；amyl.acetate（27 个 NA）不能判定；`aroma_hops_sum` 的 NA 啤酒偏高，不是低值（`src/q1_methodology_checks.py` 检查 2）。

**如何应用：** 不要把“NA = 未检出”当作已确立的事实来引用；对照臂和判定规则见 `model/q1_methodology.md` §2.1 和 `model/sensitivity_analysis.md`。`aroma_hops_sum` 的 NA 保留为 NaN。

相关：[[signal-floor-truncation]]
