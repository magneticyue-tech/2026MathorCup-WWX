# 记忆索引

分段式项目记忆。本文件夹中每个文件记录一条事实，每个文件都带
frontmatter（见 `_TEMPLATE.md`）。本索引列出所有记忆文件，每条附一行提要。每新增一个文件，就在这里加一行；不要把记忆内容写进本索引。

记忆只保存未记录在别处的事实：项目决策及其理由、固定约束、外部参考。它不复述 `COLLABORATION.md` 或 `.claude/skills/` 文件中的规则——那些规则的权威版本在彼处，复制到这里只会逐渐失真。如果某条事实在变化时需要修改 COLLABORATION 或某个 skill，那它就应放在那里，而不是这里。

类别（通过 frontmatter 的 `type` 字段指定）：
- `decision` — 建模/编码/写作方面的选择及其理由
- `constraint` — 固定要求（数据源、截止时间、套餐限额）
- `reference` — 外部资源（链接、工单、数据集来源）

## 索引
- [数据来源：Zenodo 10653704 与 23624915](data-provenance-zenodo.md) — 后续版本新增逐品评员原始评分
- [样本量 250](sample-size-250.md) — 题面“226 种啤酒”是对 226 个化学参数的误读
- [附件 4 为标准化面板均值](attachment4-standardized-panel.md) — 由原始评分按品评员 × 维度标准化得到；仅凭附件 4 无法还原 0–6 分
- [化学 NA 按列均值填充](na-below-detection.md) — 主设定使用训练折列均值；最低位编码和原生 NaN 作为敏感性对照
- [信号列下限是截断值](signal-floor-truncation.md) — −1.81781854 只按最低排名使用
- [barnyard 疑似 stale_hops 副本](barnyard-stale-hops-duplicate.md) — 预测但标记，不用于归因
- [附件 1 与附件 8 冲突以附件 1 为准](attachment1-vs-8-conflict.md) — 附件 8 用于敏感性运行
- [外部数据使用范围](no-external-data.md) — 原始评分作为 Q1 训练目标与评价真值；其他外部数据不参与建模
- [3-methyl-2-butenal 是内标](internal-standard-3-methyl-2-butenal.md) — 附件 8 标为 Internal standard solution，不作为归因候选
- [综合指标近似单一成分](composite-columns-near-duplicates.md) — 5 个综合指标各近似一个成分，不作为归因候选
- [不编写事后预期表](no-post-hoc-expectation-table.md) — 化学合理性只用附件 8 原始标签与事后解释，不计算命中率
