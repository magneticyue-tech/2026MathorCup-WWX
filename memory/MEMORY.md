# 记忆索引

分段式项目记忆。
frontmatter（见 `_TEMPLATE.md`）。本索引列出所有记忆文件，每条附一行提要。每新增一个文件，就在这里加一行；不要把记忆内容写进本索引。

记忆只保存：项目决策及其理由、固定约束、外部参考。它不复述其他规则。

类别（通过 frontmatter 的 `type` 字段指定）：
- `decision` — 建模/编码/写作方面的选择及其理由
- `constraint` — 固定要求（数据源、截止时间、套餐限额）
- `reference` — 外部资源（链接、工单、数据集来源）

## 索引
- [数据来源：Zenodo 10653704 与 23624915](data-provenance-zenodo.md) — 附件来自 Nature Communications 论文；该论文数据记录新增的逐品评员原始评分是 Q1–Q3 的感官数据来源
- [样本量 250](sample-size-250.md) — 题面“226 种啤酒”是对 226 个化学参数的误读
- [附件 4 为标准化面板均值](attachment4-standardized-panel.md) — 由原始评分按品评员 × 维度标准化得到；仅凭附件 4 无法还原 0–6 分
- [化学 NA 按列均值填充](na-below-detection.md) — 主设定使用训练折列均值；最低位编码和原生 NaN 作为敏感性对照
- [信号列下限是截断值](signal-floor-truncation.md) — −1.81781854 只按最低排名使用
- [barnyard 疑似 stale_hops 副本](barnyard-stale-hops-duplicate.md) — 预测但标记，不用于归因
- [附件 1 与附件 8 冲突以附件 1 为准](attachment1-vs-8-conflict.md) — 附件 8 用于敏感性运行
- [3-methyl-2-butenal 是内标](internal-standard-3-methyl-2-butenal.md) — 附件 8 标为 Internal standard solution，不作为归因候选
- [综合指标近似单一成分](composite-columns-near-duplicates.md) — 5 个综合指标各近似一个成分，不作为归因候选
- [不编写事后预期表](no-post-hoc-expectation-table.md) — 化学合理性只用附件 8 原始标签与事后解释，不计算命中率
- [方法决策闭环](method-decision-closure.md) — 先在方法源文件固定目标、估计与不适用口径，再由工单引用并同步下游文档
- [Q1 主模型保留](q1-main-model-retained.md) — RF 等级 MAE 更低但 QWK 更低、网格下边界经扩展检查无改善，主模型与网格不变
- [差异类指标不做 bootstrap 区间](discrepancy-metrics-no-bootstrap.md) — 经验量之差的绝对值平均与汇总量间的秩相关，重抽分布偏离点估计；只报点估计、重复极差与参照
- [Q1 划分取自 fold_id](q1-folds-from-fold-id.md) — sklearn 升级后 outer_splits 不复现 Q1 划分，下游读 fold_id；mathorcup 环境须设 MKL_THREADING_LAYER=SEQUENTIAL
