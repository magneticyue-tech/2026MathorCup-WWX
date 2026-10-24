# 敏感性分析

记录哪些参数被扰动、测试范围以及观察到的对输出的影响。每条记录都链接到产生它的脚本和图表。

| 参数 | 基线 | 测试范围 | 输出影响 | 脚本 -> 图表 |
|-----------|----------|--------------|---------------|------------------|
| Q1 尺度参数 $m$，锚定维度 | 3 | 2.5；$\min(4,\text{cap}_d)$ | 预运行，观测到的 $z$ 经转换后：在 $m=3$ 与 $m=4$ 之间，22% 的锚定评分不同；在 2 与 4 之间为 39%（40 维集合）。模型输出 <!-- [PENDING] \| Await: Q1 formal run --> | `src/q1_methodology_checks.py` 检查 4 -> <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 无锚定尺度 | C-mid | C-quantile | <!-- [PENDING] \| Await: Q1 formal run --> 等级分布与变化等级占比 | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 零值脆弱维度（sweetness、A_hops_all、bitternes、F_malt_grain） | m 模型 | C-mid | <!-- [PENDING] \| Await: Q1 formal run --> | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 零值可疑维度（F_hops_noble、A_malt_bread、F_malt_bread、acidity、F_malt_grain、F_hops_tropical、lactic） | m 模型 | C-mid | <!-- [PENDING] \| Await: Q1 formal run --> | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 化学 NA 处理 | 各列在训练折内按非缺失观测的均值填充 | 训练折中位数填充 + 缺失指示变量；HistGBR 原生 NaN；8 个单化合物按最低位编码 + 指示变量且 `aroma_hops_sum` 保留 NaN；额外遮蔽 5% 和 10% 的测试折单元格，MCAR 和最低 30% MNAR | 同源 Nature 论文将 0.1% 的缺失化学测量按属性均值填充；附件 1 实际为 73/57,750=0.126%。模型输出 <!-- [PENDING] \| Await: Q1 formal run --> 各对照臂相对主设定的 50 维平均 Spearman 差及 95% CI；全部 ≤ 0.01 则记为对 NA 处理不敏感，否则并列报告受影响方案及评分变化超过 0.5 分的维度。主设定不按结果切换 | `src/q1_methodology_checks.py` 检查 2 -> <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 barnyard | 纳入 | 排除 | <!-- [PENDING] \| Await: Q1 formal run --> 42 维平均指标 | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 `_sum` 列 | 保留 | 删除 4 列 | <!-- [PENDING] \| Await: Q1 formal run --> Spearman 变化 | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 附件 1 与附件 8 的取值 | 附件 1 | 来自附件 8 的 24 款啤酒的 80 个单元格 | <!-- [PENDING] \| Await: Q1 formal run --> Spearman 变化 | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 会话并列阈值 $k$ | 3 | 2、5 | 预运行：对于 2–5 中的每个 $k$，有 40–41 个多啤酒组，最大组为 11。CV 指标 <!-- [PENDING] \| Await: Q1 formal run --> | `src/q1_methodology_checks.py` 检查 3 -> <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 单调约束 | 由嵌套 CV 选择 | 全部开启；全部关闭 | <!-- [PENDING] \| Await: Q1 formal run --> 6 个受约束维度上的 Spearman | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 数据划分 | 按会话分组 5 折 × 5 | 论文 70/30；留一种风格法 | 预运行，风格均值基线 $R^2$：留一法 0.181，留会话法 0.115。模型指标 <!-- [PENDING] \| Await: Q1 formal run --> | `src/q1_methodology_checks.py` 检查 5 -> <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q2 模块阈值 τ | 0.6 | 0.5；0.7 | 预运行：模块数 117 / 161 / 185，模块划分 bootstrap ARI 中位 0.66 / 0.76 / 0.85；主方法化合物层入选频率按位置在三个 τ 下均为 0.56、0.35、0.19。<!-- [PENDING] \| Await: Q2 formal run --> Top-3 与基线的化合物层和模块层 Jaccard、Top-1 改变的维度数 | `src/q2_methodology_checks.py` 检查 1、3 -> <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 候选集合 | 225 个（去除 5 个综合指标与内标 3-methyl-2-butenal） | 加回 5 个综合指标；加回内标 | 预运行：综合指标与其最相关成分的 \|ρ\| 为 0.78–0.97。<!-- [PENDING] \| Await: Q2 formal run --> Top-3 改变的维度数 | `src/q2_methodology_checks.py` 检查 1 -> <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 化学 NA 处理 | 列均值填充 | 单化合物 NA 最低位并列；逐对删除（仅边际关联） | 预运行：最低位并列与基线 Top-3 的化合物层 Jaccard 0.980，Top-1 全部相同。<!-- [PENDING] \| Await: Q2 formal run --> 逐对删除下附件 6 的 44 行复算误差 | `src/q2_methodology_checks.py` 检查 4 -> <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 附件 1 与附件 8 的取值 | 附件 1 | 来自附件 8 的 24 款啤酒的 80 个单元格 | <!-- [PENDING] \| Await: Q2 formal run --> Top-3 Jaccard、维度等级改变的维度 | <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 bootstrap 中的图 | 每次在重抽样本上重建 | 固定为附件 2 | <!-- [PENDING] \| Await: Q2 formal run --> 入选频率按位置的变化 | <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 bootstrap 单位 | 品评团簇（63 个团） | 单款啤酒 | <!-- [PENDING] \| Await: Q2 formal run --> 入选频率按位置的变化 | <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 感官目标 | 附件 4 的 $z$ | 39 个有锚维度改用减去本团零点的参考评分 | <!-- [PENDING] \| Await: Q2 formal run --> Top-3 Jaccard | <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |

## 说明
- 注明任何随机运行所用的种子。
