# 敏感性分析

记录哪些参数被扰动、测试范围以及观察到的对输出的影响。每条记录都链接到产生它的脚本和图表。

| 参数 | 基线 | 测试范围 | 输出影响 | 脚本 -> 图表 |
|-----------|----------|--------------|---------------|------------------|
| Q1 尺度参数 $m$，锚定维度 | 3 | 2.5；$\min(4,\text{cap}_d)$ | 预运行，观测到的 $z$ 经转换后：在 $m=3$ 与 $m=4$ 之间，22% 的锚定评分不同；在 2 与 4 之间为 39%（40 维集合）。模型输出 <!-- [PENDING] \| Await: Q1 formal run --> | `src/q1_methodology_checks.py` 检查 4 -> <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 无锚定尺度 | C-mid | C-quantile | <!-- [PENDING] \| Await: Q1 formal run --> 等级分布与变化等级占比 | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 零值脆弱维度（sweetness、A_hops_all、bitternes、F_malt_grain） | m 模型 | C-mid | <!-- [PENDING] \| Await: Q1 formal run --> | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 零值可疑维度（F_hops_noble、A_malt_bread、F_malt_bread、acidity、F_malt_grain、F_hops_tropical、lactic） | m 模型 | C-mid | <!-- [PENDING] \| Await: Q1 formal run --> | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 化学 NA 编码（缺失性压力测试） | 删失：0 + 指示变量 | 均值插补；原生 NaN；额外遮蔽 5% 和 10% 的测试折单元格，MCAR 和最低 30% MNAR | <!-- [PENDING] \| Await: Q1 formal run --> 42 维平均 Spearman 降幅；若删失编码在 MNAR 下优于均值插补，且在 MCAR 下损失 ≤ 0.01，则保留删失编码 | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 barnyard | 纳入 | 排除 | <!-- [PENDING] \| Await: Q1 formal run --> 42 维平均指标 | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 `_sum` 列 | 保留 | 删除 4 列 | <!-- [PENDING] \| Await: Q1 formal run --> Spearman 变化 | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 附件 1 与附件 8 的取值 | 附件 1 | 来自附件 8 的 24 款啤酒的 80 个单元格 | <!-- [PENDING] \| Await: Q1 formal run --> Spearman 变化 | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 会话并列阈值 $k$ | 3 | 2、5 | 预运行：对于 2–5 中的每个 $k$，有 40–41 个多啤酒组，最大组为 11。CV 指标 <!-- [PENDING] \| Await: Q1 formal run --> | `src/q1_methodology_checks.py` 检查 3 -> <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 单调约束 | 由嵌套 CV 选择 | 全部开启；全部关闭 | <!-- [PENDING] \| Await: Q1 formal run --> 6 个受约束维度上的 Spearman | <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |
| Q1 数据划分 | 按会话分组 5 折 × 5 | 论文 70/30；留一种风格法 | 预运行，风格均值基线 $R^2$：留一法 0.181，留会话法 0.115。模型指标 <!-- [PENDING] \| Await: Q1 formal run --> | `src/q1_methodology_checks.py` 检查 5 -> <!-- [PLACEHOLDER] \| Await: Q1 coding work order --> |

## 说明
- 注明任何随机运行所用的种子。
- 用数值或图表引用报告影响，不要用形容词。
- 将等待求解器运行的行标记为 `[PENDING]`，并注明可清除该状态的工作单任务或脚本。
- 敏感性分析扰动固定假设以评估稳健性。它不同于样本外验证（后者属于方法学文档），也不能替代样本外验证。
