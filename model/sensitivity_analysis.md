# 敏感性分析

记录哪些参数被扰动、测试范围以及观察到的对输出的影响。每条记录都链接到产生它的脚本和图表。

| 参数 | 基线 | 测试范围 | 输出影响 | 脚本 -> 图表 |
|-----------|----------|--------------|---------------|------------------|
| Q1 训练目标 | 原始评分的啤酒均分 $\bar x$ | 每个训练折内估计的啤酒效应 $\alpha$ | 正式：等级 MAE 差 −0.0002 [−0.0034, +0.0028]，QWK 差 −0.0020 [−0.0064, +0.0029]；交付等级变化 5.2%，为各臂最大；`alpha_flag` 为 False | `code/scripts/solve_q1_sensitivity.py` → `q1_sensitivity.json`、`fig_q1_sensitivity.pdf` |
| Q1 残差池与离散规则 | 全局内层折外残差；预测分布中位数 | 近邻 50 款残差；四舍五入、分位数切点、QWK 切点和常数规则 | 正式：中位数减四舍五入等级 MAE −0.011 [−0.019, −0.003]，QWK −0.003 [−0.011, +0.003]；分位数切点 QWK 0.371、等级 MAE 0.578；近邻减全局等级 MAE +0.003 [−0.002, +0.007]，QWK −0.043 [−0.049, −0.035]；90% 区间覆盖 97.2%，平均宽度 1.97 级 | `code/scripts/solve_q1_scores.py`、`code/scripts/solve_q1_metrics.py` → `tab_q1_rules.tex` |
| Q1 化学 NA 处理 | 每个训练折的列均值填充 | 中位数加缺失指示、原生 NaN、8 个单化合物最低位加指示且 `aroma_hops_sum` 保留 NaN | 附件 1 缺失 73/57,750；正式等级 MAE 差：中位数加指示 −0.0003 [−0.0016, +0.0009]，原生 NaN −0.0001 [−0.0018, +0.0013]，最低位加指示 +0.0005 [−0.0013, +0.0026]；交付等级变化 1.4%–2.0% | `code/scripts/solve_q1_sensitivity.py` → `q1_sensitivity.json`、`fig_q1_sensitivity.pdf` |
| Q1 barnyard | 纳入 50 维报告，归因时标记 | 从汇总指标排除 barnyard，保留其余 49 维的原预测 | 只比较 49 维与 50 维汇总，等级不变；正式等级 MAE 49 维 0.4853、50 维 0.4794，QWK 0.295、0.290 | `code/scripts/solve_q1_sensitivity.py` → `q1_sensitivity.json` |
| Q1 四个 `_sum` 特征 | 保留 | 删除四列，沿用各折已选参数；Q3 G-S8 使用另拟合的全数据模型 | 正式等级 MAE 差 −0.0002 [−0.0017, +0.0011]，QWK 差 +0.0003 [−0.0019, +0.0022]；交付等级变化 1.8% | `code/scripts/solve_q1_sensitivity.py` → `q1_sensitivity.json`、`fig_q1_sensitivity.pdf` |
| Q1 附件 1 与 8 的冲突单元 | 附件 1 | 24 款啤酒的 80 个单元改用附件 8 | 正式等级 MAE 差 +0.0020 [+0.0006, +0.0032]，QWK 差 −0.0034 [−0.0063, −0.0007]，唯一 CI 不含 0 的臂，方向为附件 8 取值略差；交付等级变化 1.8% | `code/scripts/solve_q1_sensitivity.py` → `q1_sensitivity.json`、`fig_q1_sensitivity.pdf` |
| Q1 单调约束 | 每组由内层交叉验证选择开关 | 全部开启；全部关闭 | 正式等级 MAE 差 −0.0001 [−0.0004, +0.0002]、−0.0002 [−0.0005, +0.0002]；交付等级变化 0.09%、0.07% | `code/scripts/solve_q1_sensitivity.py` → `q1_sensitivity.json`、`fig_q1_sensitivity.pdf` |
| Q1 划分 | 日期组 5 折 × 5 次重复 | 留一风格；论文分层 70/30 只用于 B2 复现 | 留一风格合并等级 MAE 0.475，B1 0.547，差 −0.072 [−0.095, −0.047]；139 款测试啤酒与训练啤酒同属一个日期组，该划分不比主划分更严格；小样本风格只作定性解读；B2 50 维平均测试 R² 0.154（论文 0.189），不进入等级比较 | `code/scripts/solve_q1_stress.py`、`code/scripts/solve_q1_baselines.py` → `q1_stress.json`、`fig_q1_stress.pdf` |
| Q1 主模型调参网格 | `max_depth` {2,3}，`max_iter` {100,200,400}，叶节点 {10,20} | `max_depth` {1,2,3}，`max_iter` {10,25,50,100,200}，叶节点 {10,20,40}；重复 0 | 内层最优点由网格下边界移至内部（50–100 轮）；外层连续 MAE 0.4732 对 0.4733，R² 0.192 对 0.183，Spearman 0.333 对 0.352；不重算 | `experimentalCode/q1_raw_target/exp_grid_boundary.py` → `grid_boundary.json` |
| Q1 类别级 OAV 特征（可选） | 不加入 | 酯类、硫化合物 OAV 和 | 正式：内层改善 4/25 折，未达 13 折；等级 MAE 差 +0.0009 [+0.0001, +0.0018]；OAV 不纳入 | `code/scripts/solve_q1_oav.py` → `q1_oav.json` |
| Q2 模块阈值 τ | 0.6 | 0.5；0.7 | 预运行：模块数 117 / 161 / 185，模块划分 bootstrap ARI 中位 0.66 / 0.76 / 0.85；主方法化合物层入选频率按位置在三个 τ 下均为 0.56、0.35、0.19。<!-- [PENDING] \| Await: Q2 formal run --> Top-3 与基线的化合物层和模块层 Jaccard、Top-1 改变的维度数 | `src/q2_methodology_checks.py` 检查 1、3 -> <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 候选集合 | 225 个（去除 5 个综合指标与内标 3-methyl-2-butenal） | 加回 5 个综合指标；加回内标 | 预运行：综合指标与其最相关成分的 \|ρ\| 为 0.78–0.97。<!-- [PENDING] \| Await: Q2 formal run --> Top-3 改变的维度数 | `src/q2_methodology_checks.py` 检查 1 -> <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 化学 NA 处理 | 列均值填充 | 单化合物 NA 最低位并列（完整重跑）；逐对删除（只作边际相关诊断） | 预运行：最低位并列与基线 Top-3 的化合物层 Jaccard 0.980，Top-1 全部相同。<!-- [PENDING] \| Await: Q2 formal run --> 逐对删除报告边际 Spearman、有效样本数及附件 6 的 44 行复算误差；不报告 Top-3 比较 | `experimentalCode/q2_methodology_checks.py` 检查 4 -> <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 附件 1 与附件 8 的取值 | 附件 1 | 来自附件 8 的 24 款啤酒的 80 个单元格 | <!-- [PENDING] \| Await: Q2 formal run --> Top-3 Jaccard、维度等级改变的维度 | <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 bootstrap 中的图 | 每次在重抽样本上重建 | 固定为附件 2 | <!-- [PENDING] \| Await: Q2 formal run --> 入选频率按位置的变化 | <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 bootstrap 单位 | 日期组簇（25 个日期组） | 单款啤酒 | <!-- [PENDING] \| Await: Q2 formal run --> 入选频率按位置的变化 | <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q2 感官目标 | 原始评分均值 $\bar x$ 的秩 | 附件 4 的 $z$ 的秩 | <!-- [PENDING] \| Await: Q2 formal run --> Top-3 Jaccard | <!-- [PLACEHOLDER] \| Await: Q2 coding work order --> |
| Q3 风格内协方差 | 因子 + 模块块，$k=15$，$\kappa=2$ | 纯因子；Ledoit–Wolf | 预运行留出潜空间高斯评分 −240.7 / −244.9 / −267.1 nat/款（κ=2）；生成样本模块内 ρ 0.680 / 0.633（真实 0.724）。<!-- [PENDING] \| Await: Q3 formal run --> §6.2 各指标变化 | `experimentalCode/q3_methodology_checks.py` 检查 2、3 -> <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 因子数 $k$、收缩 $\kappa$ | 留出潜空间评分最优（嵌套选参） | 相邻网格点 | 预运行：κ=5 比 κ=2 低 0.1 nat；$k=10$、20 分别低 0.8、2.0 nat。<!-- [PENDING] \| Await: Q3 formal run --> | `experimentalCode/q3_methodology_checks.py` 检查 2 -> <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 风格方差 | 共享 $\Sigma_W$ | $D_s^{1/2}\Sigma_WD_s^{1/2}$；$D_{s,jj}=((n_s-1)v_{sj}+10v_{Wj})/((n_s-1+10)v_{Wj})$，$n_s\le1$ 或 $v_{Wj}=0$ 时为 1 | 预运行风格离散比：Brut 1.73，其余 0.72–1.34。<!-- [PENDING] \| Await: Q3 formal run --> | `experimentalCode/q3_methodology_checks.py` 检查 3；参数定义见 `q3_methodology.md` §6.7 -> <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 并列块的潜变量 | 截断正态条件期望 | 平均秩的正态得分 | <!-- [PENDING] \| Await: Q3 formal run --> 下限比例误差、模块内 ρ | <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 化学 NA | 列均值填充 | 单化合物最低位编码 | <!-- [PENDING] \| Await: Q3 formal run --> 生成指标；双乙酰 NA 的 2 款 Pils/Lager 的迁移上界 | <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 附件 1 与附件 8 的取值 | 附件 1 | 来自附件 8 的 24 款啤酒的 80 个单元格 | <!-- [PENDING] \| Await: Q3 formal run --> | <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 的四个 `_sum` 特征 | 主设定由潜空间岭回归派生后送入 Q1 模型 | 使用 Q1 删除四列的全数据模型；仍派生并使用 `kcalperc` | 预运行派生 5 折 $R^2$ 0.78–0.94。<!-- [PENDING] \| Await: Q3 formal run --> 画像与 $\alpha^*$ 变化 | `experimentalCode/q3_methodology_checks.py` 检查 4 -> <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 内标 | 训练中位数 | 从经验分布独立抽取 | <!-- [PENDING] \| Await: Q3 formal run --> 画像变化 | <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 缺陷阈值口径 | 附件 8 均值 | 附件 8 中位 | 预运行全体超阈比例：双乙酰 0.45 → 0.69，DMS 0.67 → 0.62。<!-- [PENDING] \| Await: Q3 formal run --> $\alpha^*$、约束起作用比例 | `experimentalCode/q3_methodology_checks.py` 检查 5 -> <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 化学层规则 | $x'\le\max(x,\theta)$ | $x'\le x$ | <!-- [PENDING] \| Await: Q3 formal run --> $\alpha^*$、目标风格概率 | <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 感官层规则 | 预测等级不升高 | 连续预测 $\hat\mu$ 不升高；不设感官层 | 旧附件 4 代理预运行：$\alpha^*$ 中位 0.70 / 0 / 1（无感官层为投影终点）；旧预测等级规则与现行 $\hat\mu$ 规则定义不同，数值不作直接比较。<!-- [PENDING] \| Await: Q3 formal run --> | `experimentalCode/q3_methodology_checks.py` 检查 7 -> <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |
| Q3 问题 1 映射 | 预测分布中位数 | 分位数切点等级 $\hat G^{\rm Q}$ | <!-- [PENDING] \| Await: Q3 formal run --> 画像等级分布、$\alpha^*$ | <!-- [PLACEHOLDER] \| Await: Q3 coding work order --> |

## 说明
- 注明任何随机运行所用的种子。
- 上述 Q1 行采用 `q1_methodology.md` 的原始评分目标。此前基于附件 4 的锚定尺度 $m$、C-mid 与 C-quantile 属于旧方法，不与当前 Q1 主线结果合表。
