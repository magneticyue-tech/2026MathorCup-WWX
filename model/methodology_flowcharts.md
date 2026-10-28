# 问题 1 方法论流程图

依据 [q1_methodology.md](q1_methodology.md)，按数据准备、连续预测、等级映射和结果输出组织方法。

## A4矩形排版

![问题1方法论流程图](../figures/q1_methodology_flowchart.png)

成图采用黑白矩形框、直角箭头、宋体正文和侧边竖排分区标题，由XeLaTeX与TikZ编译。图宽158毫米、高169毫米，适配A4纵向纸张及25.4毫米页边距。正文9.2磅、行距11.2磅，框内边距1.6毫米，纵向框间距6毫米。

- [矢量PDF](../figures/q1_methodology_flowchart.pdf)
- [矢量SVG](../figures/q1_methodology_flowchart.svg)
- [LaTeX源文件](../paper/figures/q1_methodology_flowchart.tex)

在论文中使用`\includegraphics[width=\linewidth]{q1_methodology_flowchart.pdf}`插入。
重新生成命令：`python code/scripts/fig_q1_methodology_flowchart.py`。


图中Z表示附件4的标准化连续评分，折外连续预测由化学模型产生；L、A和R为构造参照，最终0–6输出为感官等级。品评团零点用于构造局部参照A；预测等级由连续预测与选定映射共同确定。

## 方法说明

1. **数据准备。** 按啤酒编号和附件3的维度定义整理输入；化学数据的NA以训练折非缺失观测的列均值填充。
2. **连续预测。** 每个感官维度拟合一个`HistGradientBoostingRegressor`。在内层交叉验证中选择树深、叶节点样本量、迭代次数、正则化及单调约束配置；类别级OAV和零膨胀维度的障碍模型作为候选扩展，按内层Spearman比较。
3. **参照构造。** 线性参照L以训练折逐维端点确定；局部参照A以组维零点和代表性品评员尺度确定，其中尺度为`3r/(1+r²)`，r为训练折该维有效零点中位数的相反数。混合参照R在有效局部单元采用A，其余采用L。
4. **映射选择。** 两种候选共用连续预测和外层划分。直接线性映射将连续预测缩放到0–6后取整；QWK切点映射在缩放后使用6个有序切点。以5次重复的50维平均外层QWK选择映射，指标相同时选直接线性映射。
5. **结果交付。** 现有250款啤酒采用5次折外连续预测的均值，并应用最终拟合的选定映射。全体数据重新拟合的连续模型用于新啤酒及Q3。等级评价同时报告全体、局部参照单元和线性参照单元的结果；尺度敏感性取`m=2.5`和`m=min(4, cap_d)`。
