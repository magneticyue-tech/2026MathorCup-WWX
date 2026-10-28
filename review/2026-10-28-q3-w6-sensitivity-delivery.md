---
id: 2026-10-28-q3-w6-sensitivity-delivery
from: modeling
to: coding
status: open
severity: major
refs:
  - review/2026-10-28-q3-index.md
  - review/2026-10-28-q3-w2-generator-selection.md
  - review/2026-10-28-q3-w3-generator-validation.md
  - review/2026-10-28-q3-w4-portrait.md
  - review/2026-10-28-q3-w5a-chem-transfer.md
  - review/2026-10-28-q3-w5b-sensory-path.md
  - model/q3_methodology.md
  - model/sensitivity_analysis.md
  - memory/attachment1-vs-8-conflict.md
  - memory/na-below-detection.md
---

## 前置条件（未满足时不得开始）

W2、W3、W4、W5a、W5b 全部 resolved。G-S8 另依赖 Q1 W7 交付 `code/data/processed/q1/q1_models_drop_sum/` 与 `q1_drop_sum_final.json`；该臂在产物存在前单独等待，其余不依赖它的敏感性臂可先做。

## 问题

Q3 的敏感性分析（方法论 §6.7）、图、表和交付文件尚未实现。
本工单按臂逐个运行，每个臂与正式基线比较，并生成论文用图表与交付 CSV。

## 背景 / 复现

读方法论 §6.7，以及 `model/sensitivity_analysis.md` 中以 Q3 开头的 12 行；运行后每行替换占位符。
每个臂只改动一项，其余沿用基线；基线为 W2 的 final_config、W3 的主模型指标、W5b 的主设定输出。臂的结果单独成文件，单个臂失败后只重跑该臂。

## 敏感性臂

### 生成器（比较 W3 的 §6.2 各指标，用与基线相同的外层划分与种子）

| 臂 | 改动 | 实现要点 |
|---|---|---|
| G-S1 | 协方差：纯因子 | 即 W3 的纯因子生成器，直接引用，不重跑 |
| G-S2 | 协方差：Ledoit–Wolf | `StyleGen(kind=lw)`，κ 取基线选定值 |
| G-S3 | k、κ 取相邻网格点 | 对 final_config 的 k 与 κ 各取网格上下相邻值，共最多 4 个设定，k、κ 不同时改变 |
| G-S4 | 风格方差：专属对角缩放 | 按方法 §6.7 从外层训练数据估计 $v_{sj}$ 与 $v_{Wj}$，以固定 10 个伪自由度计算 $D_{s,jj}=((n_s-1)v_{sj}+10v_{Wj})/((n_s-1+10)v_{Wj})$；$n_s\le1$ 或 $v_{Wj}=0$ 时 $D_{s,jj}=1$。用 $D_s^{1/2}\Sigma_WD_s^{1/2}$ 替换共享协方差；沿用基线的 $k$、$\kappa$、结构与随机种子，不另选参。 |
| G-S5 | 并列块潜变量：平均秩正态得分 | `tie_scheme=mean_rank`；用于 `Marg`、W3 与 W5 的全部下游 |
| G-S6 | 化学 NA：单化合物最低位 | 8 个单化合物 NA 的占位值改为该列非缺失观测的最小值，不用列均值；`Marg` 仍只用非缺失观测构造逆分布；对 Pils/Lager → Tripel 逐款报告含双乙酰 NA 的 2 款的上界、α*、双乙酰是否为有效约束、感官 diacetyl 等级与结论类别，与主设定并列 |
| G-S7 | 附件 1 与附件 8 取值 | 用冲突清单替换 24 款的 80 个单元格：优先读 Q2 敏感性工单（`2026-10-28-q2-sensitivity-runs`）保存的可追溯替换表（`code/data/processed/q2/` 下，先列目录确认文件名）；不存在时，用 `experimentalCode/results/data_audit/calibrated_discordances.csv` 与附件 8 构造并保存替换表，断言 24 款、80 个单元格；不改原始附件 |
| G-S8 | Q1 输入删除四个 `_sum` 特征 | 读取 Q1 W7 的 `q1_models_drop_sum/`（全数据、227 列模型）及 `q1_drop_sum_final.json`；输入仅保留清单中的 227 列，复用其编码器和 50 个模型，不重拟合或选参。Q3 仍按主设定派生 `kcalperc` 并传入模型。依方法 §3.5 的同一 Q3 画像与迁移流程，比较画像与 α*。模型缺失时只标记该臂未运行并报告 Q1 W7 依赖，不临时训练、不等待 Modeling 决策。 |
| G-S9 | 内标：从经验分布独立抽取 | 每个合成样本的内标从训练啤酒的内标观测值中有放回独立抽取；比较画像 |

### 迁移（比较 α* 的中位数、约束起作用比例、目标风格概率中位数相对基线的变化，范围为 Pils/Lager → Tripel 的 11 款与 22×21 个风格对）

| 臂 | 改动 | 实现 |
|---|---|---|
| T-S1 | 迁移中的综合指标：只用岭回归预测 | `transfer_input231` 的综合指标不加残差；同时报告 11 款的感官预测变化 |
| T-S2 | 候选选择：α_fit | 基线运行已记 α_fit，直接汇总，不重跑 |
| T-S3 | 阈值口径：中位 | `solve_q3_transfer_chem.py --thr-kind median`，再 `solve_q3_transfer.py --chem-suffix _median` |
| T-S4 | 化学层规则：严格不增加 | `--rule strict`，同上 |
| T-S5a | 感官层规则：连续预测不升高 | `solve_q3_transfer.py --rule mu_cont` |
| T-S5b | 不设感官层 | `--rule none`，此时 α* 为 1 |
| T-S6 | Q1 映射：分位数切点 | `--rule grade_quant` |
| T-S7 | NA、冲突单元格 | 取 G-S6、G-S7 的生成器重拟合后重跑迁移，只对 11 款 |

## 文件与脚本

- `code/src/q3/sensitivity.py`：`ARMS`（臂的 id、改动、依赖的上游脚本、比较指标）；`compare_generator(base, arm, bootstrap_indices_all, bootstrap_indices_by_style) -> dict`（对每项 §6.2 标量指标复用 W3 保存的对应啤酒重抽索引，返回基线值、臂值、配对差及其 95% percentile 区间；逐风格指标另返回每风格值，以及宏平均、按全数据样本量加权平均各自的配对差区间）；`compare_transfer(base, arm) -> dict`（α* 中位数、约束起作用比例、P_t 中位数的差值；逐款 α* 变化的款数）。
- `code/scripts/solve_q3_sensitivity.py`：`--arm <id>`、`--list`、`--smoke`；每个臂写 `code/data/processed/q3/sens/<id>.npz` 与 `<id>.json`；已存在且 `--force` 未给出时跳过。
- 汇总文件 `code/data/processed/q3/q3_sensitivity.csv`：每臂一行，含基线、改动、各指标的基线值、臂值、配对差及差值 95% 区间、依赖的输入文件 SHA-256 前 12 位。区间使用 W3 的同一啤酒重抽索引，在每次重抽中对基线和敏感性臂配对计算差值。

## 图、表与交付文件

绘图脚本只读 `code/data/processed/q3/` 下的 npz 与 CSV，不调用任何求解函数；图写入 `figures/q3/`，脚本名与产物名一致（`.pdf`）。所有图使用可嵌入 PDF 的字体、明确的轴名与单位；22×22 热图的风格顺序在两个面板一致，缺失或不适用单元单独标记，不以零着色。生成后检查图例、文字和坐标刻度不裁切。

| 脚本 `code/scripts/` | 产物 | 内容 |
|---|---|---|
| `fig_q3_select_heatmap.py` | `fig_q3_select_heatmap.pdf` | 内层 L 随 k 与 κ 的热图，标出 final_config；外层各折选定设定的频数 |
| `fig_q3_c2st.py` | `fig_q3_c2st.pdf` | 各生成器的合并与条件 AUC，置换零分布 95% 分位带，真实对真实线 |
| `fig_q3_gen_metrics.py` | `fig_q3_gen_metrics.pdf` | DCR 比、近复制率、TSTR 对 TRTR、下限比例误差 |
| `fig_q3_corr.py` | `fig_q3_corr.pdf` | 相关误差、风格内相关误差、模块内 ρ，含 G1 与真实参照线 |
| `fig_q3_portrait.py` | `fig_q3_portrait.pdf` | Tripel、Kriek、Lambic 的画像：50 维的中位数与 10%–90% 区间，叠加真实啤酒预测 |
| `fig_q3_fingerprint.py` | `fig_q3_fingerprint.pdf` | 三个风格的 δ 前 10 项及 90% 区间 |
| `fig_q3_portrait_val.py` | `fig_q3_portrait_val.pdf` | 逐风格 Spearman 对样本量，生成对真实预测与真实预测对观测 |
| `fig_q3_transfer_alpha.py` | `fig_q3_transfer_alpha.pdf` | Pils/Lager → Tripel 11 款的可行路径与 α*、α_fit |
| `fig_q3_transfer_pt.py` | `fig_q3_transfer_pt.pdf` | 源、T0、T1、T4、T5、主方法、T2 的 P(Tripel) 与真实 Tripel 参照范围 |
| `fig_q3_pairs.py` | `fig_q3_pairs.pdf` | 22×22 风格对热图：α* 中位数与迁移后 P_t 中位数 |
| `fig_q3_sensitivity.py` | `fig_q3_sensitivity.pdf` | 各臂相对基线的指标变化 |
| `tab_q3_select.py` | `tab_q3_select.tex`、`.csv` | 选参结果与对照网格 |
| `tab_q3_genval.py` | `tab_q3_genval.tex`、`.csv` | §6.2 全部指标 × 生成器，含验收判定 |
| `tab_q3_transfer.py` | `tab_q3_transfer.tex`、`.csv` | 11 款逐款迁移结果与对照 |
| `tab_q3_pairs.py` | `tab_q3_pairs.tex`、`.csv` | 读 W5b 的 `q3_transfer_pairs.csv`（462 个有序风格对）；CSV 为按源风格名称排序的 22 行论文汇总，列为 `src`、`n_src`、`alpha_star_mean_median`（21 个 `alpha_star_mean` 的中位数）、`p_main_median_across_targets`（21 个 `p_main_median` 的中位数）、`frac_t0_chem_viol_mean`、`frac_alpha0_mean`、`frac_ref_mean`、`frac_improved_mean`、`frac_unchanged_mean`、`frac_no_improve_mean`（后六列均为相应逐对比例的算术平均）。TeX 第一表取 CSV 前六列，第二表取 `src` 与四个结论比例列，生成两张相同源风格顺序的 22 行表，不排 462 行长表。每个源风格断言恰有 21 个不同目标风格，`n_src` 在各目标行相同；21 个目标等权。两个表注说明汇总层级、分母、位置量与比例的不同聚合方式；`q3_transfer_pairs.csv` 作为完整机器可读补充文件。 |
| `tab_q3_sensitivity.py` | `tab_q3_sensitivity.tex`、`.csv` | 敏感性汇总 |
| `export_q3_outputs.py` | `code/data/processed/q3/q3_portraits_long.csv` | 22 个风格 × 50 维：`style`、`dim`、`mu_q10`、`mu_q50`、`mu_q90`、`mode`、`pi0`…`pi6`、`pi_tilde0`…`pi_tilde6`、`weak_dim`、`barnyard_flag` |

另交付 22 个风格的合成化合物谱：`code/data/processed/q3/q3_synthetic_profiles.npz`（每风格 1,000×225，原尺度，附列名与风格名）；不写 CSV。

## 验收标准

通用：`black --check code/` 与 `ruff check code/` 通过；`src/q3/` 无 `__main__`；每个求解脚本写 `seed` 与 `meta`。
- 删除 `figures/q3/` 后，只运行 `fig_*`、`tab_*`、`export_*` 即可重新生成全部图表，不触发求解函数；
- `tab_q3_pairs.csv` 恰为 22 行，两个 TeX 表各含 22 个源风格且按同一顺序；逐项由 462 行 `q3_transfer_pairs.csv` 复算，四类比例在每个源风格上和为 1（容差 `1e-9`）。TeX 编译并逐页检查没有裁切、重叠或缺字；462 行只保留在补充 CSV。
- 每个臂有独立结果文件，`q3_sensitivity.csv` 每臂一行，没有 NaN 伪装完成（不适用的指标写 NaN 并在 `note` 列写原因）；
- 同一臂固定种子的冒烟结果两次运行逐项一致；
- G-S6 与 G-S7 的替换表保存，G-S7 恰为 24 款、80 个单元格；
- `model/sensitivity_analysis.md` 的 Q3 各行更新：基线、对照设置、结果、脚本与结果文件链接；不留 Q3 的 PENDING 与 PLACEHOLDER；G-S4 按 §6.7 完成并报告，不留待决策项；
- 回报中列出：每个臂的运行命令与耗时；图表与 npz 的对应关系；G-S6 下含双乙酰 NA 的 2 款 Pils/Lager 的并列结果。

## 处理结果
<!-- [PLACEHOLDER] | Await: 前置条件满足后 coding 回复 ack 并逐项回报 -->
