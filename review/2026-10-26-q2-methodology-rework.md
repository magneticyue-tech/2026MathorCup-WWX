---
id: 2026-10-26-q2-methodology-rework
from: modeling
to: coding
status: open
severity: major
refs:
  - model/q2_methodology.md
  - review/2026-10-25-q2-rationality-workorder.md
  - code/src/q2/config.py
  - code/src/q2/data.py
  - code/src/q2/association.py
  - code/src/q2/inference.py
  - code/src/q2/stability.py
  - code/src/q2/full_controls.py
  - code/src/q2/controls_sensitivity.py
  - code/src/q2/simulation.py
  - code/src/q2/rationality.py
  - code/data/processed/q2/
  - figures/q2/
---

## 问题
2026-10-26 的代码和输出仍未完整落实 `model/q2_methodology.md`；本工单扩展并承接 `2026-10-25-q2-rationality-workorder.md`，以下验收以当前方法文档为准。
`association.py`、`inference.py`、`stability.py`、`full_controls.py` 与 `controls_sensitivity.py` 仍按附件 4 的 50 个维度独立处理，结果将 barnyard 作为独立维度输出；方法要求交付 49 个维度，并将 barnyard 注明为“同 stale_hops”。
`stability.py` 已重算 bootstrap 样本内秩并输出区间，但区间只记录化合物被选中时的统计量，且第 2、3 位未固定全样本已选集合；这不是 §4.4、§5.1 定义的固定化合物边际或给定固定 $S$ 的偏相关区间。
`stability.py` 的 bootstrap 图只由 225 个候选列重建；方法规定重建附件 2 同源的图和模块，图有 231 个节点（候选与注释节点）。
`inference.py` 只对感官秩逐行随机打乱，并将边际 $q_d$ 用于全部三个位置；第 2、3 位须按 §4.4 对残差实施 Freedman–Lane 置换检验。
`full_controls.py` 的 M2/M3 和风格内结果仍含 50 个维度；M2 使用未分组 CV，M3 使用普通 `KFold`，且输出只有 Jaccard 摘要，不能验收 §5.2/§5.3 的逐维 Top-3 与 Top-1 对照。
`rationality.py` 计算了风格内 Fisher 区间但将 `flag_style_ci0` 固定为空，导致 §6.2 的第一项共变判据没有进入 `covariation`；请按 §4.6 输出定义该字段，若区间方法需要模型裁决，先在本线程报告实现口径与结果，不得以空字段冒充已检验。
当前主结果 `final_top3_with_rationality.csv` 有 177 行且覆盖 50 个维度；需按 49 个维度生成主方法 Top-3，并仅为主方法与 M3 Top-1 不同的维度增加 M3 行。旧 `controls_sensitivity_results.csv`、`full_controls_results.csv` 仍为 50 行，须随正式重跑更新。

## 背景 / 复现
当前路径配置已将 Q2 输出指向 `code/data/processed/q2`，原始附件位于 `data/raw`；本工单不要求更改为不存在的目录。确保所有入口脚本使用同一配置，且失败时明确报错，不得用零分或空结果静默补齐。

2026-10-26 检查到：
- `final_top3_with_stability.csv` 与 `top3_results_with_sig.csv` 各 150 行、包含 50 个维度；前者已有非零得分和 `r_ci_lower`/`r_ci_upper`，但尚未证明区间对应方法要求的固定估计量。
- `final_top3_with_rationality.csv` 有 177 行、包含 50 个维度；检查 6 的 `flag_style_ci0` 为空，旧对照文件均为 50 行。
- 图输出有 231 个节点，候选间边 245 条，另有涉及 6 个注释节点的边；图输出目录最近更新于 2026-10-26。
- `config.py` 已设置 `N_BOOTSTRAP = 1000`、`N_PERMUTATION = 1000`，但实际循环和检验仍须使用配置值。

## 返工要求

### 1. 维度与主结果
- 按 `memory/barnyard-stale-hops-duplicate.md` 处理 barnyard，不生成独立的 Top-3，不将其计入 breadth、等级、Jaccard 或化学合理性统计；在 stale_hops 的交付行注明“同 barnyard”。输入维度列仍可保留，不得静默改写原始数据。
- 49 个交付维度每个主方法恰有 3 行。结果符号与得分来自实际选择统计，不允许对缺失字段填 0 或以陈旧文件补齐。
- 保留 §4.5 的等级规则：Top-1 显著且 `pi_c >= 0.6` 为 A；显著但不满足该频率为 B；不显著为 C。等级需在显著性及 bootstrap 结果更新后生成。

### 2. 置换与 bootstrap
- `N_PERMUTATION`、`N_BOOTSTRAP`、`SEED` 均从 `config.py` 读取。
- 第 1 位按 §4.4 的边际最大统计量置换得到 `q_d`。第 2、3 位分别固定此前全样本已选集合 $S$，按 Freedman–Lane 打乱 $e_y$，在当轮允许入选的候选中计算最大绝对偏相关的零分布及阈值。
- 品评团簇 bootstrap 每次有放回抽取 63 团；按 §2.2 在重抽样本内重新处理化学缺失值及秩，重算附件 2 同口径的逐对 Spearman 图，并在 231 节点图上重建模块后重跑选择。
- 对固定的全样本入选项计算区间：第 1 位重采样其边际 $r_{jd}$；第 2、3 位固定全样本对应的已选集合 $S$ 后计算 $r_{jd\mid S}$。不得只在该化合物进入某次 bootstrap Top-3 时记录统计量。记录 `pi_c`、`pi_m`、Jaccard、区间及双重显著性。

### 3. 对照、敏感性与模拟
- M2 的 CV 按品评团分组；M3 按 §5.2 使用问题 1 的嵌套选参结果和 5×5 外层评估，在外层测试折计算 `n_repeats=5` 的置换重要度。不得用普通 `KFold` 替代分组划分。
- 逐维保存主方法、M0、M1、M2、M3 和风格内 Top-3、模块及 Jaccard；主结果表额外列出主方法与 M3 Top-1 不同维度的 M3 Top-1 行。
- 按 §7.2 正式运行 1,000 次模拟；按 §7.3 重跑全部指定敏感性，报告 49 维化合物层/模块层 Jaccard、Top-1 改变数和等级改变数。输出不得包含独立 barnyard 维度。

### 4. 化学合理性表
- 完成并验证 `2026-10-25-q2-rationality-workorder.md` 中的附件 8 标签、阈值/OAV、breadth、输出列和不使用事后预期表等要求；该旧工单中“barnyard 保留独立行”的表述按当前方法文档修正为本工单第 1 节。
- `flag_style_ci0` 对应 §4.6 风格内关联 95% CI 是否包含 0；说明并固定区间算法。它与 `flag_breadth5`、`flag_oav_lt1` 做逻辑或生成 `covariation`。附件 8 缺失标签保留为空；无阈值 OAV 保留 NaN。
- `interpretation` 保持空白，交由 Modeling 依据 §6.4 在结果产生后撰写。代码中不得恢复维度到化合物的预期映射、命中率、事后分类或来源富集。
- 最终表包含 49×3 个 `main` 行及所有需要并列报告的 M3 Top-1 行；等级按维度一致地写入。结果字段至少符合旧工单规定的列，并保留得分、区间、稳定性及各对照证据所需字段。

### 5. 验收与交付
- `black --check code/src/q2`、`ruff check code/src/q2` 通过。
- 验证附件 2 与附件 1 的逐对 Spearman 复算误差；逐对删除敏感性下附件 6 的 44 行逐元素复现，误差不超过 $10^{-12}$。
- 验证图节点为 231、候选间阈值边为 245（注释节点边另报），主输出为 49 个维度且每维 3 个主方法行，barnyard 不独立出现。
- 验证合理性表标签与 `chemical_dictionary.csv` 逐项一致；OAV 与检查 6 的同名化合物结果一致；breadth 统计不计 barnyard；A/B/C 数量及维度名单可由最终显著性与 `pi_c` 复算。
- 汇报运行命令、种子、配置次数、各产物路径和验收数值。完整重跑前检查目标文件和目录，避免把结果写入根目录另一份副本。

## 处理结果
