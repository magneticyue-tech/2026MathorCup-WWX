---
id: 2026-10-28-q3-w3-generator-validation
from: modeling
to: coding
status: open
severity: major
refs:
  - review/2026-10-28-q3-index.md
  - review/2026-10-28-q3-w2-generator-selection.md
  - model/q3_methodology.md
  - experimentalCode/q3_methodology_checks.py
---

## 问题

Q3 缺少生成器验证（方法论 §6.2）的实现。
本工单实现各项指标、置换零分布、真实对真实参照、阳性对照、逐风格检验和验收判定，产出 `q3_genval.npz`。
本工单依赖 W1、W2 及 `q3_select.npz`，不依赖 Q1。

## 背景 / 复现

读方法论 §3.4、§6.1、§6.2。参考实现为检查 3，第 248–328 行。
预检查只实现部分指标，选参与报告共用同一划分，其数值只作回归目标，不构成验收。

## 文件与函数

### T1 `gen_metrics.py`

所有函数输入输出均为 ndarray 或 dict，不读写文件。
- `make_rf()`：`RandomForestClassifier(500, random_state=SEED, n_jobs=1)`。
- `c2st(real, gen, style_real=None, styles=None, seed=SEED, n_splits=5) -> dict`：堆叠 `real` 与 `gen`，标签 0 与 1，`gen` 第 i 行与 `real` 第 i 行成对；`style_real` 不为 None 时，两部分都追加 22 列风格独热编码，`gen` 的风格取与之成对的真实啤酒的风格；`StratifiedKFold(n_splits, shuffle=True, random_state=seed)`；返回 `auc`（各折 `roc_auc_score` 的平均）、`oof`（`cross_val_predict(method=predict_proba)` 的折外概率）、`y`。
- `c2st_perm(real, gen, style_real, styles, n_perm=200, seed=SEED) -> ndarray[n_perm]`：每次对堆叠后的标签做 `rng.permutation` 并重算 `auc`。`pvalue(obs, null) = (1 + sum(null >= obs)) / (len(null) + 1)`；95% 分位取 `np.quantile(null, 0.95)`。
- `real_vs_real(real, style_real, styles, seed)`：用 `default_rng(seed)` 把留出真实啤酒随机分成两半，对两半做同一检验，合并与条件两种都做。
- `dcr(U_query, U_train) -> ndarray`：欧氏距离，对训练啤酒取最小。`dcr_ratio(d_gen, d_real) = median(d_gen) / median(d_real)`。`near_copy_rate(d_gen, d_real) = mean(d_gen < quantile(d_real, 0.05))`。潜变量一律用该折主模型的 `latent`。
- `rf_style_accuracy(X_fit, s_fit, X_eval, s_eval) -> ndarray[bool]`：TRTR、TRTS、TSTR 共用，返回逐款是否判对。
- `floor_metrics(real, gen, mins) -> (err, row_median)`：`mins` 为 250 款非缺失观测的列最小值（描述性，不参与选参）；`has_floor` 为留出真实啤酒中处于最小值的比例大于 0.02 的列；`err` 为这些列上生成与真实下限比例之差的绝对值平均；`row_median` 为每个样本处于下限的列数的中位数，只数 `has_floor` 列。下限判断用 `np.isclose(x, mins)`。
- `corr_error(real, gen) -> float`：两者各算 225×225 的 Spearman 矩阵（NaN 置 0），取上三角逐元素绝对差的平均。
- `within_style_corr_error(U_real, s_real, U_gen, s_gen) -> float`：各自减去所属风格的样本均值后算 Spearman 矩阵，其余同上。
- `module_rho(corr, labels) -> float`：同一模块内化合物对的平均 Spearman；`labels` 为该折使用的模块。
- `dispersion(U_te, s_te, mu, C) -> DataFrame`：每款留出啤酒的 (1/p)·Σ_j (u_j − mu_sj)² / C_jj；按风格取平均，再除以 22 个风格平均值的中位数得 `ratio`；`flag = max(ratio, 1/ratio) > 1.5`。
- `boot_by_beer(metric_fn, data_by_rep, bootstrap_indices_all, bootstrap_indices_by_style)`：正式运行预先生成 2,000 组全样本啤酒 ID 索引，所有汇总指标与生成器共用；逐风格指标使用各风格内固定样本量的独立有放回索引。W6 复用对应索引作配对差。每组索引内，在每次重复重算指标，再对 5 次结果取平均。2,000 个均值的 2.5% 与 97.5% 分位为 95% 区间。C2ST/真实对真实 AUC 用对齐的折外概率与标签，按啤酒配对重抽并重算 AUC；相关误差和模块内 $\rho$ 在抽样行上重算完整相关矩阵，不重抽矩阵元素；下限与离散指标按定义重算。所有 §6.2 标量指标均报告区间；另报 5 次重复的均值与极差。区间条件于已拟合模型与划分，不解释为训练或选参不确定性。
- 聚合顺序：每次重复先合并 5 个外层折的留出行，确保 250 款啤酒各有一行，再计算汇总和逐风格指标；不得先平均折指标。`bootstrap_indices_all` 的值是 0–249 的啤酒行号，所有生成器、真实对照和重复共用同一行号向量。`bootstrap_indices_style_XX` 的值是该风格 `style_beer_ids_XX` 排序数组中的位置。逐风格重抽后重算各风格结果，再计算等权宏平均和按全数据风格样本量加权平均。W6 对基线与臂共用同一组索引，先逐次计算配对差，再对 2,000 个差值取 percentile 区间；两种逐风格汇总分别计算配对差区间。
- `acceptance(summary) -> dict[str,bool]`：实现下文验收标准中的 (a)–(e)。

### T2 `code/scripts/solve_q3_genval.py`

写出 `q3_genval.npz` 与 `q3_genval.json`。命令行：`--stage gen|metrics|merge|all`、`--smoke`、`--preset prerun`。正式 `q3_genval.npz` 保存 `style_beer_ids_00`…`style_beer_ids_21`（各风格内按行号排序的啤酒 ID）、`bootstrap_indices_all`（2000×250，啤酒行号 0–249）及 `bootstrap_indices_style_00`…`style_21`（各为 2000×该风格啤酒数，值为对应 `style_beer_ids` 中的位置索引），索引由 `default_rng(SEED)` 生成；另保存每项标量指标 2,000 个 bootstrap 均值，供 W6 复算配对差区间。冒烟产物用 20 次并写入 `_smoke/`，不可供正式比较。
- `gen`：对每个 (rep, fold)，读 `q3_select.npz` 的选定设定，在外层训练集上拟合主模型（选定设定）、纯因子（kind 取 fa，k、κ 同主模型）、G0（`global_only=True`，kind、k 同主模型）。选定 kind 已是 fa 时，纯因子与主模型相同，记 `pure_equals_main = True`。对每个外层测试啤酒，每个生成器抽 1 个同风格样本（成对）；G1、G2、G3 同。再对训练集中每个风格抽取与该风格训练啤酒数相同个数的样本，训练随机森林并判留出啤酒（TSTR）。随机数用 `np.random.default_rng([SEED, rep, fold, 生成器序号])`。每个重复结束写 `_chunks/genval/pool_<rep>.npz`，内容：250 款真实啤酒、各生成器的配对样本、风格、潜变量、该折模块标签、各折的 `mu` 与 `C`、逐款 TRTR、TRTS、TSTR 判对结果。
- `metrics`：每个 (rep, 生成器, 变体) 一个单元，变体为 marginal 与 conditional。算 AUC、200 次置换零分布与 p 值；marginal 单元另算其余指标。真实对真实参照按重复计算；G0 与 G2 作阳性对照，按重复统计 AUC 超过置换 95% 分位的比例。逐风格 C2ST：每个风格 `n_splits = min(5, n_s)`，`n_s ≤ 5` 的风格标 `evidence_limited`；逐风格置换零分布只对主模型做，200 次；其余生成器的逐风格只报 AUC。
- `merge`：对 5 个重复取平均与极差，写 npz 与 json，并写 `acceptance` 及其依据数值。`q3_genval.json` 中每个 §6.2 标量指标含 `estimate`（5 次重复均值）、`ci95`（`lo`、`hi`）、`repeat_mean`、`repeat_min`、`repeat_max`、`bootstrap_unit='beer_id'`、`n_boot=2000`、`interval_scope='conditional_on_fitted_models_and_splits'`；置换检验另列 `perm_p` 与 `perm_q95`，不得将置换零分布分位当成抽样区间。逐风格指标还分别保存每风格区间及宏平均、按样本量加权平均的区间。
- `--preset prerun`：固定 famod、k=15、κ=2，2 个重复，不做置换，只算 C2ST、DCR、TRTS、TSTR、下限、相关、模块内 ρ；只用于回归。

## 验收标准

通用：`black --check code/` 与 `ruff check code/` 通过；`src/q3/` 无 `__main__`，导入时不读写文件；脚本写 `seed` 与 `meta`。

pytest（`code/tests/test_q3_gen_metrics.py`，合成数据，秒级）：
- 两组同分布样本的 C2ST AUC 在 0.5±0.1 内；均值平移 3 个标准差时 AUC 大于 0.9。
- 置换零分布均值在 0.5±0.05 内；同种子重复结果一致。
- `gen` 为 `real` 的副本时，`dcr_ratio` 恰为 0。
- `floor_metrics` 对 `real` 与其自身比较返回误差 0。
- `acceptance` 对构造的通过样本与失败样本分别返回全 True 与对应项 False。

数值核对（`--preset prerun`，2 个重复；随机森林与抽样次序与预检查不同，容差已放宽）：
- 主模型：C2ST AUC 0.510（±0.03）；DCR 比 1.11（±0.05）；TRTS 0.662（±0.04）；TSTR 0.486（±0.04）；下限比例误差 0.020（±0.003）；每款下限列数中位数 28（±1）；相关误差 0.061（±0.005）；模块内 ρ 0.680（±0.02）。
- 对照：G0 的 C2ST AUC 0.579（±0.03）；G2 的 C2ST AUC 0.651（±0.03）；TRTR 0.540（±0.03）；真实留出啤酒的模块内 ρ 0.724（±0.02）。

正式运行：
- 每个重复的池含 250 款真实啤酒和每个生成器的 250 个配对样本；置换次数 200；真实对真实与阳性对照均有结果；输出中的 NaN 均有原因字段。
- `acceptance`（方法论 §6.2），逐项写入布尔值与所依据的数值，各项取 5 个重复的平均值判定：
  - (a) 合并 C2ST 与条件 C2ST 的 |AUC − 0.5| 均不超过 0.05，且均低于 G0 的对应 AUC；
  - (b) DCR 比不低于 0.9，近复制率不超过 0.10；
  - (c) TSTR 比 TRTR 低不超过 0.10，且高于 G0 的 TSTR；
  - (d) 下限比例误差不超过 G1 的 2 倍；
  - (e) 模块内 ρ 比真实留出啤酒低不超过 0.10，且风格内相关误差不超过 G1 的 2 倍。
- 不满足的项照实写入，不调整主模型、阈值或划分；按方法论 §6.2 自动在 `review/` 开线程，附判定项与数值。该验收失败本身不阻止 W4、W5b、W6 继续使用已生成的结果；后续报告必须保留失败标记，不得把失败解释为通过。
- Tripel、Kriek、Lambic 的逐风格结果单列；`n_s ≤ 5` 的风格标 `evidence_limited`；离散比 `flag` 为真的风格列出。
- G1、G3 的生成池内有近重复样本，其 AUC 低于 0.5 不表示更像真实啤酒；输出中对二者标注该限定。
- 耗时估计：置换共 60 个单元（5 个重复 × 6 个生成器 × 2 个变体），每个单元 200 次 × 5 折随机森林，8 个进程约 1–2 小时；逐风格主模型置换约 0.5–1 小时；以实测回报。

## 处理结果
<!-- [PLACEHOLDER] | Await: coding 回复 ack 并逐项回报 -->
