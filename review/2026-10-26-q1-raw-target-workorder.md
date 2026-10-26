---
id: 2026-10-26-q1-raw-target-workorder
from: modeling
to: coding
status: open
severity: major
refs:
  - model/q1_methodology.md
  - memory/no-external-data.md
  - data/raw/raw_sensory_data_anonymized.csv
  - data/raw/1.csv
  - data/raw/4.csv
  - experimentalCode/q1_raw_target/README.md
  - review/2026-10-24-q1-code-workorder.md
  - review/2026-10-25-q1-acceptance-issues.md
  - review/2026-10-25-q1-score-mapping-workorder.md
  - code/src/q1/
  - code/scripts/
---

## 问题

`model/q1_methodology.md` 已重写：Q1 的训练目标改为逐品评员原始评分的啤酒均值 $\bar x_{bd}$，评价真值为其取整等级 $G_{bd}$，0–6 等级由预测分布中位数给出。原先以附件 4 的 $z$ 为目标、用零点和线性端点换算 0–6 的流程，降为对照 C1。
本工单把新方法转为实现，规定代码结构、产物和验收。方法以方法论文档为准；与方法论冲突处以方法论为准，并在本线程回报。

与现有线程的关系：
- `2026-10-25-q1-score-mapping-workorder` 全部被本工单替代。混合参照和两种映射只作为 C1 实现（T4），不再用于选型或交付。
- `2026-10-24-q1-code-workorder` 中，T1 的品评团、T3 的 `convert.py`、T4 的 `uncertainty.py`、T6 的外部检查换算，以及所有以 $z$ 为目标或以 Spearman 选型的规定，被本工单替代。NA 编码对照、特征、单调约束、划分结构、对照模型骨架、B2 和图表规范继续有效，按本工单修改。
- `2026-10-25-q1-acceptance-issues` 的第 1–3 项（NA 主设定改为训练折列均值、补全参数网格、嵌套 CV 结构）仍然有效，并入本工单 T1–T2。其中针对 $z$ 目标的部分以本工单为准。

任务按 T1–T7 排列，后一项依赖前一项的 `.npz`；T4–T6 可并行。

## 背景 / 复现

### 参考实现
`experimentalCode/q1_raw_target/` 是预运行脚本。它使用固定超参数、`GroupKFold(shuffle=True)`、3 次重复，只作算法参考，不是交付代码。正式代码按本工单写在 `code/` 下，不从 `experimentalCode/` 导入。目录 README 列出各脚本与方法论章节的对应关系。

### 环境与路径
- 原始评分文件 `data/raw/raw_sensory_data_anonymized.csv` 只在根目录 `data/raw/`（已入库）；`code/data/raw/` 被 `.gitignore` 忽略，且没有该文件。`code/src/q1/paths.py` 的 `RAW_DIR` 现指向 `code/data/raw/`，改为根目录 `data/raw/`，并新增 `RAW_SENSORY`。
- 环境沿用 2026-10-24 工单的约定：固定 scikit-learn 1.7.2，每个求解脚本把 Python、numpy、pandas、scikit-learn 版本写进输出的 `meta` 字段。
- 并行：预运行中，joblib 以 15 个工作进程运行、模型线程未限制时，工作进程被系统终止；改为每折一个独立进程、`OMP_NUM_THREADS=1` 后正常。正式运行时，模型内部线程设为 1，并行只在维度或折上展开。

### 随机种子
- `SEED = 0`，定义在 `paths.py`。外层第 $r$ 次重复（$r=0..4$）用 `random_state=SEED + r`，内层 4 折用 `SEED + 100 + r`。
- 树模型 `random_state=SEED`；bootstrap 用 `np.random.default_rng(SEED)`，2,000 次；信度的折半划分用 `default_rng(SEED)`，20 次。
- 每个 `.npz` 存 `seed` 字段。

### 旧产物
`code/data/processed/q1/` 中以 $z$ 为目标的旧产物（`q1_main_oof*.npz`、`q1_scores*.npz`、`q1_final_scores*.csv`、`q1_mapping_*`、`q1_panels.npz` 等）移到 `code/data/processed/q1/legacy_z/`，新产物使用下文的文件名，不静默复用同名旧文件。

## 文件与函数

### T1 数据：`paths.py`、`data.py`

`paths.py`：`RAW_DIR = ROOT / "data" / "raw"`，`RAW_SENSORY = RAW_DIR / "raw_sensory_data_anonymized.csv"`。

`data.py`
- `DIMS`：附件 4 的 50 列顺序。`DIM_GROUPS`：§1 的 6 个维度组，断言并集恰为 `DIMS`。`ZERO_HEAVY_DIMS`：至少一半啤酒 $G=0$ 的维度，运行前由全样本 $G$ 计算一次并固定为常量（预运行为 22 个），只用于分组报告。
- `grade(v: ndarray) -> ndarray[int]`：$q(v)=\lfloor\operatorname{clip}(v,0,6)+0.5\rfloor$（§1）。
- `load_raw_ratings(path=RAW_SENSORY) -> DataFrame`：`encoding="cp1252"`；`A_ester_fruity`、`F_ester_fruity` 更名为 `A_esters_fruity`、`F_esters_fruity`；保留 `date`、`beer_name`（重命名为 `row`，取值 1–250）、`taster` 和 50 个维度列（float）。断言 2,197 行、250 款、16 名品评员、37 个日期，维度列无缺失且取值 ⊂ {0,…,6}（§1.1）。
- `date_groups(ratings) -> ndarray[int]`：长度 250，按行号排列；多日期品评啤酒所在的日期用并查集合并（§2.4）。
- `panel_targets(ratings) -> Q1Targets`：dataclass，字段 `xbar`（250×50）、`G`（250×50 int）、`n_ratings`（250）、`date_group`（250），均按附件行号排列。
- `additive_effects(ratings, rows: ndarray) -> tuple[DataFrame, DataFrame]`：只用 `rows` 中啤酒的评分，拟合 $x_{id}=\alpha_{bd}+\beta_{p(i)d}+\varepsilon$，$\beta$ 按评分条数加权中心化；交替最小二乘迭代至 $\max|\Delta|<10^{-8}$ 或 500 轮。返回 `alpha`（啤酒×50）和 `beta`（品评员×50），供 §2.2 的敏感性目标使用。
- `load_q1()` 扩展 `Q1Data`：增加 `targets: Q1Targets`；保留 `Z`，只供 C1、C2 使用。读入后断言 §1.2 的编号对应：按行号对齐时原始均分与 `Z` 的逐维 Spearman 中位数 ≥ 0.95，按 `beer` 列对齐时 ≤ 0.1。
- `ChemNAEncoder(scheme)`：`"col_mean"` 为主设定（训练折列均值，均值只在 `fit` 中计算）；`"median_ind"` 为 (a)，`"native_nan"` 为 (b)，`"floor_ind"` 为 (c)，定义见 §2.1。现有 `"main"`（填 0）删除或改名 `"zero_fill"`，不进入任何运行。
- `floor_indicators(X) -> DataFrame`：保持现状，只供线性对照使用。

`code/scripts/solve_q1_data.py` → `code/data/processed/q1/q1_data.npz`：`xbar`、`G`、`n_ratings`、`date_group`、`dims`、`reconstruct_ok`（250×50 bool，§1.3 的重建结果）、`reliability`（50）、`half_grade_exact`、`half_grade_qwk`、`attendance_share`（50）、`seed`、`meta`。重建与信度的算法见 `experimentalCode/q1_raw_target/q1_raw_data_checks.py` 的检查 B、F、G。

### T2 划分与主模型：`splits.py`、`model.py`

`splits.py`
- `outer_splits(groups, style, n_splits=5, n_repeats=5, seed=SEED)`、`inner_splits(train_idx, groups, style, seed)`：`groups` 改为日期组 `date_group`，不再使用 $z$ 推断的品评团（§2.4、§5.1）。
- `leave_one_style_splits`、`paper_split` 保持现状。
- `split_diagnostics` 保持现状，增加断言：同一日期组的啤酒不跨折。

`model.py`
- `PARAM_GRID`：§3.1 的完整网格（2×2×3×2，单调约束在循环中展开）。`MONOTONE_PAIRS` 不变。
- 删除 `HurdleRegressor` 和 `HURDLE_DIMS`（§3.4 不纳入）。
- `nested_cv_predict(X, Y, groups, style, splits, dim_groups, grid, encoder_scheme="col_mean") -> NestedCVResult`：
  - `Y` 为 250×50 连续目标，主运行传 `xbar`。
  - 对每个外层折、每个维度组，按组内平均内层 RMSE/SD 选组内共享的超参数（§3.1）。
  - 用选定配置在外层训练折上拟合，预测外层测试折。
  - NA 编码、单调约束开关等全部在对应训练折内拟合。
  - 返回 `oof_mu`（5×250×50）、`inner_mu`（5×5×250×50：外层训练折上选定配置的内层折外预测，其余位置为 NaN）、`fold_id`（5×250）、`chosen_params`。
  - `inner_mu` 是 §4.1 残差池和 §4.3 切点的唯一来源，必须保存。
- `refit_full(X, Y, groups, style, dim_groups, grid) -> dict`：在全部 250 款上，用一次分组 5 折选超参数后重拟合，供新啤酒和 Q3 使用（§3.5）。

`code/scripts/solve_q1_main.py` → `q1_main_oof.npz`：`oof_mu`、`inner_mu`、`fold_id`、`chosen_params`（JSON）、`split_diag`、`wall_time_s`、`seed`、`meta`。全数据模型写入 `code/data/processed/q1/q1_models/*.joblib`（被 `.gitignore` 忽略），选定的超参数同时写入 npz。

### T3 离散化：新建 `discretize.py`，删除 `convert.py`、`uncertainty.py` 中的 $z\to$0–6 逻辑

`discretize.py`（§4）
- `predictive_dist(mu_te: ndarray[n], resid: ndarray[m], mu_ref: ndarray[m] | None = None, k: int | None = None) -> ndarray[n, 7]`：$p_k=\frac1m\sum_i\mathbf 1\{q(\hat\mu+e_i)=k\}$。给出 `mu_ref` 和 `k` 时，只用 `mu_ref` 最接近 $\hat\mu$ 的 `k` 个残差（近邻敏感性）；距离相同时按索引取前者，保证结果确定。
- `dist_median(p) -> ndarray[int]`：$\min\{k:F_k\ge0.5\}$；`dist_interval(p, lo=0.05, hi=0.95) -> tuple[ndarray, ndarray]`（§4.2、§4.4）。比较累积概率时，用 $F_k\ge c-10^{-12}$ 吸收浮点误差。
- `const_mode(g_train) -> int`：众数相同时取较小的等级。
- `quantile_cuts(mu_inner, g_train) -> ndarray[6]`：切点取 `mu_inner` 在训练折累积等级频率处的分位数（`np.quantile` 默认线性插值）；累积频率为 1 的切点设为 $+\infty$。`apply_cuts(mu, cuts) -> ndarray[int]` 为 `searchsorted(cuts, mu, side="right")`。
- `qwk_cuts(mu_inner, g_train, n_pass=3) -> ndarray[6]`：附录规则。从 0.5, 1.5, …, 5.5 出发坐标搜索，候选点为 `mu_inner` 的全部唯一值；只在严格提高 QWK 时更新，QWK 无定义按 −1 处理。
- `apply_rules(mu_te, mu_inner, y_inner, g_inner) -> dict[str, ndarray]`：一次给出 `const`、`round`、`median`、`quant`、`qwk_cut`、`median_knn`（k=50）的等级，以及 `p`、`p_knn`。其中 `y_inner` 为训练折目标，残差 $e=y_{\text{inner}}-\mu_{\text{inner}}$。

`code/scripts/solve_q1_scores.py` 读 `q1_main_oof.npz` 与 `q1_data.npz`，写出 `q1_scores.npz`：
- 逐次重复：`p`（5×250×50×7）、`grade_<rule>`（5×250×50）。
- 交付值，合并 5 次重复的方式按 §3.5：
  - `p_final` 为 5 次 `p` 的平均；`G_hat` 为 `dist_median(p_final)`，`lo`、`hi` 由 `p_final` 计算；
  - `G_hat_Q` 为 5 次 `grade_quant` 的中位数。
- 新啤酒与 Q3 使用的参数：`final_resid`（每维，全部重复的折外残差池）、`final_qcuts`（50×6，由平均折外预测与 $G$ 拟合）。

### T4 对照：`baselines.py`、新建 `ordinal.py`、`twostage.py`

所有对照使用 `q1_main_oof.npz` 记录的同一套外层划分（§5.3）。连续对照模型都像主模型一样输出 `oof_mu` 和 `inner_mu`，再由 `apply_rules` 用各自的残差池离散。

`baselines.py`
- `b0_mean`、`b1_style_shrunk`：目标改为 `xbar`。B1 的 `inner_mu` 在内层折上按同一公式计算。
- `b0g_mode`：训练折众数等级。
- `linear_top10`、`elasticnet`、`random_forest`、`chem_plus_style`：目标改为 `xbar`；调参标准改为内层 RMSE/SD；网格保持 2026-10-24 工单 T5 的规定。
- `b2_paper`：保持在附件 4 的 $z$ 上、论文 70/30 划分，只与论文 Table 1/S3 对照，不进入等级比较。网格仍待 modeling 回复（见下）。

`ordinal.py`（K2，§5.3）
- `fit_predict_k2(X_tr, G_tr, X_te, params, monotone, min_pos=2) -> ndarray[n_te, 7]`：对 $t=1..6$ 拟合 `HistGradientBoostingClassifier`（`loss="log_loss"`、`early_stopping=False`、`random_state=SEED`）。超参数取主模型在该外层折、该维度组的选定值。训练折正例或负例少于 `min_pos` 时取训练折正例比例；$Q$ 沿 $t$ 取累积最小值，再差分得到 $p$。

`twostage.py`（C1、C2，§5.3；参考 `experimentalCode/q1_raw_target/exp_twostage.py`）
- 复用 `panels.py` 的 `infer_panels(Z, k=3)`。
- `group_zeros(Z, panels, idx) -> ndarray[250, 50]`：只用 `idx` 内的啤酒计算各组零点（组内最小值至少出现两次，`F_malt_all` 除外），赋给 `idx` 内同组啤酒，其余为 NaN。训练参照传训练索引，测试参照传测试索引。
- `local_scale(zeros, panels, idx, m=3.0) -> ndarray[50]`：$\sigma^*_d=m r_d/(1+r_d^2)$，$r_d$ 为 `idx` 内各组零点（每组计一次）中位数的相反数。
- `mixed_reference(z, zeros, sigma, lo, hi) -> tuple[ndarray[int], ndarray[bool]]`：返回 $R$ 和来源掩码 $I$。
- `c1_predict(...)`：$z$ 模型按 T2 的嵌套 CV 拟合附件 4 的 $z$（选择标准为对 $z$ 的 RMSE/SD），得到 `oof_z`、`inner_z`。候选映射为 `round_T`、`qwk_T`（切点以训练折 $R$ 为目标）；另输出 `quant_T` 供附录做规则对应比较。选型按外层平均 QWK 对 $R_{\text{te}}$，平局取 `round_T`。
- `c2_predict(...)`：在每个外层训练折上用 OLS 拟合 $z\to\bar x$，作用于 `oof_z` 和 `inner_z`，再调用 `apply_rules`。

`code/scripts/solve_q1_controls.py` → `q1_controls.npz`：每个对照的 `oof_mu`/`inner_mu`（如适用）、`grade_<rule>`、`p`、选定超参数；K2 的 `p`；C1 的 `oof_z`、`R_te`、`I_te`、选定映射与各候选等级；C2 的等级。

### T5 指标：`metrics.py`

全部在每次重复合并后的 250 款折外结果上计算，再对 5 次重复取平均（§5.2）。
- `grade_mae`、`exact`、`within1`：在 12,500 个单元上平均。
- `qwk_by_dim(G, Ghat) -> ndarray[50]`：`labels=range(7)`，无定义时为 NaN；`mean_qwk` 在配对比较中对两方同时剔除 NaN 维度，并返回剔除个数。
- `confusion_pooled(G, Ghat) -> ndarray[7, 7]`、`recall_by_class`、`balanced_accuracy`（真值中出现的等级上召回率的平均）。
- `rps(p, G)`、`log_loss(p, G, eps=0.01)`：定义见 §5.2。
- `spearman_by_dim`、`rmse_over_sd`、`r2`：$\hat\mu$ 对 $\bar x$。
- `cluster_bootstrap(stat_fn, a, b, groups, n_boot=2000, seed=SEED) -> (diff, lo, hi)`：按日期组整组重抽样，每次重抽样都重新计算统计量；QWK 按维度重算后再平均。
- `wins_by_dim(G, Ghat_a, Ghat_b) -> int`：逐维等级 MAE 更低的维度数。
- `split_report(...)`：分别报告 `ZERO_HEAVY_DIMS` 与其余维度；C1 另按 $I=1$、$I=0$ 报告，并报告 $R$ 对 $G$ 的误差。
- `coverage(G, lo, hi)`：90% 等级区间的实际覆盖率与平均宽度。

`code/scripts/solve_q1_metrics.py` 读 `q1_data.npz`、`q1_main_oof.npz`、`q1_scores.npz`、`q1_controls.npz`，写出 `q1_metrics.json`。运行前声明的判定以布尔值记录，脚本不据此改动任何模型：
- `success`：主输出的等级 MAE 低于 B1，配对差 CI 上界低于 0，且逐维 MAE 低于 B1 的维度 ≥ 25（§5.2）；
- `linear_overturn`：ElasticNet 的等级 MAE ≤ 主模型（§3.1）；
- `control_beats_main`：任一对照相对主模型的等级 MAE 配对差 CI 上界低于 0（§5.3）；
- `leak_flag`：任一维度 $\hat\mu$ 对 $\bar x$ 的折外 $R^2$ 高于该维 `reliability`（§5.5）。

### T6 敏感性、压力测试与外部检查

`code/src/q1/sensitivity.py`：对照臂沿用主运行在每个外层折、每个维度组选定的超参数，只改动该臂对应的一项，使差值只反映该项（决定见下）。

`code/scripts/solve_q1_sensitivity.py` → `q1_sensitivity.npz`，各臂见 §5.4：
- 目标：在每个外层训练折内用 `additive_effects` 估计 $\alpha$，以训练啤酒的 $\alpha$ 为目标拟合；测试啤酒仍以 $G$ 评价。另记录 `alpha_flag`（相对主设定的等级 MAE 配对差 CI 上界 < 0，§2.2）。
- 近邻残差：`median_knn`，由 `q1_scores.npz` 直接汇总。
- 化学 NA (a)–(c)，barnyard 排除，删除 4 个 `_sum` 列，附件 8 的 80 个冲突单元，单调约束全开或全关。
- 每臂给出等级 MAE、QWK 相对主设定的配对差及 95% CI，以及交付等级变化的单元比例。

`code/scripts/solve_q1_stress.py` → `q1_stress.npz`：留一风格、论文 70/30（§5.1），报告等级 MAE、QWK 与连续 Spearman；3–5 款啤酒的风格单独标出。

`code/scripts/solve_q1_external.py` → `q1_external.npz`（§5.7）：Zenodo 工作簿路径与 md5 校验沿用 2026-10-24 工单 T6，属性映射与对照杯均值复用 `experimentalCode/q1_supplement_checks.py` 区块 E。对啤酒 64 和 223，比较交付等级 `G_hat` 与对照杯均值的取整等级，并列出实测 $G$。删除旧的 $m$ 网格换算。

### T7 图表与交付文件

绘图脚本只读 `code/data/processed/q1/` 下的产物，不调用求解函数；脚本名与产物名一致，产物写入 `figures/q1/`。

| 脚本 `code/scripts/` | 产物 | 内容 |
|---|---|---|
| `fig_q1_rules.py` | `fig_q1_rules.pdf` | 各离散规则的等级 MAE 对 QWK 散点（§4.3 的取舍），标出主输出 |
| `fig_q1_confusion.py` | `fig_q1_confusion.pdf` | 主输出、K2、C1 的 7×7 汇总混淆矩阵（行归一化，标注计数） |
| `fig_q1_paired_diff.py` | `fig_q1_paired_diff.pdf` | 各对照相对主输出的等级 MAE 配对差与 95% CI |
| `fig_q1_reliability.py` | `fig_q1_reliability.pdf` | 逐维折外 $R^2$ 对信度 $\rho_d$，标出 $y=x$ |
| `fig_q1_reference_error.py` | `fig_q1_reference_error.pdf` | C1 的混合参照 $R$ 对 $G$ 的误差，按 $A$/$L$ 单元分开 |
| `fig_q1_sensitivity.py` | `fig_q1_sensitivity.pdf` | §5.4 各臂的等级 MAE 配对差与 CI |
| `tab_q1_metrics.py` | `tab_q1_metrics.tex`、`.csv` | §5.2 全部指标 × 主模型与全部对照 |
| `tab_q1_rules.py` | `tab_q1_rules.tex`、`.csv` | §4.3 规则比较表 |
| `tab_q1_report_beers.py` | `tab_q1_report_beers.tex` | 正文报告啤酒的预测等级与实测等级 |
| `export_q1_scores.py` | `code/data/processed/q1/q1_final_scores.csv`、`q1_scores_long.csv` | 宽表 250×50 为 `G_hat`；长表每行为啤酒 × 维度：`beer_row`、`beer`、`dim`、`G_hat`、`G_hat_Q`、`p0`…`p6`、`lo`、`hi`、`G_obs`、`xbar_obs`、`barnyard_flag` |

宽表和长表中的啤酒以附件行号 `beer_row` 和附件 `beer` 编号同时标识，避免混淆两种编号。

## 验收标准

通用：
- `black --check code/` 和 `ruff check code/` 通过。
- `code/src/q1/` 下没有 `__main__`，导入时不读写文件、不绘图。
- 每个求解脚本写出 `seed` 与 `meta`；正式运行在 scikit-learn 1.7.2 下进行。
- 删除 `figures/q1/` 后，只运行 `fig_*`、`tab_*`、`export_*` 即可重新生成全部图表，不触发任何求解函数。
- 秒级可完成的检查用 `pytest` 覆盖，放在 `code/tests/`；需要完整运行的核对写进 `q1_metrics.json` 的 `checks` 字段。

数值核对（与 `experimentalCode/q1_raw_target/q1_raw_data_checks.py` 和方法论一致；失败时在本线程报告，不自行改方法）：
- T1：编号对应的 Spearman 中位数，按行号为 0.969，按 `beer` 列为 −0.010；重建在 1e-6 内一致的单元为 11,274；未复现的 1,226 个单元全部落在 15 个零方差组合所在的 12 个维度；$G$ 的等级计数为 [4674, 3579, 2057, 1545, 524, 112, 9]；日期组 25 个，大小 5–24；`infer_panels(k=3)` 的 41 个多啤酒组不跨日期组；信度中位数 0.53。
- T1：主设定编码后化学表没有 NaN。
- T2：每次重复中每款啤酒恰好出现在一个测试折；同一日期组不跨折。无泄漏的单元测试：把测试折的 `xbar` 换成随机数，训练折上的选定超参数、`inner_mu` 和测试预测都不变。
- T3：`p` 每行和为 1；`lo ≤ G_hat ≤ hi`；`G_hat` 与由保存的 `p_final` 重算的结果逐元素一致；`apply_cuts` 输出 ⊂ {0,…,6}。
- T4：C1 测试参照只读测试折的 $z$。单元测试：改动测试组的 $z$ 时，只有 `R_te` 可变，C1 的预测等级和切点不变。
- T5：混淆矩阵每次重复总和为 12,500；由混淆矩阵算出的 MAE、完全一致率与直接计算一致；QWK 与 `cohen_kappa_score(weights="quadratic", labels=range(7))` 一致。
- **移植回归测试：** 用预运行设定（固定超参数：深度 3、学习率 0.05、200 轮、叶节点 20、l2=1；`GroupKFold(5, shuffle=True, random_state=r)`，$r=0,1,2$；内层 `GroupKFold(4, shuffle=True, random_state=100+r)`）运行 T2–T5，应在 ±0.003 内复现以下预运行值：
  - 主输出等级 MAE 0.484，四舍五入 0.495，分位数切点 QWK 0.370；
  - K2 等级 MAE 0.489；
  - C1 选定映射为 QWK 切点，等级 MAE 0.786；
  - $R$ 对 $G$ 的完全一致率 0.629。
  
  脚本以命令行参数 `--preset prerun` 提供该设定；正式运行不使用它。

## 需要回复的决策

modeling 已决定：
1. 敏感性对照臂和 K2 沿用主运行在各外层折、各维度组选定的超参数，不单独做嵌套 CV。
2. 5 次重复的合并顺序按 §3.5：先平均 `p` 再取中位数；`G_hat_Q` 取 5 次等级的中位数。

仍待回复：
3. OAV 三个类别的化合物名单（modeling；2026-10-24 工单遗留）。名单到位前不运行 OAV 实验。
4. B2 的 GBR 网格与 Shapiro–Wilk 阈值（modeling；2026-10-24 工单遗留）。
5. 正文报告的啤酒名单（writing）。名单到位前 `tab_q1_report_beers.py` 不运行。

决策 3–5 不阻塞 T1–T3 和主运行。

## 处理结果
<!-- [PLACEHOLDER] | Await: coding 回复 ack 并逐项回报 T1–T7 的验收结果 -->
