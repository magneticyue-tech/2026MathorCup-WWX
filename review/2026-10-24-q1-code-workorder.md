---
id: 2026-10-24-q1-code-workorder
from: modeling
to: coding
status: open
severity: major
refs:
  - model/q1_methodology.md
  - src/q1_methodology_checks.py
  - src/q1_supplement_checks.py
  - results/data_audit/chemical_dictionary.csv
  - data/raw/1.csv
  - data/raw/4.csv
  - data/raw/8.xlsx
  - environment.yml
---

## 问题
本工单把 `model/q1_methodology.md` 转为实现：化学数据 → $\hat z$ → 0–6 评分，交付 250 款 × 50 维的评分、各分值概率与 90% 区间，并产出 §5 的全部验证结果。
方法以方法论文档为准；本工单只规定代码结构、产物和验收，凡与方法论冲突处以方法论为准并在本线程回报。
任务按 T1–T7 排列，后一项依赖前一项的 `.npz`；T5–T7 可并行。

## 背景 / 复现

### 运行前需确认的环境事实
- 原始附件在仓库根目录 `data/raw/`，`code/data/raw/` 为空；现有检查脚本与审计结果都读根目录。本工单从 `data/raw/` 只读读取，不复制附件；路径集中在 `code/src/q1/paths.py` 的 `RAW_DIR` 一处，便于以后迁移。
- 输出写入 `code/data/processed/`，图表写入 `figures/`（目录尚不存在，由 T7 创建）。`.gitignore` 只忽略根目录 `data/processed/*` 和 `*.pkl`/`*.joblib`，因此 `code/data/processed/*.npz` 会入库；绘图脚本依赖它们在其他机器上重生成，应提交。
- `environment.yml` 未固定 scikit-learn；本机为 Python 3.14 / scikit-learn 1.9.0，补充核对用的是 1.7.2。T1 在 `environment.yml` 中固定 `scikit-learn=1.7.2`，正式运行在该环境下进行，每个求解脚本把 Python、numpy、pandas、scikit-learn 版本写进 `.npz` 的 `meta` 字段。
- 品评团推断与零点的参考实现在 `src/q1_methodology_checks.py` 的检查 3、检查 4。移植到 `code/src/q1/` 后输出必须与之逐元素一致（见验收）。

### 随机种子
- `SEED = 0`，定义在 `code/src/q1/paths.py`。
- 主划分第 $r$ 次重复（$r=0..4$）用 `random_state=SEED + r`；内层 4 折用同一外层的 `random_state`。
- 论文 70/30 划分用 `random_state=0`（§5.1 指定）。
- 树模型 `random_state=SEED`；bootstrap 用 `np.random.default_rng(SEED)`，2,000 次。
- 每个 `.npz` 存 `seed` 字段。

### 维度组
超参数按 §1 的 6 个组共享：香气区块 15、风味区块 15、缺陷 6、香料 3、酸啤酒 3、基本味/口感/总体 8。常量 `DIM_GROUPS` 放在 `code/src/q1/data.py`，并断言 6 组并集恰为附件 4 的 50 列。

## 文件与函数

### T1 `code/src/q1/`：数据、编码、品评团

`paths.py`：`ROOT`、`RAW_DIR`、`PROCESSED_DIR`、`FIG_DIR`、`SEED`。

`data.py`
- `load_q1(raw_dir: Path = RAW_DIR) -> Q1Data`：dataclass，字段 `X`（250×231 DataFrame）、`Z`（250×50）、`beer`、`style`（`tasting_category_fine`）、`dims`。断言两表 `beer` 顺序一致、列数 231/50。
- `class ChemNAEncoder(scheme: str)`，`fit(X_train)` / `transform(X) -> tuple[DataFrame, DataFrame]`，返回编码后的化学表和 8 个 `<col>_missing` 指示列。`scheme` 取值：
  - `"main"`：单化合物 NA → 0，`aroma_hops_sum` 保留 NaN（§2.1）；
  - `"amyl_nan"`：在 main 基础上 amyl.acetate 改为 NaN；
  - `"all_nan"`：全部单化合物 NA 保留 NaN；
  - `"col_mean"`：训练折列均值插补，均值只在 `fit` 中计算。
- `floor_indicators(X) -> DataFrame`：`round(x, 8) == -1.81781854` 的指示列，只供线性对照模型使用（§2.2）。

`panels.py`（移植检查 3、4，不改算法）
- `tie_pair_counts(Z, decimals=9) -> dict[tuple[int,int], int]`：每对啤酒在多少维上 $z$ 精确并列。
- `infer_panels(Z, k=3, decimals=9) -> np.ndarray`：连通分量标签，长度 250（§2.4）。
- `panel_zeros(Z, panels, decimals=9) -> DataFrame`：行为多啤酒品评团、列为维度；组内最小值出现 ≥2 次才记为零点，否则 NaN（§4.2）。
- `at_zero_mask(Z, panels, zeros) -> np.ndarray[bool]`：250×50，啤酒是否处于本团零点。

`code/scripts/solve_q1_panels.py` → `code/data/processed/q1_panels.npz`：`panels`（k=3）、`panels_k2`…`panels_k5`、`zeros`（品评团×50）、`zero_panel_ids`、`at_zero`、`dims`、`seed`、`meta`。

### T2 `code/src/q1/`：特征、划分、主模型

`features.py`
- `MONOTONE_PAIRS`：`ethanol..v.v.`→alcohol、`bitterness`→bitternes、`pH`→acidity（−1）、`lactic_acid.mg.L.`→lactic、`acetic_acid.g.L.`→acetic、`X4VG`→X4vg（§3.3）。
- `monotone_cst(feature_names, dim, enabled: bool) -> list[int]`：只在该维度的配对化合物上置 ±1，其余为 0。
- `oav_class_sums(X, dictionary, classes) -> DataFrame`：$\sum_{c\in\text{class}}\text{conc}_c/\text{threshold}_c$；`transform_interpretation` 以 `log10` 开头的列先取 $10^x$，阈值取 `chemical_dictionary.csv` 的 `threshold_mean`（§3.2）。
  <!-- [PLACEHOLDER] | Await: modeling 在本线程给出酯类、硫化合物、酚类三类的化合物名单 -->
  名单到位前 OAV 实验不运行，主运行不含 OAV。

`splits.py`
- `outer_splits(panels, style, n_splits=5, n_repeats=5, seed=SEED) -> list[list[tuple[ndarray, ndarray]]]`：`StratifiedGroupKFold(shuffle=True)`，单例啤酒各自为一组（§5.1）。
- `inner_splits(train_idx, panels, style, seed) -> list[tuple]`：训练折内 4 折，同样分组分层。
- `leave_one_style_splits(style)`、`paper_split(style, seed=0)`：§5.1 的压力测试与 70/30 划分。
- `split_diagnostics(splits, style) -> dict`：各折测试集大小与风格 TVD，并对同种子 `GroupKFold` 计算同一指标。

`model.py`
- `PARAM_GRID`：§3.1 的网格。单调约束开/关只在含配对维度的 3 个组（缺陷、酸啤酒、基本味/口感/总体）中展开，其余组只跑“关”，结果相同。
- `make_hgb(params, monotone) -> HistGradientBoostingRegressor`：`loss="squared_error"`、`early_stopping=False`、`random_state=SEED`。
- `class HurdleRegressor(BaseEstimator, RegressorMixin)`：`fit(X, z, at_zero, z0_tilde)`，分类器 `HistGradientBoostingClassifier` 预测“高于零点”，回归器只在高于零点的训练啤酒上拟合；`predict` 返回 $\hat\pi\hat\mu^{+}+(1-\hat\pi)\tilde z_{0,d}$（§3.4）。高于零点的标签取 `at_zero_mask` 的取反；没有本团零点的啤酒记为高于零点；$\tilde z_{0,d}$ 由训练折计算。
- `HURDLE_DIMS`：检查 4 中本团零点占比 ≥30% 的 10 个维度，在运行前固定为常量。
- `nested_cv_predict(X, Z, panels, style, splits, dim_groups, grid, use_hurdle=True, n_jobs=-1) -> NestedCVResult`：
  - 对每个外层折、每个维度组，按内层平均 Spearman 选组内共享的超参数；
  - 对 `HURDLE_DIMS` 中的每个维度，在选定超参数下比较内层 Spearman，决定障碍模型开或关；
  - 用选定配置在外层训练折上拟合，预测外层测试折。
  - 返回 `oof`（5×250×50）、`inner_resid`（5×5×250×50，外层训练折内选定配置的内层折外残差，非训练折位置为 NaN）、`chosen_params`、`hurdle_on`（5×5×10）。
  - `max_iter` 用 `warm_start` 依次加到 100/200/400，三档共用一次拟合。
  - 并行在维度上展开，模型内部线程设为 1，避免线程过量占用。
- `refit_full(X, Z, panels, style, dim_groups, grid) -> dict`：全 250 款上用一次分组 5 折选超参数后重拟合，供新啤酒和 Q3 使用（§3.5）。

`code/scripts/solve_q1_main.py` → `code/data/processed/q1_main_oof.npz`：`oof`、`zhat`（5 次重复的均值）、`inner_resid`、`fold_id`（5×250）、`chosen_params`（JSON 字符串）、`hurdle_on`、`split_diag`、`wall_time_s`、`seed`、`meta`。
全数据模型写入 `code/data/processed/q1_models/*.joblib`；`.gitignore` 忽略该文件，脚本可在本地重新生成。所选超参数同时写入 npz，供重拟合复现。

运行开销估算：不含障碍模型，每个外层折约 6,400 个“配置 × 内层折 × 维度”组合，warm_start 合并 max_iter 三档后约 2,100 次拟合，25 个外层折共约 5 万次。若 `wall_time_s` 超过 4 小时，在本线程报告实测时长，不自行缩小网格。

### T3 `code/src/q1/convert.py`：z → 0–6

所有参数只用外层训练折中的啤酒与品评团估计（§4.5、§5.5）。

- `ANCHOR_EXCLUDE = ["F_malt_all"]`（§4.2 重新归类）；`SUSPECT_ZERO_DIMS` 为 §4.2 列出的 7 个零点可疑维度，`FRAGILE_ZERO_DIMS` 为 4 个单团零点维度，均在运行前固定为常量。
- `fit_conversion(Z_train, panels_train, m: float) -> ConversionParams`，每个维度给出：
  - `z0_tilde`：训练折各品评团零点的中位数；
  - `r = -z0_tilde`；
  - `M`：训练折 $\max_b\lvert z_{bd}\rvert$；
  - `ecdf`：训练折 $z$ 的排序数组，用于 C-quantile；
  - `cap`：$6(1+r^2)/\big(r\max_b(z_{bd}-z_{0,g(b)d})\big)$，无本团零点的啤酒用 `z0_tilde`；
  - `anchored`：bool。
- `anchored` 的规则：维度不在 `ANCHOR_EXCLUDE` 中，且训练折内至少有一个品评团零点。4 个单团零点维度在其零点所在品评团进入测试折时，训练折没有零点；此时该折该维度按无锚处理，并记入 `fallback_count`。
- `to_score(z, params, m, unanchored="C-mid", zero=None) -> ndarray`：
  - 有锚维度：$\operatorname{clip}\big(m\,r/(1+r^2)\,(z-z_0),0,6\big)$，`zero=None` 时 $z_0$ 取 `z0_tilde`；
  - 无锚维度：C-mid 为 $\operatorname{clip}(3+3z/M,0,6)$，C-quantile 为 $6\hat F(z)$。
- `reference_scores(Z_test, panels_test, zeros_train_and_test, params, m) -> ndarray`：$x^{*,\text{ref}}$ 用观测 $z$ 与本团零点，单例和无本团零点的啤酒用 `z0_tilde`（§4.5 表）。本团零点由该团自己的观测 $z$ 计算，只用于参考评分，不进入交付评分。
- `below_zero_share(Z, panels, zeros, params) -> Series`：零点有效性规则中低于零点的啤酒占比（§4.2）。
- `parent_child_violations(scores, parents) -> DataFrame`：6 个父描述符 `*_all` 与其子描述符，报告子 − 父 > 0.5 的单元格占比（§4.4），只作诊断。

### T4 `code/src/q1/uncertainty.py`：概率与区间

- `score_distribution(zhat_bd, resid_d, params_d, m) -> tuple[int, ndarray]`：对训练折该维度的每个内层折外残差 $e_j$，把 $\hat z+e_j$ 转换并裁剪；交付评分为其均值取整，`p` 为 7 个分值的占比（§4.5）。
- `conformal_halfwidth(resid_d, alpha=0.1) -> float`：$\lvert e\rvert$ 的 $\lceil(n+1)(1-\alpha)\rceil/n$ 分位数。残差来自分组内层 CV，因此按品评团校准。
- `score_interval(zhat, q, params, m) -> tuple[float, float]`：把 $[\hat z-q,\hat z+q]$ 经单调转换映射到 0–6。
- `coverage(Z_test, zhat, q) -> float`：外层测试折上的 z 空间覆盖率。偏离 90% 超过 5 个百分点时，脚本把 `mondrian_needed=True` 写进输出，不自动切换（§4.5 的推翻条件）。

`code/scripts/solve_q1_scores.py` 读取 `q1_panels.npz` 与 `q1_main_oof.npz`，写出 `code/data/processed/q1_scores.npz`：
- 对每个 $m\in\{2.5,3,4\}$ 以及逐维度的 $\min(4,\text{cap}_d)$，给出 `score`（250×50 int）、`score_cont`（250×50）、`p`（250×50×7）、`lo`/`hi`（250×50）；
- 另给出 `ref_score`（同 $m$ 网格）、`unanchored_quantile`（C-quantile 运行）、`suspect_as_cmid`（可疑与脆弱零点维度改用 C-mid 的运行）；
- 以及 `anchored`（5×5×50）、`fallback_count`、`cap`（5×5×50）、`coverage`、`mondrian_needed`、`below_zero_share`、`parent_child`。

<!-- [PENDING] | Await: modeling 在本线程确认“5 次重复如何合并到交付评分” -->
暂定做法：每次重复用该折自己的转换参数和训练折残差得到 `score_cont` 与 `p`，再在 5 次重复上取平均，交付评分为平均后的 `score_cont` 取整。§3.5 写的是先对 $\hat z$ 取平均，但各次重复的转换参数不同，先平均 $\hat z$ 时无法指定唯一的一套训练折参数。若 modeling 改为先平均 $\hat z$，需同时指定使用哪一套参数。

### T5 `code/src/q1/baselines.py` 与 `metrics.py`：对照与指标

`baselines.py`，全部使用 `q1_main_oof.npz` 中记录的同一套外层划分，预处理只在训练折内拟合（§5.2）：
- `b0_mean(Z, splits)`、`b1_style_shrunk(Z, style, splits, k=5)`：B1 为 $(n_s\bar z_{s,d}+5\bar z_d)/(n_s+5)$，未见过的风格退回 $\bar z_d$。
- `linear_top10(X, Z, splits)`：训练折内按 $\lvert\rho_{\text{Spearman}}\rvert$ 选前 10 个特征，`QuantileTransformer(n_quantiles=200)` 后 `StandardScaler`，加 `floor_indicators`，最小二乘。
- `elasticnet(X, Z, splits, panels, style)`：全部 239 个特征加下限指示列，同样的分位数变换与标准化；内层分组 CV 网格为 `alpha` ∈ logspace(−3, 1, 9)，`l1_ratio` ∈ {0.1, 0.5, 0.9}，按维度组共享。这是 §3.1 推翻检验所用的模型。
- `random_forest(X, Z, splits, panels, style)`：`n_estimators=500`，`max_features` ∈ {0.33, `"sqrt"`}，`min_samples_leaf` ∈ {5, 10}，按维度组共享。由于 `RandomForestRegressor` 不接受 NaN，`aroma_hops_sum` 用训练折中位数填补并另加缺失指示列。
- `chem_plus_style(...)`：主模型特征加风格 one-hot，网格与主模型相同（§3.5 回退方案）。
- `b2_paper(X, Z, style)`：论文流程只在 `paper_split(seed=0)` 上运行：Shapiro–Wilk 判定后取对数、列均值插补、标准化、`GradientBoostingRegressor`，训练集内 5 折网格搜索按 $R^2$ 选参。
  <!-- [PLACEHOLDER] | Await: modeling 在本线程给出论文 GBR 的搜索网格与 Shapiro–Wilk 判定阈值 -->
  参照值取论文 Table S3 逐维测试 $R^2$ 的 50 维平均 0.189，与本脚本的汇总方式相同；Table 1 的 0.21 汇总方式不同，不用于比较。差距超过 0.03 时在本线程报告，不为逼近该值而调参。

`metrics.py`，全部在合并后的 250 款 $\hat z$ 上计算（§5.3）：
- `spearman_by_dim(Z, Zhat) -> Series`，并列取平均名次；`mean_spearman(..., exclude=())` 同时报告含与不含 barnyard 的结果（§2.3）。
- `cluster_bootstrap_diff(Z, Zhat_a, Zhat_b, panels, n_boot=2000, seed=SEED) -> (diff, lo, hi)`：按品评团整团重抽样，单例各自为一团。
- `wins(Z, Zhat_a, Zhat_b) -> int`：逐维 Spearman 更高的维度数。
- `rmse_over_sd`、`r2`。
- `score_metrics(score, ref_score) -> dict`：MAE、±1 准确率、`cohen_kappa_score(weights="quadratic", labels=range(7))`；有锚与无锚分开报告。
- `changed_share(score_m3, score_m4, anchored)`：有锚单元格中两档 $m$ 评分不同的比例。
- `reliability_bound(Z, n) -> Series`：§5.4 的 $R_n$；任一维度 $R^2>R_{16}$ 时写出 `leak_flag`。

`code/scripts/solve_q1_baselines.py` → `code/data/processed/q1_baselines_oof.npz`：每个对照的 `oof`（5×250×50）与 `zhat`、各自选定的超参数、`b2_test_idx`、`b2_pred`。

`code/scripts/solve_q1_metrics.py` 读取 `q1_main_oof.npz`、`q1_baselines_oof.npz`、`q1_scores.npz`，写出 `code/data/processed/q1_metrics.json`。其中运行前声明的判定以布尔值记录，不由脚本改动任何模型：
- `success`：主模型平均 Spearman > B1、配对差 CI 下界 > 0、且胜过 B1 的维度 ≥ 25（§5.3）；
- `linear_overturn`：ElasticNet 的 50 维平均 Spearman ≥ 主模型（§3.1）；
- `control_beats_main`：每个对照模型相对主模型配对差的 CI 下界 > 0 时为真；为真时在 `review/` 开新线程（§5.2）。
- `split_diag`：B1 在 LOO 与分组划分下的 $R^2$，参考值为 0.181 和 0.115（检查 5）。

### T6 敏感性与外部检查

`code/src/q1/sensitivity.py`
- `mask_cells(X, test_idx, frac, mechanism, rng) -> DataFrame`：只在测试折、只在原本有观测的单化合物单元格上遮蔽。`mechanism="MCAR"` 为均匀随机遮蔽；`mechanism="MNAR"` 只在该列训练折最低 30% 分位以下的测试单元格中抽取。遮蔽后的单元格交给当前对照臂的 `ChemNAEncoder` 处理。
- `run_arm(scheme, mask=None, fixed_params=...) -> ndarray`：对照臂沿用主运行在每个外层折、每个维度组选定的超参数和障碍模型开关，只换 NA 编码和遮蔽，因此差值只反映编码，不混入调参的随机性；代价是各臂没有按自身编码调参。
  <!-- [PENDING] | Await: modeling 在本线程确认对照臂复用主运行超参数，或改为各臂独立嵌套 CV（运行时间约为主运行的 7 倍） -->

`code/scripts/solve_q1_sensitivity.py` → `code/data/processed/q1_sensitivity.npz`，内容如下（§2.1）：
- 对照臂 (a) `amyl_nan`、(b) `all_nan`、(c) `col_mean`；
- 遮蔽实验 {5%, 10%} × {MCAR, MNAR} × {main, all_nan, col_mean}，遮蔽用 `rng = default_rng(SEED + 100)`；
- 每项给出 `oof`、相对主设定的 50 维平均 Spearman 差及其品评团簇 bootstrap 95% CI，以及 `robust` 判定（全部差的绝对值 ≤ 0.01）；
- `robust=False` 时另写出该臂的交付评分，并列出交付评分变化超过 0.5 分的维度。

`code/scripts/solve_q1_stress.py` → `code/data/processed/q1_stress.npz`，用主模型配置运行留一风格划分与论文 70/30 划分（§5.1），逐风格给出 Spearman；3–5 款啤酒的风格单独标出。

`code/scripts/solve_q1_external.py` → `code/data/processed/q1_external.npz`（§5.6）：
- 读取 Zenodo 工作簿，路径来自环境变量 `ZENODO_XLSX`，md5 必须为 `52645e38d90aafda272532e19e2fea1c`；缺少文件时脚本退出并给出提示，不影响其他任务。属性映射与对照杯均值的计算直接复用 `src/q1_supplement_checks.py` 区块 E。
- 对啤酒 64 和 223，在 $m\in\{2.5,3,\min(4,\text{cap}_d)\}$ 下分别给出两套评分与实测均值的差：一是参考评分，即观测 $z$ 换算的结果；二是交付评分，即折外 $\hat z$ 换算的结果。
- 参考评分在 $m=3$ 时的有锚平均误差应复现区块 E 的 −0.47；差距超过 0.02 时说明实现与补充核对不一致，先在本线程报告。
- Zenodo 数据只用于报告，不进入任何拟合或选择（§5.5）。

### T7 图表与交付文件

每个绘图脚本只读 `code/data/processed/` 下的产物，不调用求解函数；脚本名与产物名一致。

| 脚本 `code/scripts/` | 产物 `figures/` | 内容 |
|---|---|---|
| `fig_q1_panels.py` | `fig_q1_panels.pdf` | 品评团大小分布；k=2–5 的组数与最大组 |
| `fig_q1_spearman.py` | `fig_q1_spearman.pdf` | 50 维逐维 Spearman：主模型、B1、ElasticNet，按维度组排列 |
| `fig_q1_paired_diff.py` | `fig_q1_paired_diff.pdf` | 各对照相对主模型的 50 维平均 Spearman 差与 95% CI |
| `fig_q1_conversion.py` | `fig_q1_conversion.pdf` | 有锚维度的 $\hat x^*$ 对参考评分；$m=3$ 与 $m=4$ 的评分变化比例 |
| `fig_q1_sensitivity.py` | `fig_q1_sensitivity.pdf` | NA 对照臂与遮蔽实验的 Spearman 差及 CI，标出 ±0.01 |
| `tab_q1_metrics.py` | `tab_q1_metrics.tex`、`tab_q1_metrics.csv` | §5.3 指标表：主模型、B0、B1、线性回归、ElasticNet、随机森林、B2、化学 + 风格 |
| `tab_q1_report_beers.py` | `tab_q1_report_beers.tex` | 正文报告的 12 款啤酒 × 50 维评分 |
| `export_q1_scores.py` | `code/data/processed/q1_scores.csv`、`q1_scores_long.csv` | 宽表 250×50；长表每行包含啤酒、维度、评分、`p0`…`p6`、`lo`、`hi`、`anchored`、`barnyard_flag` |

<!-- [PLACEHOLDER] | Await: writing 在本线程给出正文报告的 12 款啤酒编号 -->
名单到位前 `tab_q1_report_beers.py` 不运行，其余脚本不受影响。
无锚维度在长表中标为 `convention="C-mid"`，barnyard 的 `barnyard_flag` 为“疑似 stale_hops 副本”（§2.3、§4.4）。

## 验收标准

通用：
- `black --check code/` 和 `ruff check code/` 通过。
- `code/src/q1/` 下没有 `__main__`，导入时不读写文件、不绘图。
- 每个求解脚本把 `seed` 与 `meta` 版本信息写进输出，正式运行在固定后的 scikit-learn 1.7.2 环境中进行。
- 删除 `figures/` 后，只运行 `fig_*`、`tab_*` 和 `export_*` 脚本即可重新生成全部图表，不触发任何求解函数。

数值核对，失败时在本线程报告，不自行改方法：
- T1：`infer_panels(k=3)` 得到 41 个多啤酒组，覆盖 228 款，最大组 11 款；`panel_zeros` 与 `src/q1_methodology_checks.py` 检查 4 的 `W` 逐元素一致；10 个无锚维度与 §4.2 的名单一致。
- T1：编码后有 8 个 `_missing` 列；`aroma_hops_sum` 有 13 个 NaN；其余列没有 NaN。
- T2：250 款啤酒在每次重复中恰好出现在一个测试折；同一品评团的啤酒不跨折；测试折大小在 47–52 范围内，风格 TVD 均值约 0.34。
- T2：任何拟合量都不使用测试折。单元测试检查方法：把测试折的 $z$ 换成随机数，训练折上的预测与超参数应不变。
- T3：$m=3$ 下全部有锚维度满足 $m\le\text{cap}_d$；F_malt_all 为无锚；`below_zero_share` 与 §4.2 列出的占比相差不超过 0.5 个百分点（全样本口径）。
- T4：`p` 每行和为 1；`lo ≤ score ≤ hi`；报告 `coverage`。
- T5：B1 在 LOO 和分组划分下的 50 维平均 $R^2$ 复现 0.181 和 0.115，误差在 0.002 以内。
- `code/tests/` 下用 `pytest` 覆盖上述可在秒级完成的检查；需要完整运行的核对写进 `q1_metrics.json` 的 `checks` 字段。

## 需要回复的决策
1. OAV 三个类别的化合物名单（T2）。
2. 5 次重复合并的顺序：先转换再平均，或先平均 $\hat z$（T4）。
3. 敏感性对照臂是否复用主运行超参数（T6）。
4. B2 的 GBR 网格与 Shapiro–Wilk 阈值（T5）。
5. 正文 12 款啤酒的名单，由 writing 提供（T7）。

决策 1–5 不阻塞 T1–T3 和主运行。

## 处理结果
<!-- [PLACEHOLDER] | Await: coding 回复 ack 并逐项回报 T1–T7 的验收结果 -->
