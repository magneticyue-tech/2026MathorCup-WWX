---
id: 2026-10-28-q1-w7-sensitivity
from: modeling
to: coding
status: resolved
severity: major
refs:
  - model/q1_methodology.md
  - review/2026-10-28-q1-w1-data.md
  - review/2026-10-28-q1-w2-main-model.md
  - review/2026-10-28-q1-w3-scores.md
  - review/2026-10-28-q1-w4-baselines.md
  - review/2026-10-28-q1-w6-metrics.md
  - experimentalCode/q1_supplement_checks.py
  - experimentalCode/results/data_audit/calibrated_discordances.csv
  - code/src/q1/sensitivity.py
  - code/src/q1/external.py
  - code/scripts/solve_q1_sensitivity.py
  - code/scripts/solve_q1_stress.py
  - code/scripts/solve_q1_external.py
---

## 问题

- 实现方法论 §5.4 的敏感性各臂、§5.1 的压力测试（留一风格、论文 70/30）、§5.7 的外部检查。
- 现有实现的错误：ALS 臂用全部 250 款估计 α 再与 G 比较，是样本内拟合，MAE 0.096 没有意义（`solve_q1_sensitivity.py:25-40`）；「KNN 变化比例」是在 xbar 上加随机噪声得到的（同文件 `:43-44`）；§5.4 其余各臂没有运行；压力测试只有四舍五入的 MAE，没有 QWK 与 Spearman，留一风格的超参数取自全部 250 款的主运行（`solve_q1_stress.py:24-47`）；外部检查的 MAE 0.48、0.42 和 PASSED 是写死的，并且读取的是别人电脑上的绝对路径（`solve_q1_external.py:17-38`）。
- 本工单依赖 W1（`load_q1`、`load_raw_ratings`、`additive_effects`、`ChemNAEncoder`、`DIMS`、`DIM_GROUPS`、`ZERO_HEAVY_DIMS`、`SUM_NA_COL`）、W2（`fit_outer_fold`、`inner_splits`、`outer_splits`、`leave_one_style_splits`、`paper_split`、`make_hgb`、`monotone_cst`、`q1_main_oof.npz` 的 `chosen_params`、`final_params`、`oof_mu`）、W3（`apply_rules`、`dist_median`、`dist_interval`、`q1_scores.npz`）、W4（`simple_fold`）、W6（`bootstrap_weights`、`cluster_bootstrap`、`pair_stat_fn`、`grade_mae`、`exact`、`qwk_by_dim`、`spearman_by_dim`、`r2_by_dim`）。
- W6 的函数按输入数组的行数计算。若 W6 的实现把 250 行写死，先回到 W6 修正，不在本工单内另写一份。
- 只创建或修改：`code/src/q1/sensitivity.py`、`code/src/q1/external.py`、`code/scripts/solve_q1_sensitivity.py`、`code/scripts/solve_q1_stress.py`、`code/scripts/solve_q1_external.py`、`code/tests/test_q1_sensitivity.py`、`code/tests/test_q1_external.py`。
- 运行环境与通用约定见本工单末尾「通用约定」。

## 固定规则

- 敏感性各臂沿用主运行在每个外层折、每个维度组选定的超参数（`chosen_params` 的键 `r_k_组名`，含 `mono`），只改动该臂对应的一项，使差值只反映该项。各臂不再单独调参（方法论 §5.3）。
- 外层划分、内层划分和种子与主运行完全相同：外层用 W2 的 `outer_splits(date_group, style)`，内层种子为 `SEED + 100 + r`。这样「恒等臂」应逐元素复现主运行，作为一致性检查。
- 残差池与分位数切点只用该外层折训练行的内层折外预测，规则与 W3 相同，直接调用 W3 的 `apply_rules`。
- 所有臂都用 G（由 x̄ 取整）评价，包括 α 目标臂。测试啤酒的评分不参与任何拟合。
- 比较规则取预测分布中位数（键 `median`）。

## 任务 1：`sensitivity.py`（重写，删除旧的 `additive_effects`，该函数已在 W1 的 `data.py`）

常量：
- `SUM_COLS = ('esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum')`。
- `ARMS = ('na_median_ind', 'na_native_nan', 'na_floor_ind', 'drop_sum', 'att8', 'mono_on', 'mono_off', 'alpha')`。
- `DISCORDANCE_CSV = ROOT / 'experimentalCode' / 'results' / 'data_audit' / 'calibrated_discordances.csv'`（Q2 已采用同一做法，允许读取）。

函数：
- `alpha_target(ratings, train_idx, dims=DIMS) -> ndarray (250, 50)`：全 NaN 起步；`alpha, _ = additive_effects(ratings, rows=np.sort(train_idx) + 1)`，其中 `rows` 是附件行号（1–250）；把 `alpha` 按行号重排后写入训练行，其余行保持 NaN。只用训练啤酒的评分，测试啤酒的评分不得进入。
- `apply_att8(X_raw) -> tuple[DataFrame, dict]`：读 `DISCORDANCE_CSV`（`encoding='utf-8-sig'`，列 `beer`、`field`、`csv1_literal`、`calibrated_8`）。其中 `beer` 是附件 1 的 `beer` 列取值，不是行号：用 `X_raw` 对应的 `beer` 列（由调用方传入 `beer: Series`）找行。对每个单元断言当前值与 `csv1_literal` 相差不超过 1e-6，然后写入 `calibrated_8`。返回新表（不改原表）与 `{'n_cells': 80, 'n_beers': 24}` 的实际计数。断言单元数为 80、啤酒数为 24。函数签名 `apply_att8(X_raw, beer)`。
- `fit_fold_fixed(X_raw, Y_fit, dims, dim_groups, groups, style, train_idx, test_idx, inner_seed, chosen, scheme='col_mean', mono_mode='chosen', drop_cols=()) -> dict`：
  1. `tr = np.sort(train_idx)`，`te = np.sort(test_idx)`；`X_use = X_raw.drop(columns=list(drop_cols))`。
  2. `chosen` 为 `{组名: best}`，`best` 含 `max_depth`、`min_samples_leaf`、`learning_rate`、`l2_regularization`、`max_iter` 和 `mono`。`params` 取去掉 `mono` 的部分。`mono_mode` 为 `'chosen'` 时开关取 `best['mono']`，`'on'` 时全 True，`'off'` 时全 False；`cst = monotone_cst(names, 维度名, 开关)`，`names` 是编码后的列名。
  3. 内层：对 `inner_splits(tr, groups, style, inner_seed)` 的每个内层折 `(a, b)`，`enc = ChemNAEncoder(scheme).fit(X_use.iloc[a])`，编码 `a`、`b`，对每个维度用 `make_hgb(params, cst)` 在 `Y_fit[a, d]` 上拟合并预测 `b`，写入 `inner[np.searchsorted(tr, b), d]`。
  4. 外层：`enc` 在 `tr` 上拟合，编码 `tr`、`te`，对每个维度在 `Y_fit[tr, d]` 上拟合并预测 `te`，得到 `oof`。
  5. 返回 `{'train_idx': tr, 'test_idx': te, 'oof': (len(te), 50), 'inner': (len(tr), 50)}`。
  - `Y_fit` 的测试行允许是 NaN，函数不得读取 `Y_fit[te]`。
- `arm_kwargs(arm, X_raw, beer) -> tuple[DataFrame, dict]`：返回该臂使用的化学表与 `fit_fold_fixed` 的额外参数：
  - `na_median_ind`：`scheme='median_ind'`；`na_native_nan`：`scheme='native_nan'`；`na_floor_ind`：`scheme='floor_ind'`；
  - `drop_sum`：`drop_cols=SUM_COLS`，断言 `X_raw` 中四列都在；
  - `att8`：用 `apply_att8` 替换后的表；
  - `mono_on`：`mono_mode='on'`；`mono_off`：`mono_mode='off'`；
  - `alpha`：不改化学表，目标由 `alpha_target` 逐折给出（见脚本）。
  - 未知臂名 `raise ValueError`。
- `unit_grades(mu_te, mu_inner, y_tr) -> dict`：逐维调用 `apply_rules(mu_te[:, d], mu_inner[:, d], y_tr[:, d])`，返回 `{'median': (n_te, 50), 'round': ..., 'quant': ..., 'p': (n_te, 50, 7)}`。`y_tr` 是该臂训练行的拟合目标（α 臂为 α，其余臂为 x̄）。
- `changed_share(g_a, g_b) -> float`：`np.mean(g_a != g_b)`。
- `merge_arm(units, shape=(5, 250, 50)) -> dict`：`units` 为 `{(r, k): 单元字典}`，返回 `oof_mu`（5, 250, 50）、`inner_mu`（5, 5, 250, 50，非训练行为 NaN）、`grade_median`、`grade_round`、`grade_quant`（5, 250, 50）、`p`（5, 250, 50, 7）；断言每次重复每款啤酒恰好写入一次，且 `inner_mu[r, k]` 的非 NaN 行恰为训练行。
- `delivered(p_runs) -> dict`：`p_final = p_runs.mean(0)`，`G_hat = dist_median(p_final)`，`lo, hi = dist_interval(p_final)`。

## 任务 2：`code/scripts/solve_q1_sensitivity.py`（重写）

- 脚本最前面（`import numpy` 之前）设置三个线程环境变量为 1。参数：`--smoke`（只跑重复 0 的折 0、1，臂只取 `identity` 与 `alpha`，输出写入 `SMOKE_DIR`）、`--workers`（默认 8，不得超过 8）、`--arm NAME`（可重复，只计算这些臂的中间文件，不写最终文件）。
- 流程：`require_sklearn()`、`ensure_dirs()`；读 `q1_data.npz`（`xbar`、`G`、`date_group`）、`q1_main_oof.npz`（`oof_mu`、`fold_id`、`chosen_params`）、`q1_scores.npz`（`grade_median`、`p`、`G_hat`）；`load_q1()` 取 `X`、`beer`、`style`；`load_raw_ratings()`；`outer_splits(date_group, style)`，断言每次重复每折的测试行与 `fold_id` 一致。
- 单元为 `(臂, r, k)`。臂取 `ARMS` 加 `identity`（`scheme='col_mean'`、`mono_mode='chosen'`、不改表，只跑重复 0 的折 0、1，共 2 个单元），合计 8 × 25 + 2 = 202 个单元。`chosen = {组名: chosen_params[f'{r}_{k}_{组名}'] for 组名 in DIM_GROUPS}`。用 `joblib.Parallel(n_jobs=workers)` 并行；每个单元写 `CHUNK_DIR/sens/<臂>_r{r}_k{k}.npz`（键 `train_idx`、`test_idx`、`oof`、`inner`、`median`、`round`、`quant`、`p`），已存在则读取。
- α 臂：对每个外层折先 `y_fit = alpha_target(ratings, train_idx)`，再调用 `fit_fold_fixed(..., Y_fit=y_fit, ...)`，`unit_grades` 的 `y_tr = y_fit[tr]`。其余臂 `Y_fit = xbar`。
- 合并后，对每个臂：
  1. `merge_arm`、`delivered`；
  2. 与主设定比较：`W = bootstrap_weights(date_group)` 只生成一次，各臂共用；对每个臂调用 `cluster_bootstrap(pair_stat_fn(G, a, b, xbar), W)`，`a` 为该臂 `{'g': grade_median, 'p': p, 'mu': oof_mu}`，`b` 为主设定 `{'g': q1_scores 的 grade_median, 'p': q1_scores 的 p, 'mu': q1_main_oof 的 oof_mu}`；记录 `PAIR_STATS` 七项的 `point`、`lo`、`hi`；`wins_a`（该臂逐维 MAE 严格低于主设定的维度数）；
  3. `changed_delivered = changed_share(G_hat_臂, G_hat_主)`；`changed_median = changed_share(grade_median 臂, grade_median 主)`；
  4. `zero_heavy` 与其余维度的等级 MAE 之差（该臂减主设定），点估计即可。
- `alpha_flag`：α 臂等级 MAE 配对差的 `hi < 0`（方法论 §2.2）。只记录，不改动任何模型；为 True 时在回报中指出并自动开 `review/` 线程讨论目标假设，线程不阻塞 W8。
- barnyard 臂不重新拟合：超参数固定、其他维度的模型不依赖 barnyard，所以交付等级与主设定完全相同。该臂只报告主设定在 49 个维度（去掉 barnyard）上的等级 MAE、完全一致率、QWK 与 50 个维度的值，写入 `barnyard_excluded`，不做配对差与 CI，并写 `barnyard_reason`（说明等级不变的原因）。
- 正式完整运行时，为 Q3 G-S8 另拟合一个全数据 `drop_sum` 模型；`--smoke` 与 `--arm` 不生成它。目标为 `q1_data.npz` 的 `xbar`，输入删除四个 `_sum` 列后的 227 列；读取 W2 `q1_main_oof.npz` 的 `final_params`，沿用其每个维度组的参数与 `mono`，不重新选参。在全部 250 款上拟合 `ChemNAEncoder('col_mean')` 和 50 个维度模型，保存至 `code/data/processed/q1/q1_models_drop_sum/`（`encoder.joblib`、`<dim>.joblib`），并写 `q1_drop_sum_final.json` 记录输入列顺序、参数、目标、种子及源产物摘要。此模型仅供 Q3 G-S8，不参与 Q1 主输出或评价。
- 输出：
  - `PROCESSED_DIR/q1_sensitivity.npz`：对每个臂 `<臂>__oof_mu`（5, 250, 50）、`<臂>__grade_median`（5, 250, 50）、`<臂>__p_final`（250, 50, 7）、`<臂>__G_hat`（250, 50）；`arms`、`identity__oof`（2 个单元的外层预测，用于检查）、`checks`（JSON 列表）、`seed`、`meta`、`wall_time_s`。
  - `PROCESSED_DIR/q1_sensitivity.json`：键 `arms`（每臂的 `point`、`lo`、`hi`、`wins_a`、`changed_delivered`、`changed_median`、`zero_heavy_delta_mae`、`other_delta_mae`）、`barnyard_excluded`、`alpha_flag`、`checks`、`seed`、`meta`。`json.dump(..., allow_nan=False)`，NaN 写 null 并在同一条目写 `<字段>_reason`。
  - `PROCESSED_DIR/q1_drop_sum_final.json`：全数据 `drop_sum` 模型的来源与拟合清单；其模型文件与编码器写入 `PROCESSED_DIR/q1_models_drop_sum/`。
- 任一结构类检查项不通过时仍写出两个文件，随后以非零退出码结束并在终端列出失败项。

## 任务 3：`code/src/q1/external.py`（新建）与 `code/scripts/solve_q1_external.py`（重写）

方法论 §5.7：Zenodo *Validation Results 1–3* 是同一品评小组的另一次品评，含 8 个属性的 0–6 原始评分。读取方式与属性对应沿用 `experimentalCode/q1_supplement_checks.py` 的区块 D、E，不从该文件导入。

常量：
- `EXPECTED_MD5 = '52645e38d90aafda272532e19e2fea1c'`。
- `ATT = ('Malt', 'Sweetness', 'Hops', 'Bitterness', 'Esters', 'Fullness (body)', 'Alcohol (warming)', 'Overall appreciation')`。
- `ATT_TO_DIMS = {'Malt': ['A_malt_all', 'F_malt_all'], 'Sweetness': ['sweetness'], 'Hops': ['A_hops_all', 'F_hops_all'], 'Bitterness': ['bitternes'], 'Esters': ['A_esters_all', 'F_esters_all'], 'Fullness (body)': ['body'], 'Alcohol (warming)': ['alcohol'], 'Overall appreciation': ['overall']}`，共 11 个维度。
- `CHEM_MAP = {'Ethanol (v/v)': 'ethanol..v.v.', 'Ethyl acetate (mg/l)': 'ethyl_acetate', 'Ethyl hexanoate (mg/l)': 'ethyl_hexanoate', 'Isoamyl acetate (mg/l)': 'isopentyl_acetate', 'Phenylethyl acetate (mg/l)': 'phenethyl_acetate', 'Glycerol (g/l)': 'glycerol', 'Lactic acid (g/l)': 'lactic_acid.mg.L.'}`。
- `BASE_STYLES = ('blond', 'NABLAB')`；`EXPECTED_BASE_CODES = {'blond': 64, 'NABLAB': 223}`；`EXPECTED_CONTROL_N = {'blond': 33, 'NABLAB': 13}`。

函数：
- `resolve_workbook(env=os.environ) -> Path`：路径取环境变量 `ZENODO_XLSX`；未设置时取 `Path(env.get('TEMP', '.')) / 'zenodo_10653704' / 'Supplemental Files and Figure source files.xlsx'`。文件不存在时 `raise FileNotFoundError`，信息写明下载地址 https://doi.org/10.5281/zenodo.10653704 与设置方法。不得硬编码任何个人电脑路径。
- `check_md5(path) -> str`：计算 md5，不等于 `EXPECTED_MD5` 时 `raise RuntimeError`，返回 md5。
- `load_control_ratings(path) -> DataFrame`：读 `Validation Results 1`、`2`、`3`，`date = pd.to_datetime(Time).dt.date`，合并；记 `n_all`；`drop_duplicates(subset=['date', 'Name', 'Beer'] + list(ATT))`，记去重数；读 `Validation Details`（取 `SetName` 非空的行，`date = pd.to_datetime(Date).dt.date`），按 `left_on=['date', 'Beer']`、`right_on=['date', 'Code']` 合并 `BeerStyle`、`SampleName`，断言 `SampleName` 无缺失；只保留 `SampleName == 'Control'`；返回该表与 `{'n_all': ..., 'n_dedup': ...}`。
- `match_base_beers(path, cal) -> dict`：`cal` 为附件 8 `Calibrated chemical dataset` 工作表。读 `Validation Concentrations`（`drop_duplicates('BeerStyle')`，以 `BeerStyle` 为索引）；`L = np.log10(cal[list(CHEM_MAP.values())].to_numpy(float) + 1e-3)`，`sd = np.nanstd(L, 0)`；对每个 `BASE_STYLES` 中的风格，`v = np.log10(conc.loc[风格, list(CHEM_MAP)].to_numpy(float) + 1e-3)`，`dist = np.sqrt(np.nansum(((L - v) / sd) ** 2, axis=1))`；返回 `{风格: {'code': 最近者的 beer 列取值, 'dist': 最近距离, 'dist2': 第二近距离}}`。风格名按大小写不敏感匹配。不得直接写入 64 与 223，匹配结果必须由数据算出。
- `control_means(ctrl, style_name) -> DataFrame`：对该风格（`BeerStyle.str.lower()` 比较）的对照杯评分，按 `ATT` 给出 `attr`、`n`、`mean`、`se`（`std(ddof=1) / sqrt(n)`）；再按 `ATT_TO_DIMS` 展开到 11 个维度，列为 `dim`、`attr`、`n`、`mean`、`se`。
- `compare_rows(code, rows_of_beer, table, G_hat, G_hat_Q, G_obs, xbar, p_final, dims) -> DataFrame`：`code` 是附件 `beer` 列取值，`rows_of_beer` 是其行下标；对 `table` 的每个维度，取 `grade(mean)` 作对照杯等级，列出 `G_hat`、`G_hat_Q`、`G_obs`、`xbar`、`ctrl_mean`、`ctrl_se`、`ctrl_grade`、`p_hat_at_ctrl`（`p_final` 在 `ctrl_grade` 处的概率）、`err_hat = G_hat - ctrl_grade`、`err_obs = G_obs - ctrl_grade`。
- `summarize(df) -> dict`：对每款啤酒与合计：单元数、`G_hat` 对 `ctrl_grade` 的 MAE、完全一致数、±1 以内数、平均有符号误差；`G_obs` 对 `ctrl_grade` 的同样几项（反映两次品评之间的可重复程度）。

`solve_q1_external.py`：
- 开头 `require_sklearn()`、`ensure_dirs()`；参数 `--smoke`（输出写入 `SMOKE_DIR`，其余同正式）。
- 流程：`resolve_workbook()`、`check_md5()`（文件缺失或 md5 不符时以非零退出码结束，只打印原因，不写任何输出文件，不得写 PASSED 或任何占位结果）；读 `data/raw/8.xlsx` 的 `Calibrated chemical dataset`；`match_base_beers`，断言每款的 `code` 与 `EXPECTED_BASE_CODES` 一致（不一致即 `AssertionError`）；由 `load_q1().beer` 找行：`rows = np.where(beer.to_numpy() == code)[0]`，断言恰有 1 行（`beer` 是附件的 `beer` 列取值，不是行号，见方法论 §1.2）。
- 先读 `q1_metrics.json` 的 `success`。True 时从 W3 `q1_scores.npz` 读取交付的 `G_hat`、`G_hat_Q`、`p_final`；False 时从 W4 `q1_baselines.npz` 读取 `chemstyle_final_G_hat`、`chemstyle_final_G_hat_Q`、`chemstyle_final_p`。两种来源均为五次外层折外结果汇总，不使用全数据拟合值评价原样本。再读 `q1_data.npz`（`G`、`xbar`、`dims`），对两款啤酒生成对照表；`G_obs` 只作与对照杯重复性的参照。
- 输出 `PROCESSED_DIR/q1_external.npz`：`model_key`（`main` 或 `chemstyle`）、`beer_codes`、`beer_rows`、`match_dist`（2）、`match_dist2`（2）、`dims`（11）、`ctrl_mean`（2, 11）、`ctrl_se`（2, 11）、`ctrl_n`（2）、`ctrl_grade`（2, 11）、`G_hat`（2, 11）、`G_hat_Q`（2, 11）、`G_obs`（2, 11）、`xbar`（2, 11）、`p_final`（2, 11, 7）、`summary`（JSON）、`md5`、`n_all`、`n_dedup`、`checks`（JSON 列表）、`seed`、`meta`。另写 `q1_external.json`，包含相同的 `model_key`、汇总和逐行表。结果只报告，不用于选择。

## 任务 4：`code/scripts/solve_q1_stress.py`（重写）

- 脚本最前面设置三个线程环境变量为 1。参数 `--smoke`（只跑 2 个风格与论文划分的前 3 个维度，输出写入 `SMOKE_DIR`）、`--workers`（默认 8，不得超过 8）。
- 留一风格：`leave_one_style_splits(style)` 返回 22 个 `(train_idx, test_idx, name)`。对每个划分调用 W2 的 `fit_outer_fold(X, xbar, dims, DIM_GROUPS, date_group, style, train_idx, test_idx, inner_seed=SEED + 100)`，在该训练集内重新做嵌套选参（不得沿用主运行的超参数，否则含测试风格的信息）；再调用 `unit_grades`。基线：`simple_fold('b0', ...)`、`simple_fold('b1', ...)`（B1 遇到未见过的风格退回全局均值），同样经 `unit_grades` 离散。每个划分写 `CHUNK_DIR/stress/loso_{i}.npz`（`i` 是风格在排序后列表中的序号，不要用风格名作文件名：`Low/No alcohol`、`Pils/Lager`、`Brett/cofermented` 含斜杠）。
- 论文 70/30：`paper_split(style, seed=SEED)` 得到训练 175 款、测试 75 款，调用 `fit_outer_fold(..., inner_seed=SEED + 100)`、`simple_fold` 的 b0 与 b1，`unit_grades` 同上。
- 合并：`loso_oof_mu`（250, 50）、`loso_grade_median`、`loso_grade_round`、`loso_grade_quant`（250, 50）、`loso_p`（250, 50, 7）；`b0`、`b1` 同样各一份；论文划分的对应数组放在 `paper_` 前缀下，仅测试行有效。断言留一风格的 22 个测试集互不相交且并集为 250 款；断言每个测试风格在对应训练集中不出现。
- 报告（写入 `PROCESSED_DIR/q1_stress.json`）：
  1. 逐风格：`style`、`n`、`small`（`n <= 5`，方法论 §5.1 规定只作定性解读）、`mae`（`median` 与 `round`）、`exact`、`within1`、连续 `cont_mae`、`spearman`（逐维 Spearman 对 50 维取平均，常数预测的维度剔除并记个数）；同一风格上 `b0`、`b1` 的 `mae` 与 `cont_mae`；QWK 在 `n < 10` 时写 null 并附 `qwk_reason`。
  2. 合并（250 款各预测一次）：主模型、b0、b1 各自的 `mae`、`exact`、`within1`、`qwk`、`cont_mae`、`cont_r2`、`cont_spearman`；`main - b1` 与 `main - b0` 的配对差与 95% CI（`bootstrap_weights(date_group)` 与 `pair_stat_fn`，数组取 `R=1`）。
  3. 论文划分：同上各项，在 75 款测试集上计算。
  4. 同日期组暴露：留一风格时，测试风格啤酒所在日期组仍有训练啤酒的款数（只报告，不改变划分；日期组由多个风格混合，纯度中位数 0.92，见方法论 §2.4）。
  5. 主划分对照：主运行外层折外结果（`q1_scores.npz` 的 `grade_median` 与 `q1_main_oof.npz` 的 `oof_mu`）在 `small` 风格上的 `mae`，用于与留一风格对比。
- 输出 `PROCESSED_DIR/q1_stress.npz`：`loso_styles`（22）、`loso_n`（22）、上述数组、`paper_test_idx`、`checks`（JSON 列表）、`seed`、`meta`、`wall_time_s`；另写 `q1_stress.json`。
- 任一结构类检查项不通过时仍写出文件，随后以非零退出码结束并在终端列出失败项。

## 任务 5：测试

`code/tests/test_q1_sensitivity.py`：
- `alpha_target`：人造评分（2 名品评员，品评员效应固定为 +0.5、−0.5，6 款啤酒，每款 2 名品评员各评一次）上，训练行的 α 还原出啤酒效应；改变测试啤酒的评分后，训练行的 α 完全不变；测试行为 NaN。
- `apply_att8`：人造 3 款啤酒的小表与两行差异表（用 `monkeypatch` 指向临时 CSV），验证只有指定单元被替换、原表不被修改、当前值与 `csv1_literal` 不符时抛 `AssertionError`；真实数据上单元数为 80、啤酒数为 24。
- `arm_kwargs`：`drop_sum` 后化学表为 227 列且不含四个 `_sum` 列；`att8` 表与原表不同的单元恰为 80 个；未知臂名抛 `ValueError`。
- `fit_fold_fixed` 恒等：真实数据重复 0 的折 0，`dim_groups={'Basic': ['alcohol', 'body']}`、`dims=['alcohol', 'body']`、`configs` 取 `max_depth=2, min_samples_leaf=10, l2_regularization=0.0` 的 3 项：先运行 `fit_outer_fold` 得 `chosen`、`oof`、`inner`；再用同一 `chosen` 运行 `fit_fold_fixed`，`oof` 与 `inner` 在 `atol=1e-10` 内相等。
- `fit_fold_fixed` 无泄漏：把 `Y_fit` 的测试行换成 `np.random.default_rng(1).uniform(0, 6)` 后，`oof` 与 `inner` 不变；训练行加 1 后 `oof` 必须改变。
- `mono_mode`：`'off'` 时 `monotone_cst` 全为 0；`'on'` 时对 `alcohol` 只有乙醇列为 +1；`'chosen'` 取 `best['mono']`。
- `merge_arm`：用 2 个人造单元验证重复写入时报错、`inner_mu` 的非 NaN 行恰为训练行。
- `changed_share`：手算 4 个单元。

`code/tests/test_q1_external.py`：
- `resolve_workbook`：环境变量指向不存在的文件时抛 `FileNotFoundError`，信息含下载地址；不得出现个人电脑路径（断言异常信息与源码中不含 `C:\Users`，源码检查用 `Path(external.__file__).read_text()`）。
- `check_md5`：临时文件的 md5 不符时抛 `RuntimeError`。
- `control_means`：人造对照杯评分表（两种风格、大小写不同的风格名）上，均值、标准误、`n` 与手算一致，并正确展开到 11 个维度。
- `compare_rows`：`G_hat` 与 `ctrl_grade` 的差、`p_hat_at_ctrl` 与手算一致；主模型与 chemstyle 两种输入来源均测试。
- 真实工作簿相关的测试用 `pytest.mark.skipif` 在文件不存在时跳过；文件存在时断言 md5 通过、两款啤酒的匹配结果为 64 与 223、对照杯评分条数为 33 与 13。

## 验收

命令（仓库根目录）：
```
PY=/e/anaconda3/python.exe
export ZENODO_XLSX='<Zenodo 工作簿的实际路径>'
$PY code/scripts/solve_q1_sensitivity.py --smoke
$PY code/scripts/solve_q1_stress.py --smoke
$PY code/scripts/solve_q1_external.py --smoke
$PY code/scripts/solve_q1_sensitivity.py --workers 8
$PY code/scripts/solve_q1_stress.py --workers 8
$PY code/scripts/solve_q1_external.py
cd code && $PY -m pytest tests -q
```

运行时间：每个敏感性单元约 250 次拟合，202 个单元用 8 个进程约 1.5–2.5 小时，先启动；留一风格与论文划分共 23 个嵌套折，约 30–60 分钟；外部检查数秒。

`checks`（每项含 `name`、`observed`、`expected`、`tol`、`passed`，`passed` 由代码比较得出）：

结构类（任一不通过则以非零退出码结束；冒烟与正式运行都做）：

| 名称 | 预期 | 容差 |
|---|---|---|
| 敏感性外层划分与 `q1_main_oof.npz` 的 `fold_id` 一致 | 是 | 精确 |
| 恒等臂的外层预测与主运行 `oof_mu` 逐元素相同（重复 0 的折 0、1） | 是 | 1e-10 |
| 各臂 `inner_mu` 的非 NaN 行恰为训练行 | 是 | 精确 |
| 各臂 `oof_mu` 全部有限（`na_native_nan` 臂的预测也须有限） | 是 | 精确 |
| 各臂每次重复每款啤酒恰好测试一次 | 是 | 精确 |
| 各臂 `p_final` 每行和为 1 | 是 | 1e-12 |
| 各臂所用超参数与主运行 `chosen_params` 逐字段相同 | 是 | 精确 |
| `att8` 臂替换单元数 / 啤酒数 | 80 / 24 | 精确 |
| `drop_sum` 臂化学表列数 | 227 | 精确 |
| Q3 `drop_sum` 全数据模型参数与 W2 `final_params` 逐字段相同 | 是 | 精确 |
| Q3 `drop_sum` 全数据模型输入列数与模型数量 | 227 列、50 个维度模型及 1 个编码器 | 精确 |
| `na_native_nan` 臂的 NaN 单元数 | 73 | 精确 |
| `na_median_ind`、`na_floor_ind` 臂的缺失指示列数 | 9、8 | 精确 |
| `alpha` 臂的训练目标只含训练行评分（改变测试啤酒评分后 α 不变） | 是 | 精确 |
| 留一风格：22 个测试集互不相交，并集为 250 款 | 是 | 精确 |
| 留一风格：测试风格在对应训练集中不出现 | 是 | 精确 |
| 论文划分：训练 / 测试款数 | 175 / 75 | 精确 |
| 外部检查：工作簿 md5 | `52645e38d90aafda272532e19e2fea1c` | 精确 |
| 外部检查：匹配到的底酒 `beer` 列取值（blond、NABLAB） | 64、223 | 精确 |
| 外部检查：对照杯评分条数（blond、NABLAB） | 33、13 | 精确 |
| 外部检查：每款啤酒匹配到的行恰为 1 行 | 是 | 精确 |
| 外部检查：对照杯表的维度数 | 11 | 精确 |
| 所有结果 JSON 中没有字符串形式的数值或状态（除 `_reason` 字段） | 是 | 精确 |

数值类（只在正式运行做；这些是合理性检查，失败时 `passed=false` 并照实回报，不改方法，也不阻塞 W8）：

| 名称 | 预期 | 容差 |
|---|---|---|
| `alpha - main` 的 `cont_mae` 点估计（预运行 −0.0026） | −0.0026 | ±0.004 |
| 各臂（不含 barnyard 排除）等级 MAE 与主设定之差的绝对值 | ≤ 0.03 | 上界 |
| 各臂 `changed_delivered`（交付等级变化的单元比例） | ≤ 0.15 | 上界 |
| `mono_on` 与 `mono_off` 的等级 MAE 之差的绝对值 | ≤ 0.03 | 上界 |
| 留一风格合并后主模型 `cont_mae` 低于 b0 的 `cont_mae` | 是 | 严格 |

回报中另列（不作为通过条件）：

- 各臂相对主设定的 `mae`、`qwk`、`exact`、`cont_mae` 点估计与 95% CI（表格），`wins_a`，`changed_delivered`，`changed_median`，零分为主维度与其余维度的 MAE 之差。
- `alpha_flag` 的取值。
- 留一风格 22 个风格的表（`n`、`small`、`mae`、`exact`、`spearman`、`cont_mae`，及 b0、b1 的 `mae`）；合并与论文划分的汇总；主模型与 b1 的配对差与 CI。
- 外部检查：两款啤酒各 11 行的对照表（`ctrl_mean`、`ctrl_se`、`ctrl_grade`、`G_hat`、`G_obs`）与汇总（`G_hat` 对 `ctrl_grade` 的 MAE、完全一致数、平均有符号误差；`G_obs` 对 `ctrl_grade` 的同样几项）；匹配距离 `dist` 与 `dist2`。啤酒 223 只有 13 个评分，需在表中同时给出 `ctrl_se`。
- 三个脚本各自的 `wall_time_s`。

其余：Q1 范围 `black --check`、`ruff check` 通过；`code/src/q1/` 无 `__main__`；`git status` 中 `code/data/processed/q1/` 新增 `q1_sensitivity.*`、`q1_stress.*`、`q1_external.*`。

## 禁止事项

- 不得在代码、`checks` 或输出中写入任何结果数值、状态字符串（含 PASSED、SKIPPED）或占位结果。
- 外部检查缺少工作簿或 md5 不符时，不得生成任何输出文件，不得用 NaN 或 0 代替；以非零退出码结束并在回报「卡点」中写明。
- 不得硬编码底酒编号 64、223 作为匹配结果；不得把附件 `beer` 列取值当作行号使用。
- α 目标必须只用训练啤酒的评分估计，不得用全部 250 款；各臂不得另做超参数调参。
- 留一风格与论文划分必须在各自训练集内重新嵌套选参，不得使用主运行的超参数或全数据重拟合的超参数。
- 不得在 xbar 上加噪声来构造「变化比例」。
- 不得根据敏感性结果、压力测试或外部检查改动主模型、规则、超参数或容差。

## 通用约定

执行顺序：W1 → W2 → W3 → W4 与 W5（可并行）→ W6 → W7 → W8；W9（OAV，可选）在 W2 之后随时可做。每个工单单独领取、单独验收。

### 方法论中已定的决定

- OAV 类别名单见 `model/q1_methodology.md` §3.2；B2 的设定与网格见 §5.3；正文报告啤酒的选取规则见 §5.2。这些不在工单中另行决定，工单与方法论冲突时以方法论为准，并在回报中写明。
- 敏感性对照臂和 K2 沿用主运行在各外层折、各维度组选定的超参数，不单独做嵌套 CV。
- 5 次重复先平均 `p` 再取中位数；`G_hat_Q` 取 5 次分位数切点等级的中位数。

### 运行环境

- 正式运行用 `PY=/e/anaconda3/python.exe`：Python 3.13.9、scikit-learn 1.7.2、numpy 2.3.5、pandas 2.3.3，已装 pytest、black、ruff。`E:/anaconda3/envs/mathorcup`（scikit-learn 1.9.1）和 PATH 上默认的 `python`（3.14、scikit-learn 1.9.0、无 pytest）不用于正式运行。
- 每个求解脚本开头调用 `require_sklearn('1.7.2')`，版本不符即退出；`meta` 记录实际版本。
- `HistGradientBoostingRegressor` 在 `warm_start` 下，`max_iter` 取 100、200、400 与分别拟合的结果一致（本机核对，最大差 0.0）。一次拟合到 400 轮约 1 秒（单线程）。
- Q1 范围（black、ruff 只检查这些文件，不动 Q2、Q3 代码）：`code/src/q1/`、`code/scripts/solve_q1_*.py`、`code/scripts/export_q1_*.py`、`code/scripts/fig_q1_*.py`（不含 `fig_q1_methodology_flowchart.py`）、`code/scripts/tab_q1_*.py`、`code/tests/`。
- 不要求逐元素复现预运行。完整执行实验，各项数值与方法论中的预运行数字在工单给定容差内一致即可。

### 执行规则

1. 一次只领一个工单；领取时把 `status` 改为 `ack`。
2. 只创建或修改本工单列出的文件。不得新建 `final_*`、`fix_*`、`gen_*`、`patch_*`、`perfect_*` 一类补丁脚本；需要改动时修改原文件。
3. 代码中不得写入任何结果数值，包括指标、MAE、R²、状态字符串和 `checks` 的布尔值。输出的每个数由本次运行算出。`checks` 为列表，每项含 `name`、`observed`、`expected`、`tol`、`passed`，`passed` 由代码比较得出。
4. 预运行数值仅作回归核对。超出容差时写明观测值与预期值，不修改方法或容差；偏差超过容差 3 倍时自动开 `review/` 线程并附复现材料，但仅数值偏差不阻断当前任务或独立下游任务。只有方法预先定义的结构不变量失败、产物不完整或方法无法解释的实质偏离，才暂停依赖该产物的任务并在对应线程定位问题。
5. 不吞异常：不用 `try/except` 返回 0 或 NaN 代替失败；文件缺失即抛错退出。指标确实不适用时写 NaN，并在同一输出中写原因字符串。
6. `code/src/q1/` 不得有 `__main__`，导入时不得读写文件、建目录或绘图。目录由脚本调用 `paths.ensure_dirs()` 创建。
7. 求解脚本不绘图，绘图脚本不求解；绘图脚本只读 `code/data/processed/q1/`。
8. 先写 pytest 再写实现。测试放 `code/tests/`，秒级完成。
9. 每个求解脚本接受 `--smoke`：1 次重复、2 个外层折、bootstrap 20 次，输出写入 `code/data/processed/q1/_smoke/`。冒烟和测试通过后才启动正式运行。
10. 耗时任务按单元把中间结果写入 `code/data/processed/q1/_chunks/<任务名>/<单元>.npz`；已存在的单元跳过；最终 npz 由中间文件合并，合并前断言单元个数。`_smoke/` 与 `_chunks/` 已加入 `.gitignore`（W1 任务 0.7）。
11. 并行：`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 设为 1；用 joblib 进程池，进程数不超过 8，并行只在外层折、重复或维度组上展开。
12. 种子：`SEED = 0`。外层第 r 次重复（r=0..4）用 `SEED + r`，内层用 `SEED + 100 + r`；树模型 `random_state=SEED`；bootstrap 用 `np.random.default_rng(SEED)`，2,000 次；每个 npz 存 `seed`。
13. 格式：`PY -m black --check` 与 `PY -m ruff check` 对 Q1 范围通过（配置见 W1 任务 0.5）。git commit 须先征得用户同意。
14. 卡住时，在回报中写「卡点」：文件、命令、报错全文、已尝试的做法。不绕过、不降级、不写占位结果。
15. 回报格式：运行命令、耗时、产物路径与 SHA-256 前 12 位、验收项逐条的观测值与通过状态、偏离工单之处。

## 处理结果

coding 回报（2026-10-29，状态 ack，待 modeling 验收）。

运行命令（`PY=C:/Users/11600/anaconda3/envs/mathorcup_new/python.exe`；工作簿取默认路径 `$TEMP/zenodo_10653704/`，md5 与 `EXPECTED_MD5` 一致）：按「验收」七条命令依次执行。

耗时：敏感性 1522 秒；压力测试 7348 秒；外部检查数秒。

产物（SHA-256 前 12 位）：`q1_sensitivity.npz` `0138d1c13109`、`q1_sensitivity.json` `a8eb75f4c524`、`q1_stress.npz` `f048ac97cd7e`、`q1_stress.json` `b82dc5a44bc2`、`q1_external.npz` `489244ed5bec`、`q1_external.json` `75e9cc68e6f8`、`q1_drop_sum_final.json` `e646141fba2e`；`q1_models_drop_sum/` 含编码器与 50 个维度模型（被忽略）。

`checks`：敏感性 19 项、压力测试 5 项、外部检查 6 项，全部通过。

敏感性各臂（臂减主设定，点估计 [95% CI]；`changed_delivered`）：

| 臂 | 等级 MAE | QWK | 连续 MAE | 交付变化 |
|---|---|---|---|---|
| na_median_ind | −0.0003 [−0.0016, +0.0009] | +0.0010 | −0.0000 | 1.35% |
| na_native_nan | −0.0001 [−0.0018, +0.0013] | +0.0011 | −0.0000 | 1.58% |
| na_floor_ind | +0.0005 [−0.0013, +0.0026] | +0.0003 | −0.0002 | 2.00% |
| drop_sum | −0.0002 [−0.0017, +0.0011] | +0.0003 | −0.0001 | 1.82% |
| att8 | +0.0020 [+0.0006, +0.0032] | −0.0034 | +0.0001 | 1.75% |
| mono_on | −0.0001 [−0.0004, +0.0002] | +0.0003 | +0.0000 | 0.09% |
| mono_off | −0.0002 [−0.0005, +0.0002] | +0.0001 | +0.0000 | 0.07% |
| alpha | −0.0002 [−0.0034, +0.0028] | −0.0020 | −0.0030 | 5.22% |

`alpha_flag` = False。

留一风格（22 个风格，主模型规则为中位数；`small` 为款数 ≤ 5）：

| 风格 | n | small | 等级 MAE | 完全一致 | Spearman | 连续 MAE | b0 MAE | b1 MAE |
|---|---|---|---|---|---|---|---|---|
| Pils/Lager | 11 | 否 | 0.335 | 0.682 | 0.185 | 0.340 | 0.455 | 0.455 |
| Lambic | 11 | 否 | 0.369 | 0.660 | 0.341 | 0.371 | 0.524 | 0.493 |
| Dubbel | 10 | 否 | 0.382 | 0.652 | 0.193 | 0.390 | 0.378 | 0.374 |
| Brown | 12 | 否 | 0.400 | 0.620 | 0.334 | 0.349 | 0.422 | 0.465 |
| Faro | 3 | 是 | 0.407 | 0.687 | −0.101 | 0.469 | 0.560 | 0.513 |
| Wheat | 12 | 否 | 0.410 | 0.605 | 0.288 | 0.371 | 0.468 | 0.472 |
| Blond | 31 | 否 | 0.422 | 0.621 | 0.201 | 0.426 | 0.463 | 0.463 |
| Brett/cofermented | 6 | 否 | 0.423 | 0.603 | 0.364 | 0.396 | 0.523 | 0.530 |
| Amber | 12 | 否 | 0.448 | 0.603 | 0.279 | 0.441 | 0.480 | 0.487 |
| West Flanders ale | 7 | 否 | 0.463 | 0.580 | 0.131 | 0.470 | 0.537 | 0.489 |
| Flanders old brown | 5 | 是 | 0.476 | 0.564 | 0.155 | 0.410 | 0.476 | 0.424 |
| Tripel | 28 | 否 | 0.486 | 0.569 | 0.122 | 0.481 | 0.465 | 0.475 |
| Low/No alcohol | 8 | 否 | 0.487 | 0.575 | 0.240 | 0.543 | 0.590 | 0.570 |
| Saison | 10 | 否 | 0.510 | 0.576 | 0.165 | 0.486 | 0.598 | 0.618 |
| Hoppy | 17 | 否 | 0.513 | 0.579 | 0.251 | 0.528 | 0.654 | 0.648 |
| Kriek | 10 | 否 | 0.514 | 0.552 | 0.219 | 0.532 | 0.666 | 0.626 |
| Scotch | 5 | 是 | 0.524 | 0.556 | 0.085 | 0.522 | 0.656 | 0.672 |
| Strong ale | 18 | 否 | 0.553 | 0.521 | 0.170 | 0.508 | 0.717 | 0.739 |
| Christmas | 9 | 否 | 0.571 | 0.524 | 0.223 | 0.534 | 0.651 | 0.633 |
| Stout/Porter | 13 | 否 | 0.582 | 0.523 | 0.157 | 0.544 | 0.626 | 0.648 |
| Fruitbeer | 8 | 否 | 0.662 | 0.460 | −0.051 | 0.637 | 0.802 | 0.743 |
| Brut | 4 | 是 | 0.670 | 0.475 | 0.393 | 0.696 | 0.740 | 0.730 |

合并（250 款各预测一次）：主模型等级 MAE 0.4754、连续 MAE 0.4659；b0 0.5494 / 0.5581；b1 0.5473 / 0.5581；`main − b1` 等级 MAE −0.0719 [−0.0951, −0.0473]，`main − b0` −0.0741 [−0.1002, −0.0473]。论文 70/30 划分（75 款）：主模型 0.4699 / 0.4606；b0 0.5381 / 0.5514；b1 0.4840 / 0.4839；`main − b1` −0.0141 [−0.0363, +0.0064]。同日期组暴露 139 款。主划分在 4 个 small 风格（17 款）上的等级 MAE 0.539。

外部检查（交付来源 `main`）：匹配 blond → 64（距离 0.069，第二近 0.420）、NABLAB → 223（距离 0.061，第二近 9.083）；对照杯评分 33、13 条。`G_hat` 对对照杯等级：64 号 MAE 0.909、完全一致 3/11、平均有符号误差 −0.727；223 号 0.364、7/11、−0.182；`G_obs` 对对照杯等级：64 号 0.455、6/11；223 号 0.273、8/11。

偏离工单之处：
- 工作簿取自 Zenodo 记录 10653704 官方下载；`~/Downloads` 下同名文件 md5 不符，未使用。
- 冒烟时外部检查与正式运行内容相同（只是输出目录不同）。
- 压力测试 `--smoke` 只跑款数最少的 2 个风格。
- 各脚本复用 `solve_q1_metrics.py` 的 `sanitize`、`has_string_numbers`，以及 `solve_q1_data.py` 的 `make_check`。

modeling 验收（2026-10-29）：通过。

- `q1_sensitivity` 19/19、`q1_stress` 5/5、`q1_external` 6/6 内置检查通过；α 目标的品评员效应只用训练啤酒评分估计。偏离项全部接受。
- 敏感性结论：8 个臂中只有附件 8 冲突单元臂的等级 MAE 配对差 CI 不含 0（+0.0020 [+0.0006, +0.0032]），其余均含 0；交付等级变化比例 0.07%–5.2%（α 目标最大）。`alpha_flag` 为 False，不开线程。
- 留一风格的测试啤酒中有 139 款与训练啤酒同属一个日期组，该划分不比主划分更严格；写作只作定性解读。
- 核对方式：在与正式运行相同版本的环境（Python 3.10.22、scikit-learn 1.7.2、numpy 2.2.6、pandas 2.3.3、scipy 1.15.3）中运行 `cd code && $PY -m pytest tests -q` 与 Q1 范围的 black、ruff；读取正式产物内置 `checks`；对照方法源文件点查关键实现。未重新求解。
