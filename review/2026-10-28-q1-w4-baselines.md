---
id: 2026-10-28-q1-w4-baselines
from: modeling
to: coding
status: open
severity: major
refs:
  - review/2026-10-28-q1-index.md
  - review/2026-10-28-q1-w2-main-model.md
  - review/2026-10-28-q1-w3-scores.md
  - model/q1_methodology.md
  - code/src/q1/baselines.py
  - code/scripts/solve_q1_baselines.py
---

## 问题

- 实现方法论 §5.3 中除 K2、C1、C2 以外的全部基线与对照：B0、B0g、B1、线性回归、ElasticNet、随机森林、化学加风格、B2。K2、C1、C2 在 W5。
- 现有实现的错误：标为 EN 的对照实际是主模型参数的 HistGBR（与主模型折外预测相关 0.99994）；RF 在 `q1_metrics.json` 中为 0.0；线性回归、B0g、化学加风格缺失；B2 的目标用了 x̄，应为附件 4 的 z；对照只取第 0 次重复、只用四舍五入。
- 所有连续对照与主模型使用同一套外层划分和同一种子，像主模型一样产生 `oof` 与 `inner`，再由 W3 的 `apply_rules` 用各自的残差池离散。
- 只创建或修改：`code/src/q1/baselines.py`、`code/scripts/solve_q1_baselines.py`、`code/tests/test_q1_baselines.py`。
- 依赖：W1（`ChemNAEncoder`、`DIMS`、`DIM_GROUPS`、`FLOOR_VALUE`、`load_q1`）、W2（`outer_splits`、`inner_splits`、`paper_split`、`fit_outer_fold`、`CONFIGS`、`q1_main_oof.npz`）、W3（`apply_rules`、`grade`、`dist_median`、`q1_scores.npz`）。先读 `review/2026-10-28-q1-index.md` 的「通用约定」。

## 模型清单

键名 `MODELS = ('b0', 'b1', 'lin10', 'enet', 'rf', 'chemstyle')`；B0g 与 B2 单独处理。

| 键 | 模型 | 输入 | 调参 |
|---|---|---|---|
| b0 | 训练折均值 | 无 | 无 |
| b1 | 风格均值向全局均值收缩，k=5 | 风格 | 无 |
| lin10 | 训练折内按 Spearman 取前 10 个特征的最小二乘 | 列均值编码、分位数变换、标准化，加所选列的下限指示 | 无 |
| enet | ElasticNet | 全部 231 列加下限指示，同上变换 | `alpha` ∈ logspace(−3, 1, 9) × `l1_ratio` ∈ {0.1, 0.5, 0.9}，共 27 项，按维度组共享 |
| rf | RandomForestRegressor | 列均值编码后的 231 列 | `max_features` ∈ {0.33, 'sqrt'} × `min_samples_leaf` ∈ {5, 10}，共 4 项，`n_estimators=500` |
| chemstyle | 主模型加风格 one-hot | 231 列加 22 个风格列 | 与主模型相同的 24 项 × 单调开关，直接调用 W2 的 `fit_outer_fold` |

B0g（训练折众数等级）不是单独的模型：W3 的 `const` 规则只依赖训练折等级，B0 的 `const` 规则输出就是 B0g，也必须与主模型的 `grade_const` 逐元素相同，作为划分一致性的检查。

调参标准与主模型一致：内层 4 折平均的 RMSE/SD（`sd = np.std(y_val, ddof=0)`），在每个外层折、每个维度组各选一次，得分相同取排在前面的配置。

## 任务 1：`baselines.py`（重写）

常量：
- `B1_K = 5`。
- `LIN_CONFIGS = [{}]`。
- `EN_CONFIGS = [{'alpha': float(a), 'l1_ratio': l1} for l1 in (0.1, 0.5, 0.9) for a in np.logspace(-3, 1, 9)]`。
- `RF_CONFIGS = [{'n_estimators': 500, 'max_features': mf, 'min_samples_leaf': leaf} for mf in (0.33, 'sqrt') for leaf in (5, 10)]`。
- `RULES = ('const', 'round', 'median', 'quant', 'qwk_cut', 'median_knn')`。
- `B2_GRID = {'n_estimators': [100, 200, 400], 'max_depth': [2, 3, 4], 'learning_rate': [0.05, 0.1]}`。

简单基线（逐维向量化，`y_tr` 为训练行的 x̄）：
- `b0_mean(y_tr, n_te) -> ndarray`：`np.full(n_te, y_tr.mean())`。
- `b1_style_shrunk(y_tr, style_tr, style_te, k=B1_K) -> ndarray`：对每个测试行，`n_s` 为 `style_tr` 中与其同风格的行数，`mean_s` 为这些行的 `y_tr` 均值，预测 `(n_s * mean_s + k * y_tr.mean()) / (n_s + k)`；`n_s = 0` 时取 `y_tr.mean()`。
- `simple_fold(kind, Y, style, groups, train_idx, test_idx, inner_seed) -> dict`：`kind` 为 `'b0'` 或 `'b1'`，`Y` 为 250×50 的 x̄。先把 `train_idx`、`test_idx` 排序。`oof` 形状 `(n_te, 50)`：用全部训练行拟合后预测测试行。`inner` 形状 `(n_train, 50)`：对 `inner_splits(train_idx, groups, style, inner_seed)` 的每个内层折 `(a, b)`，用行 `a` 拟合、预测行 `b`，写入位置 `np.searchsorted(train_idx, b)`。返回 `{'train_idx', 'test_idx', 'oof', 'inner'}`。

特征准备。`prep(raw_fit, raw_others) -> (X_fit, [X_other, ...], names)`，`raw_fit`、`raw_others[i]` 是原始化学表的行子集（DataFrame，含 NaN）：
- `prep_plain`：`enc = ChemNAEncoder('col_mean').fit(raw_fit)`，对 `raw_fit` 与各 `raw_others` 编码，转 float64 数组；`names` 为列名列表。
- `prep_linear`：
  1. 同 `prep_plain` 得到 `A_fit`、`A_other`。
  2. `QuantileTransformer(n_quantiles=min(200, len(raw_fit)), output_distribution='uniform', random_state=SEED)` 在 `A_fit` 上拟合并变换全部矩阵。
  3. 下限指示列：`fc = [c for c in raw_fit.columns if (raw_fit[c].round(8) == FLOOR_VALUE).any()]`，列集合只由 `raw_fit` 决定；每个矩阵的指示为 `(raw[fc].round(8) == FLOOR_VALUE).to_numpy(float)`（NaN 比较为假，记 0）。不要调用 `data.floor_indicators`，它的列集合随输入变化。
  4. 把分位数输出与指示列横向拼接，`StandardScaler` 在 `raw_fit` 部分拟合并变换全部矩阵（零方差列变为 0，不报错）。
  5. `names = list(raw_fit.columns) + [f'{c}_floor' for c in fc]`。

逐配置拟合预测。`fit_predict(config, X_tr, y_tr, X_list, names) -> list[ndarray]`，对 `X_list` 中每个矩阵给出预测：
- `fit_predict_lin10`：只在前 `len(raw 列数)` 个化学列（不含 `_floor` 列）中，用 `scipy.stats.spearmanr` 计算每列与 `y_tr` 的相关，NaN 当作 0，取 `np.argsort(-np.abs(rho), kind='stable')[:10]`；预测变量为这 10 列，加上其基列在 `names` 中有对应 `_floor` 列的指示列；`LinearRegression()`。
- `fit_predict_enet`：`ElasticNet(alpha=config['alpha'], l1_ratio=config['l1_ratio'], max_iter=10000)`；在函数内用 `warnings.catch_warnings()` 只屏蔽 `ConvergenceWarning`，其余警告不动。
- `fit_predict_rf`：`RandomForestRegressor(n_estimators=..., max_features=..., min_samples_leaf=..., random_state=SEED, n_jobs=1)`。

通用外层折：
- `fit_outer_fold_generic(X_raw, Y, dims, dim_groups, groups, style, train_idx, test_idx, inner_seed, configs, prep, fit_predict) -> dict`：
  1. `tr = np.sort(train_idx)`，`te = np.sort(test_idx)`。
  2. 对 `inner_splits(tr, groups, style, inner_seed)` 的每个内层折 `(a, b)`：调用一次 `prep(X_raw.iloc[a], [X_raw.iloc[b]])`（每个内层折只准备一次，不要在配置或维度循环里重复）；`pos_b = np.searchsorted(tr, b)`。
  3. 对每个维度组：对每个配置、每个内层折、每个维度，拟合并预测 `b`，算 `rmse / sd`；配置得分为全部内层折和该组全部维度的平均；保存每个配置在各内层验证行上的预测，选出得分最小者（并列取靠前），直接取其预测作为 `inner`，不重新拟合。
  4. 外层：`prep(X_raw.iloc[tr], [X_raw.iloc[te]])`，对每个维度用所选配置在全部训练行上拟合并预测测试行。
  5. 返回 `{'train_idx': tr, 'test_idx': te, 'oof': (n_te, 50), 'inner': (n_train, 50), 'chosen': {组名: 配置字典}, 'scores': {组名: 得分列表}}`；列顺序与 `dims` 一致。
  - `Y` 的测试行只在评价时使用，函数内不得读取 `Y[te]`。
- `merge_generic(units, n=250, n_dims=50, n_rep=5, n_fold=5) -> tuple`：`units` 为 `{(r, k): 单元字典}`；返回 `oof`（n_rep, n, n_dims）、`inner`（n_rep, n_fold, n, n_dims，非训练行为 NaN）、`fold_id`（n_rep, n）。断言每次重复每款啤酒的 `oof` 恰好写入一次，且 `inner[r, k]` 的非 NaN 行恰为训练行。

chemstyle 的输入：`X_style = pd.concat([X.reset_index(drop=True), pd.get_dummies(style.reset_index(drop=True), prefix='style', dtype=float)[sorted_columns]], axis=1)`，风格列按风格名排序，22 列，列集合来自全部啤酒的风格标签（标签不是目标，可以用）。然后直接调用 W2 的 `fit_outer_fold(X_style, xbar, ...)`。风格列无 NaN，`col_mean` 编码对它们不起作用，单调约束对它们为 0。

B2（只与论文 Table S3 对照，不进入等级比较）：
- `b2_log_columns(X_tr) -> list[str]`：对训练集 `X_tr` 的每一列，取非 NaN 值；全部为正、至少 3 个值、且不是常数的列做 `scipy.stats.shapiro`，`p < 0.05` 的列入选。
- `b2_prepare(X_tr, X_te) -> tuple[ndarray, ndarray]`：入选列在两个表上都取 `np.log10`；训练集列均值（取对数后、忽略 NaN）填补两个表；`StandardScaler` 在训练集拟合。
- `b2_fit_dim(X_tr, y_tr, X_te, seed=SEED) -> tuple[ndarray, dict]`：`GridSearchCV(GradientBoostingRegressor(random_state=seed), B2_GRID, cv=KFold(5, shuffle=True, random_state=seed), scoring='r2', refit=True, n_jobs=1)`；返回测试预测和 `best_params_`。
- 目标为附件 4 的 `Z`（W1 的 `load_q1().Z`），划分为 `paper_split(style, seed=SEED)`。

## 任务 2：`code/scripts/solve_q1_baselines.py`（新写）

- 脚本最前面（`import numpy` 之前）设置三个线程环境变量为 1。
- 参数：`--smoke`（只跑重复 0 的折 0、1；`RF_CONFIGS` 的 `n_estimators` 改为 20；B2 只跑前 3 个维度；输出写入 `SMOKE_DIR`）、`--workers`（默认 8，不得超过 8）、`--only MODEL`（可重复；只计算这些模型的中间文件，不写最终 npz）。
- 流程：`require_sklearn()`、`ensure_dirs()`；读 `q1_data.npz`（`xbar`、`G`、`date_group`）、`q1_main_oof.npz`（`fold_id`）、`q1_scores.npz`（`grade_const`）；`load_q1()` 取 `X`、`Z`、`style`；`splits = outer_splits(date_group, style)`，并断言由它生成的每折测试行与 `fold_id` 完全一致。
- 单元为 `(模型, r, k)`。b0、b1 在主进程直接计算；lin10、enet、rf、chemstyle 用 `joblib.Parallel(n_jobs=workers)`；每个单元结果写入 `CHUNK_DIR/baselines/<模型>_r{r}_k{k}.npz`（键：`train_idx`、`test_idx`、`oof`、`inner`、`chosen`、`scores`，后两个为 JSON 字符串），已存在则读取。不带 `--only` 时，全部单元就绪后合并并写最终文件。
- 合并后对每个模型、每个 `(r, k, d)`：`apply_rules(oof[r, test, d], inner[r, k, train, d], xbar[train, d])`，把 `RULES` 中各规则的等级写入对应测试行，并把 `p` 写入。
- B2：50 个维度用 joblib 并行（`--workers`），写 `b2_test_idx`、`b2_pred`（75, 50）、`b2_r2`（50）、`b2_params`（JSON）、`b2_log_cols`（JSON，每维入选的对数列个数）。
- 输出 `PROCESSED_DIR/q1_baselines.npz`，键：
  - 对每个模型 `m`：`oof_<m>`（5, 250, 50）、`inner_<m>`（5, 5, 250, 50）、`chosen_<m>`（JSON，键 `r_k_组名`；b0、b1 为空对象）、`p_<m>`（5, 250, 50, 7）、`grade_<m>_<rule>`（5, 250, 50），`rule` 取 `RULES` 全部六项；
  - `b2_*`、`checks`（JSON 列表）、`seed`、`meta`。
- 任一 `checks` 项不通过时仍写出 npz，随后以非零退出码结束并在终端列出失败项。

## 任务 3：`code/tests/test_q1_baselines.py`

- `b0_mean`、`b1_style_shrunk`：`y_tr=[1, 3, 5, 7]`、`style_tr=['a', 'a', 'b', 'b']`、`style_te=['a', 'c']` 时，B1 为 `[24/7, 4.0]`（风格 a：n=2、均值 2、全局均值 4；风格 c 未见过，取全局均值）。
- `simple_fold`：真实数据重复 0 的折 0，`inner` 的非 NaN 行恰为训练行（b1 与 b0 各验一次）；把测试行的 `Y` 换成随机数，`oof` 与 `inner` 不变。
- `prep_linear`：人造表中，某列只有测试行取下限值，则输出 `names` 中没有该列的 `_floor`；改动测试行的值，`X_fit` 不变。
- `fit_predict_lin10`：人造 100×15 数据，`y = 2 * X[:, 3]` 加小噪声（种子 0），前 70 行训练、后 30 行预测，选入的特征含第 3 列，预测与真值的 $R^2$ 大于 0.9。
- 通用外层折无泄漏：真实数据重复 0 的折 0，`dim_groups={'Basic': ['alcohol', 'body']}`，`dims=['alcohol', 'body']`，`EN_CONFIGS` 取前 3 项：把 `Y` 的测试行换成 `np.random.default_rng(1).uniform(0, 6)` 后，`chosen`、`scores`、`inner`、`oof` 都不变；把训练行的 `Y` 加 1，`oof` 必须改变。
- RF：`n_estimators=20` 的配置在上述设定上运行，`oof`、`inner` 全部有限、形状正确。
- `b2_log_columns`：含非正值的列不入选；改动测试集的值，入选列不变。
- chemstyle 输入：风格列个数等于 `style.nunique()`（22），列名以 `style_` 开头且按风格名排序。

## 验收

命令（仓库根目录）：
```
PY=/e/anaconda3/python.exe
$PY code/scripts/solve_q1_baselines.py --smoke
$PY code/scripts/solve_q1_baselines.py --only rf --workers 8
$PY code/scripts/solve_q1_baselines.py --only enet --only lin10 --only chemstyle --workers 8
$PY code/scripts/solve_q1_baselines.py --workers 8
cd code && $PY -m pytest tests -q
```

运行时间：rf 约 2.1 万次、每次 1–2 秒量级，8 个进程约 1–1.5 小时，最耗时，先启动；chemstyle 与主模型相当，约 1–2 小时；enet 约 10 万次小拟合，约 10 分钟；B2 约 4,500 次，约 30 分钟。

`checks`（每项含 `name`、`observed`、`expected`、`tol`、`passed`；连续指标对 x̄，5 次重复分别计算后平均，$R^2$ 与 Spearman 逐维计算后取平均）：

| 名称 | 预期 | 容差 |
|---|---|---|
| 外层划分与 `q1_main_oof.npz` 的 `fold_id` 一致 | 是 | 精确 |
| 6 个模型的 `inner` 非 NaN 行恰为训练行 | 是 | 精确 |
| 6 个模型的 `oof` 全部有限 | 是 | 精确 |
| `grade_b0_const` 与 `q1_scores.npz` 的 `grade_const` 逐元素相同 | 是 | 精确 |
| B1 连续 MAE | 0.527 | ±0.02 |
| B1 $R^2$ | 0.053 | ±0.04 |
| B1 Spearman | 0.177 | ±0.04 |
| B0 $R^2$ | ≤ 0.001 | 上界 |
| B1 $R^2$ 大于 B0 $R^2$ | 是 | 严格 |
| chemstyle 的 `oof` 与主模型 `oof` 不完全相同 | 是 | 严格 |
| B2 测试集款数 | 75 | 精确 |
| B2 50 维平均测试 $R^2$（对 z） | 0.189 | ±0.03 |

B2 的差距超过 0.03 时，`passed=false` 并照实回报，不为逼近 0.189 而调参、不改网格。

回报中另列（不作为通过条件）：各模型的连续 MAE、$R^2$、Spearman；各模型在 `median` 规则下的等级 MAE（对 G）；enet 与 rf 各维度组被选中的配置个数；每个模型的 `wall_time_s`。

其余：Q1 范围 `black --check`、`ruff check` 通过；`code/src/q1/` 无 `__main__`。

## 禁止事项

- 不得用主模型的超参数代替 enet、rf 的调参；不得把 HistGBR 当作 enet 或 rf 输出。
- 不得在全部 250 款上拟合分位数变换、标准化、编码器或特征筛选后再划分。
- 不得只取第 0 次重复，也不得只用四舍五入离散。
- B2 的目标不得改用 x̄；B2 不进入任何等级比较。

## 处理结果
<!-- [PLACEHOLDER] | Await: coding 回复 ack 并按索引的回报格式逐项回报 -->
