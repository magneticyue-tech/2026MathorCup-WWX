---
id: 2026-10-28-q1-w2-main-model
from: modeling
to: coding
status: open
severity: blocker
refs:
  - review/2026-10-28-q1-index.md
  - review/2026-10-28-q1-w1-data.md
  - model/q1_methodology.md
  - experimentalCode/q1_raw_target/run_fold_1s.py
  - code/src/q1/splits.py
  - code/src/q1/model.py
  - code/src/q1/features.py
  - code/scripts/solve_q1_main.py
---

## 问题

- 实现方法论 §3.1、§3.3、§3.5、§5.1：按日期组分组、按风格分层的嵌套交叉验证，主模型为每个维度一个 HistGradientBoostingRegressor，目标为 x̄。
- 现有实现的错误：超参数只在 `splits[0][0]` 一个外层训练集上选一次，再用于全部 25 个外层折（`solve_q1_main.py:65-67`）；NA 编码在全部 250 款上拟合（`solve_q1_main.py:25-27`）；多加了缺失指示列（`solve_q1_main.py:28`）；`inner_mu` 只对所选配置保存，但没有选参过程的记录。
- 本工单依赖 W1 的 `q1_data.npz`（键 `xbar`、`date_group`、`reliability`）和 `ChemNAEncoder`（`transform` 只返回一个 DataFrame）。先读 `review/2026-10-28-q1-index.md` 的「通用约定」。
- 只创建或修改：`code/src/q1/splits.py`、`model.py`、`features.py`、`code/scripts/solve_q1_main.py`、`code/tests/test_q1_splits.py`、`code/tests/test_q1_model.py`。

## 嵌套交叉验证的固定规则（先读懂再写代码）

- 外层：5 次重复 × 5 折，共 25 个外层折，单元记为 (r, k)。每个外层折有训练集 `train_idx` 和测试集 `test_idx`（绝对行号，0..249）。
- 对每个外层折，在其训练集内再分 4 个内层折。内层折只用于选超参数和产生 `inner_mu`。
- 每个外层折、每个维度组各自选一次超参数。25 个外层折 × 6 个维度组 = 150 次选择，每次选择只使用该外层折的训练集。
- 任何拟合量（NA 列均值、超参数、模型）都只能用当前拟合所在折的训练行；测试行的 x̄ 不得出现在任何计算里。
- NA 编码器：每个内层折用内层训练行拟合一次，填充内层训练和内层验证行；外层预测时用外层训练行重新拟合一次，填充外层训练和测试行。

## 任务 1：`features.py`

保留 `MONOTONE_PAIRS` 和 `monotone_cst`。新增：
- `assert_pair_features(feature_names: list[str]) -> None`：6 个配对的化合物列都在 `feature_names` 中，缺任何一个即 `raise KeyError`。
- 不得加入 diacetyl 的配对（方法论 §3.3：已排除并披露）。

## 任务 2：`splits.py`（重写）

- `outer_splits(groups, style, n_splits=5, n_repeats=5, seed=SEED) -> list[list[tuple[ndarray, ndarray]]]`：第 r 次重复用 `StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)`，调用 `.split(np.zeros(n), style.to_numpy(), groups)`，返回每折的 `(train_idx, test_idx)`。最小的风格只有 3 款，少于折数时 sklearn 会发 UserWarning，在本函数内用 `warnings.catch_warnings()` 只屏蔽该警告，不要全局关闭警告。
- `inner_splits(train_idx, groups, style, seed, n_splits=4) -> list[tuple[ndarray, ndarray]]`：只在 `train_idx` 的行上，用 `StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)`，返回绝对行号的 `(inner_train_abs, inner_val_abs)`。绝对行号即属于 0..249 的行号，不是 `train_idx` 内的位置。
- `leave_one_style_splits(style) -> list[tuple[ndarray, ndarray, str]]`：每个元素为 `(train_idx, test_idx, style_name)`，按风格名字典序排列。
- `paper_split(style, seed=SEED) -> tuple[ndarray, ndarray]`：`train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=style)`。
- `split_diagnostics(splits, groups, style) -> dict`：对每次重复断言每款啤酒恰好在一个测试折出现一次、同一日期组不跨训练集和测试集，违反时 `raise AssertionError`；返回每折测试集大小列表、每折测试集日期组个数。

## 任务 3：`model.py`（重写）

常量：
- `ITERS = (100, 200, 400)`。
- `CONFIGS`：24 个字典，顺序固定：`for max_depth in (2, 3): for min_samples_leaf in (10, 20): for l2_regularization in (0.0, 1.0): for max_iter in ITERS:`；每个字典键为 `max_depth`、`min_samples_leaf`、`learning_rate`（固定 0.05）、`l2_regularization`、`max_iter`。
- `MONO_OPTIONS = (False, True)`。
- `GROUPS_WITH_MONO`：`DIM_GROUPS` 中含 `MONOTONE_PAIRS` 键的维度的组名集合（应为 Defect、Sour、Basic），由代码计算，不手写。只有这些组才展开单调约束 True；其余组 True 的得分记为 NaN。

函数：
- `make_hgb(params, cst, warm_start=False)`：`HistGradientBoostingRegressor(loss='squared_error', early_stopping=False, random_state=SEED, monotonic_cst=cst, warm_start=warm_start, **params)`；`params` 含上面 5 个键。`cst` 为 `None` 或整数列表。
- `fit_predict_path(X_tr, y_tr, X_pred_list, params_base, cst, iters=ITERS) -> list[list[ndarray]]`：`params_base` 为不含 `max_iter` 的字典。用一个 `warm_start=True` 的模型依次把 `max_iter` 设为 `iters` 中各值并 `fit`，每步对 `X_pred_list` 中每个矩阵预测一次。返回外层列表对应 `iters`，内层列表对应 `X_pred_list`。（`warm_start` 与分别拟合的结果一致，已在本机核对，最大差 0.0；仍须按下文测试在含单调约束的情形再核对一次。）
- `tune_group(inner_data, Y_cols, feature_names, dim_names, group, configs=CONFIGS) -> tuple[dict, ndarray, ndarray]`：
  - `inner_data`：列表，每个内层折一个元组 `(X_tr, X_va, tr_pos, va_pos)`，`X_tr`、`X_va` 为已编码的 float64 数组，`tr_pos`、`va_pos` 为这些行在外层训练集内的位置。
  - `Y_cols`：形状 `(n_train, n_dims_in_group)`，外层训练集该组各维度的 x̄。
  - 对每个 `mono` ∈ `MONO_OPTIONS`（非 `GROUPS_WITH_MONO` 的组跳过 True）、每个配置、每个内层折、每个维度：拟合后在内层验证行预测，计算 `rmse / sd`，`sd = np.std(y_va, ddof=0)`。配置得分 = 该配置在全部内层折和该组全部维度上的平均。同一 `(max_depth, min_samples_leaf, l2_regularization)` 的三个 `max_iter` 用 `fit_predict_path` 一次拟合得到。
  - `cst` 对每个维度由 `monotone_cst(feature_names, dim, mono)` 生成。
  - 选得分最小的配置；得分相同取排在前面的（先 `mono=False` 后 `True`，配置按 `CONFIGS` 顺序）。
  - 返回 `(best, scores, inner_pred)`：`best` 为该配置的参数字典外加键 `mono`（bool）；`scores` 形状 `(2, len(configs))`，不适用位置为 NaN；`inner_pred` 形状 `(n_train, n_dims_in_group)`，为所选配置在各内层验证行上的预测，位置按 `va_pos` 填入。
  - 实现提示：对每个内层折先把全部配置的验证预测存下来，选完后直接取所选配置的预测，不要为得到 `inner_pred` 重新拟合。
- `fit_outer_fold(X_raw, Y, dims, dim_groups, groups, style, train_idx, test_idx, inner_seed, configs=CONFIGS, scheme='col_mean') -> dict`：
  1. 对每个内层折 `(a, b)` ∈ `inner_splits(train_idx, groups, style, inner_seed)`：`ChemNAEncoder(scheme).fit(X_raw.iloc[a])`，编码 `X_raw.iloc[a]` 和 `X_raw.iloc[b]`；位置由 `np.searchsorted(train_idx_sorted, a)` 得到（`train_idx` 先排序）。
  2. 对每个维度组调用 `tune_group`。
  3. 用外层训练行拟合 `ChemNAEncoder(scheme)`，编码训练和测试行；用所选配置对该组每个维度拟合并预测测试行。
  4. 返回 `{'train_idx', 'test_idx', 'oof': (len(test), len(dims)), 'inner': (len(train), len(dims)), 'chosen': {group: best}, 'scores': {group: scores}}`；`oof`、`inner` 的列顺序与 `dims` 一致，行顺序与排序后的 `train_idx`、`test_idx` 一致。
- `NestedCVResult`：dataclass，字段 `oof_mu`（R, 250, 50）、`inner_mu`（R, K, 250, 50，外层训练行填内层折外预测，其余 NaN）、`fold_id`（R, 250，每款啤酒作为测试行所在的外层折号）、`chosen`（字典，键 `f'{r}_{k}_{group}'`）、`scores`（字典，同键）。
- `merge_units(units, shape) -> NestedCVResult`：`units` 为 `{(r, k): fit_outer_fold 的返回值}`，按上述含义拼接；断言每个 `(r, k)` 的 `inner` 填入的行数等于训练行数，且每次重复中每款啤酒的 `oof_mu` 被写入恰好一次。
- `nested_cv_predict(X_raw, Y, dims, dim_groups, groups, style, splits, repeats=None, folds=None, configs=CONFIGS, seed=SEED) -> NestedCVResult`：串行循环调用 `fit_outer_fold`（`inner_seed = seed + 100 + r`）再 `merge_units`；`repeats`、`folds` 为要运行的下标子集，默认全部。供测试和冒烟使用；正式运行由脚本并行。
- `refit_full(X_raw, Y, dims, dim_groups, groups, style, configs=CONFIGS, seed=SEED) -> dict`：全部 250 款上，`inner_splits(np.arange(n), groups, style, seed + 200, n_splits=5)` 选每个组的超参数（同一 `tune_group`）；编码器在 250 款上拟合；对每个维度用所选配置拟合最终模型。返回 `{'encoder', 'feature_names', 'chosen': {group: best}, 'models': {dim: model}}`。

## 任务 4：`code/scripts/solve_q1_main.py`（重写）

- 脚本最前面（`import numpy` 之前）设置 `os.environ['OMP_NUM_THREADS'] = '1'`，`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 同。
- 参数：`--smoke`（只跑重复 0 的折 0、1；输出写入 `SMOKE_DIR`）、`--workers`（默认 8，不得超过 8）。
- 流程：`require_sklearn()`、`ensure_dirs()`；读 `q1_data.npz` 取 `xbar`、`date_group`、`reliability`；`load_q1()` 取 `X`、`style`；`splits = outer_splits(date_group, style)`；生成 25 个单元；用 `joblib.Parallel(n_jobs=workers)` 并行，每个单元调用 `fit_outer_fold`，结果立即写入 `CHUNK_DIR/main/r{r}_k{k}.npz`（已存在则跳过，读取即可）；全部完成后 `merge_units`。
- 全数据重拟合：调用 `refit_full`，把 `encoder` 与每个维度的模型用 `joblib.dump` 写入 `PROCESSED_DIR/q1_models/`（`encoder.joblib`、`<dim>.joblib`）；`.gitignore` 已忽略 `*.joblib`。
- 写 `PROCESSED_DIR/q1_main_oof.npz`，键：`oof_mu`、`inner_mu`、`fold_id`、`chosen_params`（JSON：键 `r_k_group`）、`inner_scores`（JSON，同键）、`final_params`（JSON，`refit_full` 的 `chosen`）、`split_diag`（JSON）、`wall_time_s`、`checks`（JSON 列表）、`seed`、`meta`。
- 任一检查项不通过时仍写出 npz，随后以非零退出码结束并在终端列出失败项。

## 任务 5：测试

`code/tests/test_q1_splits.py`：
- 真实数据上 `outer_splits`：每次重复每款啤酒恰好出现在一个测试折；同一日期组不跨训练集和测试集；同一种子两次调用结果相同；不同重复的划分不完全相同。
- `inner_splits`：返回的行号全部属于 `train_idx`；4 个内层验证集互不相交且并集等于 `train_idx`；同一日期组不跨内层训练和验证。
- `paper_split`：测试集 75 款（250 的 30%），训练和测试不相交。

`code/tests/test_q1_model.py`：
- `assert_pair_features` 在真实数据的 231 列上通过；`monotone_cst` 对 `alcohol` 开启时只有乙醇列为 +1，对 `acidity` 开启时 pH 列为 −1，对 `diacetyl` 开启时全为 0，关闭时全为 0。
- `warm_start` 等价：真实数据 col_mean 编码后，对维度 `alcohol`、`mono=True`、`max_depth=3, min_samples_leaf=10, l2_regularization=1.0`，`fit_predict_path` 在 100、200、400 的预测与分别拟合的 `make_hgb` 预测在 `atol=1e-10` 内相等。
- 无泄漏：取真实数据、重复 0 的折 0；`dim_groups={'Basic': ['alcohol', 'body']}`、`dims=['alcohol', 'body']`、`configs` 只取 `CONFIGS` 中 `max_depth=2, min_samples_leaf=10, l2_regularization=0.0` 的 3 项；运行 `fit_outer_fold` 得结果 A。把 `Y` 中测试行的值换成 `np.random.default_rng(1).uniform(0, 6)` 得到 `Y2`，再运行得结果 B。断言 `chosen` 完全相同、`scores` 相等、`inner` 在所有位置相等、`oof` 相等。反向对照：把训练行的 `Y` 加 1 后，`oof` 必须改变。
- 合并：用上面的小设定对重复 0 的 2 个折运行 `nested_cv_predict`，断言 `inner_mu[0, k]` 非 NaN 的行恰为该折训练行，`inner_mu[0, k]` 在测试行为 NaN。

## 验收

命令（仓库根目录）：
```
PY=/e/anaconda3/python.exe
$PY code/scripts/solve_q1_main.py --smoke
$PY code/scripts/solve_q1_main.py --workers 8
cd code && $PY -m pytest tests -q
```

运行时间：每个外层折约 1,600–2,400 次拟合（单线程，每次 ≤ 1 秒量级）；25 个折用 8 个进程约 1–2 小时。`wall_time_s` 超过 4 小时时，在回报中写实测时长，不缩小网格。

`checks`（每项含 `name`、`observed`、`expected`、`tol`、`passed`，`passed` 由代码比较得出）：

| 名称 | 预期 | 容差 |
|---|---|---|
| 外层单元个数 | 25 | 精确 |
| 每次重复每款啤酒恰好测试一次 | 是 | 精确 |
| 同一日期组不跨外层折 | 是 | 精确 |
| `inner_mu[r,k]` 的非 NaN 行等于训练行 | 是 | 精确 |
| `inner_mu` 有限值个数 | 25 × 训练行数 × 50 之和，且无 inf | 精确 |
| 折外连续 MAE（对 x̄，5 次重复各自计算后平均） | 0.479 | ±0.02 |
| 折外 R²（逐维、逐次重复，再取平均） | 0.167 | ±0.04 |
| 折外 Spearman（逐维平均） | 0.368 | ±0.04 |
| 折外 R² 超过该维 `reliability` + 0.05 的维度数 | 0 | 精确 |

后三项以外的容差是预运行固定超参数结果的参考范围：预运行用固定超参数、`GroupKFold` 和 3 次重复，正式运行调参并用 `StratifiedGroupKFold`，二者有合理差别。观测值超出容差时，`passed=false` 并照实回报，不调整参数；超出 3 倍容差时暂停 W3 及以后。

回报中另列（不作为通过条件）：每个维度组在 25 个外层折上选出的不同配置个数、各配置被选中的次数、`mono=True` 被选中的次数、`wall_time_s`。若全部 150 次选择只出现 1 种配置，在回报中指出。

其余：Q1 范围 `black --check`、`ruff check` 通过；`code/src/q1/` 无 `__main__`；`git status` 中 `code/data/processed/q1/` 新增 `q1_main_oof.npz`，`q1_models/` 被忽略。

## 禁止事项

- 不得在任何一个外层折之外选超参数，不得跨外层折复用所选超参数。
- 不得加入缺失指示列；不得对化学表在全部 250 款上拟合编码器后再划分。
- 不得使用 `GridSearchCV`、`early_stopping`、缩小或改动网格。
- 不得在 `src/` 中写文件；写文件只在脚本中。

## 处理结果
<!-- [PLACEHOLDER] | Await: coding 回复 ack 并按索引的回报格式逐项回报 -->
