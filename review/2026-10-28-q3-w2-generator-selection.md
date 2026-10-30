---
id: 2026-10-28-q3-w2-generator-selection
from: modeling
to: coding
status: open
severity: major
refs:
  - review/2026-10-28-q3-index.md
  - review/2026-10-28-q3-w1-foundation.md
  - model/q3_methodology.md
  - experimentalCode/q3_methodology_checks.py
---

## 问题

Q3 缺少风格条件生成器、对照生成器和嵌套选参的实现。
本工单实现 `StyleGen`、G1–G3 抽样、外层与内层划分和嵌套选参，产出 `q3_select.npz`。
本工单依赖 W1 的 `marg.py`、`modules.py`、`data.py`，不依赖 Q1。

## 背景 / 复现

读方法论 §2.3、§3.1–§3.4、§6.1。参考实现：`Gen` 第 102–161 行，`folds` 第 164–168 行，检查 2 第 217–245 行，检查 3 第 267–280 行。
预检查在同一套留出划分上选参并报告。正式运行改为嵌套：每个外层训练集内用内层 5 折按留出潜空间高斯评分 L 选 k、κ 与结构，用选定设定在外层训练集上重拟合；外层留出啤酒只用于 W3 的报告。

## 文件与函数

### T1 `generator.py`

`StyleGen(kind=famod, k=15, kappa=2.0, global_only=False, tie_scheme=cond_mean, seed=SEED)`，`kind` 取 famod、fa、lw、diag；fa 与 famod 须给 k。

`fit(X_tr, s_tr, styles, module_labels=None) -> self`：`X_tr` 为 n×225，可含 NaN；`s_tr` 为风格数组；`styles` 为全部 22 个风格名；`module_labels` 为长度 225 的整数数组，kind 为 famod 且为 None 时，用 `modules_from_data` 在训练折的填充数据上计算。步骤：
1. `Marg(tie_scheme).fit(X_tr)`；`colmean = nanmean(X_tr, 0)`；`Xf` 为填充后的矩阵；`U = transform(Xf)`。
2. g 为 U 的列均值。`global_only`：所有风格的均值取 g，残差 R = U − g，dof = n − 1。否则：风格 s 有 n_s 款时 mu[s] = (n_s·mean_s + κ·g) / (n_s + κ)，n_s 为 0 时 mu[s] = g；残差 R = U − 各风格自身的普通均值（不收缩），dof = n − S，S 为训练集中出现的风格数。
3. R ← R·sqrt(n / dof)。
4. 协方差 C：diag 取 `diag(R.var(0))`（ddof 为 0）；lw 取 `LedoitWolf(assume_centered=True).fit(R).covariance_`；fa 取 `FactorAnalysis(n_components=k, random_state=seed).fit(R).get_covariance()`；famod：LL = comps.T @ comps，Res = RᵀR / n − LL，B = diag(clip(diag(Res), 1e-3))，对每个成员数 ≥ 2 的模块，把 B 中该模块的子块换成 Res 的对应子块，B 做特征分解并把特征值 clip 到 1e-3 后重组，C = LL + 重组结果。
5. L = cholesky(C + 1e-9·I)。
属性：`U`、`s`、`mu`（dict）、`C`、`L`、`mg`、`colmean`。

方法：
- `latent(X)`：先用 `colmean` 填充，再 `mg.transform`。
- `score(X, s) -> ndarray[n]`：L(x_b) = −0.5·‖L⁻¹(u_b − mu[s_b])‖² − Σ log diag(L) − 0.5·225·log(2π)。训练集中没有该风格时对应位置为 NaN，不以全局均值代替。
- `sample_latent(style, n, rng)`：`mu[style] + rng.standard_normal((n,225)) @ L.T`；训练集中没有该风格时抛 `ValueError`。
- `sample(style, n, rng)`：`mg.inverse(sample_latent(...))`。
- `log_density(u, style)`：高斯对数密度，供 W5b 使用。
- `with_kappa(kappa) -> StyleGen`：返回浅拷贝，共享 `C`、`L`、`U`、`mg`，只按步骤 2 重算 `mu`。

`fit_grid(X_tr, s_tr, styles, configs, module_labels=None) -> dict`：`configs` 为 (kind, k, kappa) 的列表。`Marg`、`U`、残差 R、模块只算一次；每个 (kind, k) 的 `FactorAnalysis` 只拟合一次，fa 与 famod 共用；κ 只改变 `mu`。返回以 config 为键的 `StyleGen`。`StyleGen.fit` 内部调用同一组辅助函数，两条路径结果一致。

### T2 `controls.py`

- `draw_g1(Xf_tr, s_tr, style, n, rng)`：在训练集同风格啤酒中有放回抽取 n 行，取均值填充后的原行。
- `draw_g2(Xf_tr, s_tr, style, n, rng)`：对每一列独立地在同风格训练啤酒中有放回抽取。
- `draw_g3(gen, s_tr, style, n, rng)`：在同风格训练啤酒中有放回抽取 a、b，λ = `rng.uniform(size=(n,1))`，返回 `gen.mg.inverse(λ·gen.U[a] + (1−λ)·gen.U[b])`。
- G0 为 `StyleGen(global_only=True)`，其余参数同主模型；纯因子为 `StyleGen(kind=fa)`，k、κ 同主模型。训练集中没有该风格时，三个抽样函数抛 `ValueError`。

### T3 `selection.py`

- `outer_splits(style, n_repeats=N_REPEATS, seed=SEED) -> list[tuple[int,int,ndarray,ndarray]]`：元素为 (rep, fold, train_idx, test_idx)，`StratifiedKFold(5, shuffle=True, random_state=seed+rep)` 按 `style` 分层。scikit-learn 对样本量小于 5 的风格给出警告，只过滤这一条（消息含 least populated class）。
- `inner_splits(train_idx, style, rep, seed=SEED)`：`StratifiedKFold(5, shuffle=True, random_state=seed+100+rep)` 作用于 `style[train_idx]`，返回全局索引。
- `CONFIGS`：fa、famod 各 7 个 k × 6 个 κ 共 84 个，加 diag、lw 各 6 个 κ 共 12 个，总计 96 个；`SELECTABLE` 为前 84 个。
- `inner_scores(X, style, train_idx, rep, module_labels=None) -> (scores[96], small_scores[96], n_excluded)`：对每个内层折用 `fit_grid` 拟合，对内层留出啤酒算 `score`；所有内层留出啤酒的 L 逐款平均，得到每个 config 的内层评分。训练集内没有其风格的留出啤酒不计入，返回被排除的啤酒数。`small_scores` 只用 `SMALL_STYLES` 四个风格（17 款）的内层留出啤酒平均，只报告，不参与选参。
- `select_config(scores) -> dict`：在 `SELECTABLE` 中取评分最大者；评分完全相同时取 k 较小者，再取 κ 较小者，再取 fa。这是实现约定，不改变方法。

### T4 `code/scripts/solve_q3_select.py`

写出 `q3_select.npz` 与 `q3_select.json`。
- 外层：25 个 (rep, fold) 单元，每个外层训练集调用 `inner_scores`（`module_labels=None`，模块在内层训练集内重算），用 `select_config` 选定；按 (rep, fold) 并行，每个单元一个中间文件。
- 全数据：用全部 250 款，`inner_scores(rep=0)`，`select_config` 得 `final_config`。内层拟合同样重算模块；交付的全数据生成器在 W4、W5 中用 `modules_from_q2` 重拟合。
- npz 字段：`outer_rep`、`outer_fold`（各 25）；`chosen_kind`、`chosen_k`、`chosen_kappa`（各 25）；`configs`（96 个，JSON 字符串）；`inner_L`（25×96）；`inner_small_L`（25×96）；`inner_excluded`（25）；`final_config`（JSON）；`final_inner_L`（96）；`final_small_L`（96）；`seed`；`meta`。
- `--preset prerun`：不嵌套，`StratifiedKFold(5, shuffle=True, random_state=SEED+rep)`，rep 取 0、1 共 10 折，对下列 5 个设定计算留出 L 的平均并打印，写入 `_smoke/q3_select_prerun.json`；只用于回归，正式运行不使用。

## 验收标准

通用：`black --check code/` 与 `ruff check code/` 通过；`src/q3/` 无 `__main__`，导入时不读写文件；脚本写 `seed` 与 `meta`。

pytest（`code/tests/test_q3_generator.py`，用 60 款、30 列的合成数据，秒级）：
- 同种子同输入，`sample` 两次结果相同。
- `C` 对称，最小特征值大于 0；`L @ L.T` 与 `C + 1e-9·I` 相差不超过 1e-8。
- `with_kappa(κ)` 与以该 κ 重新 `fit` 的 `mu`、`score` 一致（1e-9）。
- `fit_grid` 与逐个 `StyleGen.fit` 在 3 个 config 上的 `C`、`score` 一致（1e-9）。
- `sample` 每列取值落在训练观测的 [min, max] 内。
- 风格不在训练集时，`score` 对应位置为 NaN，`sample_latent` 抛 `ValueError`。
- `outer_splits` 共 25 个元素；每个重复内每款啤酒恰好出现在一个测试折。

数值核对（`--preset prerun`，10 折平均留出 L，单位 nat/款，容差 ±1.0）：
- famod，k=15，κ=2：−240.7。
- fa，k=15，κ=2：−244.9。
- lw，κ=2：−267.1。
- diag，κ=2：−297.3。
- global_only 的 fa，k=15：−241.9。
- famod 超出容差时，先核对：模块是否在训练折内由填充数据的 Spearman 重算；残差是否按 sqrt(n / dof) 缩放；`FactorAnalysis` 的 `random_state`。

划分核对（写入 `q3_select.json` 的 `checks` 字段）：
- 外层训练集含 Faro 2–3 款、Brut 3–4 款（方法论 §3.3）。
- 125 个内层训练集都含至少 1 款 Faro、至少 2 款 Brut；含恰好 1 款 Faro 的内层训练集数，方法论记为 30，写入回报，不作停止条件。

正式运行：
- `q3_select.npz` 各字段形状如上；25 个外层折全部有选定设定，逐折打印。
- `inner_excluded` 全为 0（分层划分下预期如此）；非 0 时在回报中列出折号与风格。
- 回报中给出：选定设定的频数表、`final_config`、外层折选定的 k 与 κ 的范围、小样本风格内层 L 随 κ 的曲线。
- 耗时估计：外层 25 折 × 内层 5 折，每个训练集约 10 次协方差拟合（7 个 k 的 FactorAnalysis、lw、diag），8 个进程约 0.5–1.5 小时；以实测回报。

## 处理结果
<!-- [PLACEHOLDER] | Await: coding 回复 ack 并逐项回报 -->
