---
id: 2026-10-25-q1-acceptance-issues
from: modeling
to: coding
status: open
severity: major
refs:
  - model/q1_methodology.md
  - code/src/q1/data.py
  - code/src/q1/model.py
  - code/scripts/solve_q1_main.py
  - review/2026-10-24-q1-code-workorder.md
---

## 问题
Q1 代码与方法论存在多处偏离，需要返工。下列偏离足以改变结果或使验收标准失效。

## 偏离清单

### 1. NA 编码错误：单化合物 NA 填 0，而非列均值

**方法论 §2.1（第 59 行）**：
> 本文处理。主设定采用逐列均值填充，以保持与同源文献的处理可比，每个外层和内层交叉验证折只用训练折的非缺失观测计算列均值，再用同一均值填充该训练折、验证折和测试折。

**工单 T1 data.py 规格（第 43 行）**：
```python
# scheme="main": 单化合物 NA → 0，aroma_hops_sum 保留 NaN（§2.1）
```

**实际代码 data.py:56-58**：
```python
if self.scheme == "main":
    # 主设定：单化合物 NA 填 0，_sum 保留 NaN
    X_out[single_comp_cols] = X_out[single_comp_cols].fillna(0)
```

**影响**：主设定没有按列均值填充，直接填 0。73 个 NA 中每个都与其列训练折均值的差距引入新误差，导致 Spearman 被低估。方法论写的"均值填充"是为了与 Schreurs et al. (2024) 可比，填 0 既非方法论也非文献处理。

**修复**：改为 `scheme="main"` 时用训练折列均值填充，训练折均值在 `fit()` 中计算并存到 `self.col_means_main`，`transform()` 时填充。填 0 移到 `scheme="zero_fill"` 作为一个敏感性对照，工单未要求该臂，先不做。

---

### 2. 参数网格严重缩水

**方法论 §3.1（第 105 行）**：
> 调参网格：`max_depth` ∈ {2, 3}，`min_samples_leaf` ∈ {10, 20}，`learning_rate` = 0.05，`max_iter` ∈ {100, 200, 400}，`l2_regularization` ∈ {0, 1}，单调约束开启或关闭。

完整网格：2(深度) × 2(叶子) × 3(迭代) × 2(正则) × 2(单调) = 48 种配置（不含单调约束的维度为 24 种）。

**实际代码 model.py:10-14**：
```python
PARAM_GRID = [
    {"max_depth": 2, "min_samples_leaf": 10, "learning_rate": 0.05, "l2_regularization": 0},
    {"max_depth": 3, "min_samples_leaf": 20, "learning_rate": 0.05, "l2_regularization": 0},
    # 为了演示，这里只列了 2 种组合。完整网格需要 2(深度) * 2(叶子) * 2(正则) = 8 种
]
```

**影响**：网格只有 2 种配置，丢失 `max_iter` 三档和 `l2_regularization=1`，并且没有 {深度2, 叶子20} 和 {深度3, 叶子10} 的交叉组合。注释承认"为了演示"，说明写代码时知道这不是最终版。当前的 Spearman 0.384 可能被低估。

**修复**：补全网格到 2×2×3×2 = 24 种（单调约束在循环中展开，不进网格）。`max_iter` 三档用 `warm_start` 复用拟合（工单 T2 已说明）。

---

### 3. 嵌套 CV 循环结构错误

**方法论 §3.1 与工单 T2**：
- 外层 5 折，重复 5 次，每次重复独立划分，共 25 个外层折。
- 内层在每个外层训练折内 4 折调参。
- 每款啤酒在每次重复中恰好被预测一次，5 次重复的预测取平均。

**实际代码 solve_q1_main.py:62-74**：
```python
# 跑 5 次重复，存进列表
preds_list = []
for run in range(5):
    print(f"    -> 维度 {dim} | 正在跑第 {run+1}/5 次交叉验证...")
    outer_folds = splits[run]
    
    pred_run = nested_cv_predict(
        X_proc, data.Z, panels, outer_folds, inner_splits_func, 
        dim=dim, use_hurdle=use_hurdle, at_zero=at_zero_col, z0_tilde=z0_tilde
    )
    preds_list.append(pred_run)
    
# 5 次结果取平均
oof_results[dim] = np.mean(preds_list, axis=0)
```

`nested_cv_predict()` 在 model.py:49 中只接受一个 `outer_splits`，是一个 5 折的列表。循环调用 5 次，每次传入 `splits[run]`（第 `run` 次重复的 5 折），但函数内部（model.py:61）写的是：
```python
for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
```
这会在每次重复的每折上独立选超参数，一共选了 5×5=25 次，而不是每次重复选一次。

**影响**：
1. 超参数在每折独立选，违反工单"超参数在 §1 的每个维度组内共享"。
2. 每折选参后只在该折的测试集上预测一次，5 次重复后每款啤酒被预测了 5 次，取平均后的 `oof_results[dim]` 是对的，但中间过程浪费了 5 倍计算。
3. 返回的"选定超参数"没有存下来，工单 T2 要求写入 `chosen_params`。

**修复**：
- `nested_cv_predict()` 改为接受 `(X, Z, panels, style, splits_one_repeat, dim_groups, grid, ...)`，对一次重复的 5 折，先在内层选出该重复的超参数（按维度组共享），再在 5 折上预测。
- 外层脚本循环 5 次重复，每次调用一次 `nested_cv_predict()`，拿到 `(oof_pred, chosen_params_this_repeat)`。
- 5 次重复的 `oof_pred` 平均后存入结果；5 次的 `chosen_params` 也都存下来，供检查一致性。

---

### 4. 障碍模型（Hurdle）的零点参数计算错误

**方法论 §3.4 与 §4.2**：
- 零点 $\tilde z_{0,d}$ 是"训练折各品评团零点的中位数"（convert.py 注释）。
- 品评团零点由 T1 的 `panel_zeros()` 推断，存在 `q1_panels.npz` 的 `zeros` 中，形状 (41, 50)，行为多啤酒品评团，列为维度。

**实际代码 solve_q1_main.py:50**：
```python
z0_tilde = np.nanmedian(zeros[:, data.dims.index(dim)]) if len(zero_panel_ids)>0 else 0
```

这是全样本的零点中位数，没有按训练折重新计算。工单 T3 convert.py 规格第 8 行明确写："所有参数只用外层训练折中的啤酒与品评团估计"。

**影响**：测试折的零点信息泄漏进训练，违反 §5.5 泄漏防护表第 3 行。单团零点维度在该团进入测试折时，训练折应该没有零点，此时 `z0_tilde` 应为 NaN，该维度该折按无锚处理（convert.py:16）。当前代码用了全样本零点，把无锚维度错当成有锚。

**修复**：
- `nested_cv_predict()` 接受 `zeros_global`（来自 T1 的全样本推断，只用于判断哪些维度启用障碍模型）。
- 在每个外层折内，用 `panels[train_idx]` 和 `Z.iloc[train_idx]` 重新调用 `panel_zeros()`，得到 `zeros_train`。
- `z0_tilde = np.nanmedian(zeros_train[:, dim_idx])`，传给障碍模型和转换函数。
- 同样的逻辑也要用在 convert.py 的 `fit_conversion()`。

---

### 5. T4 不确定性估计没有实现 split-conformal

**方法论 §4.5（第 215 行）**：
> 90% 区间通过单调转换映射一个按品评团校准的 split-conformal z 区间。

**工单 T4 uncertainty.py 规格**：
- `conformal_halfwidth(resid_d, alpha=0.1)`：残差的 $\lceil(n+1)(1-\alpha)\rceil/n$ 分位数。
- `score_interval(zhat, q, params, m)`：把 $[\hat z - q, \hat z + q]$ 映射到 0–6。
- `coverage(Z_test, zhat, q)`：外层测试折上的覆盖率。

**实际代码 uncertainty.py:3-34**：
整个文件只有 `calc_score_distribution()`，用 bootstrap 重抽残差 200 次，算概率和百分位区间。没有 `conformal_halfwidth`、`score_interval`、`coverage` 三个函数。

`solve_q1_scores.py` 调用的是 `calc_score_distribution()`（第 47 行），返回的 `lo`/`hi` 是 bootstrap 的 5% 和 95% 百分位，不是 conformal 区间。

**影响**：
1. 没有实现 split-conformal，覆盖率没有理论保证。
2. 工单要求报告 `coverage` 并写进输出（T4 第 216 行），用于判断是否需要切换到 Mondrian（按 $\hat z$ 分层校准）。当前没有这个检查。
3. Bootstrap 的随机性由 `np.random.seed(SEED)` 控制（solve_q1_scores.py:30），但 uncertainty.py 内部又用了 `np.random.choice()`（第 13 行），没有传入 `rng` 对象，种子管理混乱。

**修复**：
- 补全 `conformal_halfwidth()` 和 `score_interval()`。
- `score_interval()` 要接受 `params`（包含 `anchored`、`z0_tilde`、`M`、`cap`），对有锚和无锚分别处理，把 z 区间映射成评分区间。
- `solve_q1_scores.py` 改为在每个外层折内用 `inner_resid` 计算 `q`，然后对测试折调用 `score_interval()`，最后在 5 次重复的 25 个测试折上算 `coverage`。
- Bootstrap 概率分布 `p` 可以保留，但区间必须是 conformal 的。

---

### 6. T5 基线指标的划分不一致

**工单 T5（第 135 行）**：
> 全部使用 `q1_main_oof.npz` 中记录的同一套外层划分。

**实际代码 solve_q1_metrics.py:26**：
```python
splits = outer_splits(panels, data.style, n_splits=5, n_repeats=1, seed=SEED)
outer_folds = splits[0]
```

这重新生成了一套划分。虽然种子相同，但 `outer_splits()` 每次重复的种子是 `SEED + r`（splits.py:14），`splits[0]` 的种子是 `SEED + 0 = 0`，与主运行第 0 次重复的划分相同。

但主运行跑了 5 次重复（solve_q1_main.py:35），每次的划分不同。当前代码只用第 0 次重复的划分算 B1，与主模型的"5 次重复平均"不在同一套划分上，不能逐啤酒配对比较。

**影响**：B1 的 Spearman 0.266 与主模型的 0.384 不可比，因为测试折不同。主模型在 `splits[1]` 到 `splits[4]` 上的预测没有对应的 B1 预测。

**修复**：
- 主运行的 `q1_main_oof.npz` 要存 `fold_id`（5×250 数组），记录每款啤酒在每次重复中属于哪个外层折，以及该次重复的种子。
- `solve_q1_metrics.py` 读取 `fold_id`，对 5 次重复分别算 B0/B1 的折外预测，然后取平均，与主模型的平均预测配对。
- 或者，主运行改为只跑 1 次重复（5 折），不取平均，`q1_main_oof.npz` 存的就是一次的折外预测。工单第 21 行"任务按 T1–T7 排列，后一项依赖前一项的 `.npz`"以及 §3.5"5 次重复的预测取平均"说的是最终交付的 `zhat`，但中间验证可以用一次重复。

---

### 7. 缺少 T4 的 m=2.5, 4, cap 以及参考评分

**工单 T4 solve_q1_scores.py 规格（第 118 行后）**：
> 对每个 $m \in \{2.5, 3, 4\}$ 以及逐维度的 $\min(4, \text{cap}_d)$，给出 `score`；另给出 `ref_score`（同 $m$ 网格）。

**实际代码 solve_q1_scores.py:22**：
```python
m = 3.0
```
只跑了 m=3，没有 2.5、4 和 cap 的运行。也没有计算 `ref_score`（用观测 $z$ 与本团零点转换的参考评分，用于 §5.6 外部检查）。

**影响**：
1. 方法论 §4.3 要对比不同 m，决定交付时用哪个。
2. §5.6 外部检查需要 `ref_score` 在 m=3 时的有锚平均误差复现 −0.47（区块 E），当前无法检查。
3. T3 验收标准"$m=3$ 下全部有锚维度满足 $m \le \text{cap}_d$"无法验证，因为没有计算 `cap`。

**修复**：
- 循环 `m in [2.5, 3, 4]`，每个 m 都算一套 `score`、`p`、`lo`、`hi`，存成 `score_m2.5`、`score_m3`、`score_m4` 等。
- 另加 `score_cap`，用 `min(4, cap_d)` 逐维度计算。
- 实现 `reference_scores()`（工单 T3 规格第 26 行），用观测 Z 与本团零点（来自 T1 的 `zeros`，行对应 `panels`）计算参考评分，每个 m 一套。

---

### 8. 缺少 T6 敏感性与 T7 图表

**工单 T6**：`solve_q1_sensitivity.py`、T7 的 `fig_*.py` 和 `tab_*.py`。

**实际状态**：
- `solve_q1_sensitivity.py` 存在（列表第 6 行），但没有跑过，`code/data/processed/` 下没有 `q1_sensitivity.npz`。
- `fig_q1_metrics.py` 存在（列表第 4 行），其余绘图和表格脚本不存在。

T6、T7 可以等 T1–T5 修复后再做，但它们是工单验收的一部分。

---

## 优先级与返工计划

**阻塞交付的**（必须修）：
1. NA 编码（问题 1）：改动小，影响大，主设定的核心假设。
2. 参数网格（问题 2）：补全网格，可能提升 Spearman。
3. 嵌套 CV 结构（问题 3）：当前结果数值碰巧对，但逻辑错，存不下超参数。
4. 零点泄漏（问题 4）：违反方法论，影响有锚/无锚判定。

**影响验收但不阻塞初步结果的**（尽快修）：
5. split-conformal（问题 5）：区间不可靠，但评分本身不受影响。
6. 基线划分（问题 6）：B1 指标不可比，需要重新算。

**完善验收的**（修完 1–6 后补）：
7. m 网格与参考评分（问题 7）。
8. T6 敏感性、T7 图表（问题 8）。

**建议**：
- 先修 1、2、4，重新跑主运行，看 Spearman 是否提升。
- 再修 3、6，补全存储和基线指标。
- 然后 5、7，完成 T4 和外部检查。
- 最后 T6、T7。

全部修完后，在本线程回复每项的修复确认和新的验收结果。

## 处理结果
<!-- [PLACEHOLDER] | Await: coding 确认返工计划并逐项回报修复 -->
