---
id: 2026-10-28-q3-w5a-chem-transfer
from: modeling
to: coding
status: open
severity: major
refs:
  - review/2026-10-28-q3-index.md
  - review/2026-10-28-q3-w1-foundation.md
  - review/2026-10-28-q3-w2-generator-selection.md
  - model/q3_methodology.md
  - experimentalCode/q3_methodology_checks.py
  - code/data/processed/q2/graph_nodes.csv
---

## 问题

Q3 缺少缺陷约束、源表示、化学层投影和三个消融对照（T0、T1、T3）的实现。
本工单只做化学层，不含感官层，不依赖 Q1。
本工单依赖 W1 的 `data.py`、`marg.py`、`modules.py`，正式运行依赖 W2 的 `q3_select.json`。

## 背景 / 复现

读方法论 §2.3、§5.1–§5.4、§5.6 中 T0、T1、T3 的定义。参考实现为预检查 7 的第 422–565 行，其中 `chem_bounds`、`project`、原尺度平移与 462 个风格对统计可直接对照。
预检查的化学上界与迁移起点取均值填充后的 x_b；现行定义取源表示 x° = Marg.inverse(u_b)。两者只在 9 个双乙酰 NA 单元格上不同（方法论 §6.8）。
`--config prerun` 取 famod、k=15、κ=2，模块取 Q2 表，供开发与冒烟使用；`--config final` 读 `code/data/processed/q3/q3_select.json` 的 `final_config`，文件不存在时报错，不回退到预运行设定。两种配置都用全部 250 款拟合，模块用 `modules_from_q2`。

## 文件与函数

### T1 `marg.py` 追加

`Marg.inverse_column(j, v) -> ndarray`：只对第 j 列的潜变量向量 v 做逆变换，与 `inverse` 同一插值，供约束上界求解使用。补 pytest：对随机 U，`inverse_column(j, U[:, j])` 与 `inverse(U)[:, j]` 逐元素相等。

### T2 `defects.py`

- `DefectSpec`（dataclass）：`cols`（9 个字段名，顺序取 `config.DEFECT_CHEM`）、`idx`（各列在 225 个生成列中的位置，断言全部存在）、`theta`（9 个阈值，换到该列的尺度，无阈值为 NaN，调用 `data.threshold`）、`has_threshold`（9 个 bool）。
- `build_defect_spec(gen_cols, dic, kind=threshold_mean) -> DefectSpec`。
- `chem_bounds(x_src_row, spec, rule=threshold) -> ndarray[9]`：rule 为 threshold 时，有阈值者 b_j = max(x_j, θ_j)，无阈值者 b_j = x_j；rule 为 strict 时一律 b_j = x_j。`x_src_row` 为源表示的 225 列。
- `LatentBounder(mg, spec)`：构造时对 9 个缺陷列各算一次 `GRID = np.linspace(-6, 6, 4801)` 上的 `mg.inverse_column(j, GRID)`；`beta(b, u_row) -> ndarray[9]`：β_j 取网格中满足逆变换值 ≤ b_j + 1e-12 的最大网格点，再取 `max(β_j, u_row[idx_j])`。后一步是因为源潜变量本身可行（Q(Φ(u_bj)) = x°_bj ≤ b_j），而网格点可能略小于它。网格中没有满足条件的点时抛 `ValueError`。
- `violation(x_new, b, spec, tol=1e-9) -> ndarray[9] bool`：x_new[idx] > b + tol。

### T3 `transfer.py`

- `translate(u, mu_s, mu_t)`：返回 u + (mu_t − mu_s)。
- `project(u_tilde, beta, C, idx, max_iter=50) -> (u_star, active, lam)`：求 min (u − ũ)ᵀ C⁻¹ (u − ũ)，约束 u[idx_k] ≤ β_k。有效集法：A 取 ũ[idx_k] > β_k 的 k；每轮若 A 非空，lam = solve(C[ix_(ja, ja)], ũ[ja] − β[A])，u = ũ − C[:, ja] @ lam，ja 为 A 对应的坐标；drop 为 A 中 lam < −1e-10 的 k，add 为不在 A 中且 u[idx_k] > β_k + 1e-9 的 k；drop 与 add 均为空时返回，否则 A ← (A − drop) + add。迭代 max_iter 轮仍未收敛时抛 `RuntimeError`，不静默返回。`active` 为 9 维 bool，`lam` 为 9 维，非有效约束处为 0。参考实现为第 449–463 行，区别是不收敛时报错。
- `truncate(u_tilde, beta, idx)`：对 idx 处取 min(ũ, β)，其余不动（对照 T1 的终点）。
- `shift_raw(x_filled_src, mean_src, mean_tgt)`：原尺度均值平移（对照 T3），`mean_*` 为各风格非缺失观测的列均值 `np.nanmean`。

### T4 `code/scripts/solve_q3_transfer_chem.py`

命令行：`--config prerun|final`、`--rule threshold|strict`（默认 threshold）、`--thr-kind mean|median`（默认 mean）、`--smoke`。输出 `q3_transfer_chem{suffix}.npz`，主设定后缀为空，strict 为 `_strict`，median 为 `_median`，两者同时改动为 `_strict_median`。`--smoke` 只取前 2 个风格作源、前 3 个作目标，写入 `_smoke/`。
步骤：
1. 拟合全数据生成器（按 `--config`），u_full、x_src 取自 `StyleGen.U` 与 `mg.inverse(U)`；`build_defect_spec`；对 250 款算 `b`（250×9）与 `beta`（250×9）。
2. 对 22 个源风格 × 21 个目标风格 × 源风格的每款啤酒：ũ = translate；`x_t0 = mg.inverse(ũ)`，T0 违反 = `violation(x_t0, b)`；`project` 得 u*、active、lam；T1 终点 = truncate。
3. Pils/Lager → Tripel 的 11 款另存全部中间量：ũ、u*、u_trunc、x_t0、x_proj、x_trunc（225 列）、b、beta、active、lam、T0 违反掩码。
4. T3：对 11 款算 `shift_raw`，统计单元格总数、超出 250 款观测范围的个数、有下限列中低于下限的个数、线性列（字典 `transform_interpretation` 以 线性 开头）中为负的个数。
5. 风格对汇总：`pair_src`、`pair_tgt`、`pair_n_src`、`pair_viol_frac`（T0 至少违反一项化学约束的比例）、`pair_mean_active`（投影的平均有效约束数）、`pair_n_src_missing_defect`（源风格中含源缺失缺陷单元格的啤酒数，由 `na_mask_gen` 在 9 个缺陷列上取得）。
6. npz 另存：`spec_cols`、`theta`、`b`、`beta`、`active`（250×22×9 bool，对角为 False）、`viol_t0`（同形状）、`u_star`（250×22×225，对角为 u_full 本身）、`src_missing_defect`（250×9 bool）、`config`（JSON）、`seed`、`meta`。

## 验收标准

通用：`black --check code/` 与 `ruff check code/` 通过；`src/q3/` 无 `__main__`，导入时不读写文件；脚本写 `seed` 与 `meta`。

pytest（`code/tests/test_q3_transfer.py`、`test_q3_defects.py`，合成数据，秒级）：
- `project`：u* 满足全部约束（容差 1e-9）；有效约束处 u*_A = β_A；lam ≥ 0；与等式条件均值公式 ũ − C[:, A] C_AA⁻¹ (ũ_A − β_A) 一致（1e-9）。
- `project`：对 6 维随机正定矩阵与 2 个约束的小问题，结果与 `scipy.optimize.minimize(method=SLSQP)` 的解相差不超过 1e-5。
- `project`：ũ 已满足全部约束时返回 ũ 本身，有效约束数为 0。
- `project`：构造不收敛的情形（max_iter=1 且需要两轮）时抛 `RuntimeError`。
- 零迁移：对 250 款，`LatentBounder.beta(b, u) >= u[idx]`；`violation(mg.inverse(u_full), b)` 全为 False，即 α=0 精确可行；非缺失单元格的 x° 与 x_filled 相差不超过 1e-9。
- `chem_bounds`：strict 下 b 等于 x；threshold 下有阈值列 b ≥ x，无阈值列 b = x。

数值核对（正式运行，写入 `q3_transfer_chem.json` 的 `checks`，容差见括号；预检查的 462 对统计只在 9 个双乙酰 NA 单元格上与现行上界不同，故容差较小）：
- Pils/Lager → Tripel 共 11 款；T0 平移后违反化学约束的啤酒数：H2S、MeSH、X.E..2.nonenal 各 11；DMS 10；DES 2（±1）；X2.3.butanedione 3（±2，该列与源缺失有关，写入回报）。
- 11 款投影的有效约束数为 3–5（±1），中位 4（±1）。
- T3 原尺度平移：2,475 个单元格中超出观测范围 163、低于下限 102、线性列负值 1（均要求精确相等）。
- 462 个有序风格对：`pair_viol_frac` 的中位数 1.00（±0.02），范围 0 至 1.00（±0.02）；`pair_mean_active` 的中位数 2.0（±0.3）。
- 源缺失：`src_missing_defect` 的 True 个数为 9，对应 X2.3.butanedione，其中 Pils/Lager 的 2 款为 W1 记录的 `beer` 值。
- 超出容差时按通用约定 9 停止，在回报中给出观测值，不改方法。

耗时估计：462 个风格对 × 每款一次投影，投影只涉及 9 个坐标，预计数分钟；以实测回报。

## 处理结果
<!-- [PLACEHOLDER] | Await: coding 回复 ack 并逐项回报 -->
