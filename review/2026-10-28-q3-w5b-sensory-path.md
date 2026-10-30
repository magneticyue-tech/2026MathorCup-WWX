---
id: 2026-10-28-q3-w5b-sensory-path
from: modeling
to: coding
status: open
severity: major
refs:
  - review/2026-10-28-q3-index.md
  - review/2026-10-28-q3-w4-portrait.md
  - review/2026-10-28-q3-w5a-chem-transfer.md
  - model/q3_methodology.md
  - experimentalCode/q3_methodology_checks.py
  - code/data/processed/q2/final_top3_with_rationality.csv
  - code/data/processed/q2/full_controls_results.csv
---

## 前置条件（未满足时不得开始）

1. W4 的前置条件全部满足，且 W4 的 `q1_bridge.py` 通过验收。
2. W5a 已 resolved，`q3_transfer_chem.npz` 存在。

## 问题

Q3 缺少感官层路径搜索、α*、α_fit、同约束对照（T4、T5、T2）、迁移指标、结论类别和逐款报告（方法论 §5.2、§5.5–§5.7、§6.4）的实现。
本工单产出 `q3_transfer.npz`、逐款与风格对 CSV。

## 背景 / 复现

读方法论 §5.2、§5.3 末段、§5.5–§5.7、§6.4。感官主约束比较 Q1 预测分布中位等级；`mu_cont` 敏感性规则直接比较 0–6 分连续预测 μ̂，不使用附件 4 的 z 线性换算。参考实现为预检查 7，第 466–540 行（感官层）与第 488–496 行（判别器与马氏距离）；预检查的感官基准用重构输入，不得沿用。
本工单用 W5a 的 `defects.py`、`transfer.py`，用 W4 的 `q1_bridge.py`。全数据生成器配置与 W5a 的 `--config final` 相同。

## 文件与函数

### T1 `source_input.py`

- `SourceRepr`（dataclass）：`u`（225）、`x_src`（225）、`comp_obs_latent`（5，观测综合指标的潜变量 v_bc，NA 先按列均值填充）、`comp_src`（5，`comp.to_original(comp_obs_latent)`，非缺失为观测值，缺失为占位值的往返值）、`internal_std`（观测值，无 NA）。
- `build_source(gen, comp, X_chem, b) -> SourceRepr`。
- `source_input231(sr) -> ndarray[231]`：225 列取 `x_src`，综合指标取 `comp_src`，内标取 `internal_std`。
- `transfer_input231(gen, comp, sr, u_new) -> ndarray[n,231]`：225 列取 `gen.mg.inverse(u_new)`；综合指标的潜变量取 `comp.predict_latent(u_new) + (sr.comp_obs_latent − comp.predict_latent(sr.u))`，再 `to_original`；内标取 `sr.internal_std`。`u_new = sr.u` 时必须得到 `source_input231(sr)`（1e-9）。
- 重构误差另报：`resid_comp = comp_obs_latent − predict_latent(u)` 对 250 款的分布；Pils/Lager 的 11 款只用岭回归预测（不保留残差）时感官预测相对基准的变化。

### T2 `sens_search.py`

- `DEFECT_SENS_IDX`：6 个感官缺陷维度在 `DIMS` 中的位置，断言 dms、metallic、X4vg、t2n、diacetyl、stale_hops 都存在；barnyard 不设约束。
- `sens_baseline(bundle, sr, rule) -> ndarray[6]`：源输入经 `predict_mu`；rule 为 grade（主设定）取 `grade_median`，mu_cont 取连续预测，grade_quant 取 `grade_quant`，none 返回 None。
- `sens_feasible(vals[n,6], base[6], rule) -> bool[n]`：grade 与 grade_quant 为 vals ≤ base；mu_cont 为 vals ≤ base + 1e-9；none 全 True。
- `path_search(gen, comp, bundle, sr, b, spec, u_end, rule, thr_tol=1e-9) -> dict`：对 α 网格 u(α) = u_b + α(u_end − u_b)：
  - 组装 231 列输入、预测、感官可行性；
  - 原尺度复查：`x_new = gen.mg.inverse(u(α))`，`violation(x_new, b, spec)` 全 False 才可行；
  - 返回 `ok_sens`、`ok_chem_orig`、`feasible`（两者与）、`alpha_star`（最大可行网格点）、`first_infeasible`（沿路径第一个不可行网格点的索引，无则 −1）、`violated_dims`（该点违反的感官维度）、`n_recheck_fail`（化学上界在潜空间满足而原尺度复查不通过的网格点数）、`discontinuous`（可行集不是 α=0 起的连续区间则为 True）；α=0 不可行时抛 `RuntimeError`（源表示下应精确可行）。
- `alpha_fit(gen, feasible, u_path, target, log_prior) -> float`：在可行网格点中取目标风格后验 log p̂(t | u(α)) 最大者，后验 = log N(u; mu_t, C) + log_prior_t − logsumexp 各风格；并列取较小 α。log_prior 为 250 款的风格比例。

### T3 `evaluate.py`

- `StyleDiscriminator`：对全部 250 款的 x_filled（225 列）做 `StratifiedKFold(5, shuffle=True, random_state=SEED)` 按风格分层，每折训练 `RandomForestClassifier(500, random_state=SEED, n_jobs=1)`（与预检查 7 相同）；`fold_of(b)`；`p_target(b, x225, style)`：用 b 所在折的判别器（没见过 b）；`p_real_oof(style)`：该风格真实啤酒的折外 P(style)；`ref_q25(style)`：其 25% 分位。
- `d2(gen, x225_or_u, style)`：对 x 先 `gen.latent`，(u − mu_t)ᵀ C⁻¹ (u − mu_t)；各方法都用此式，保证可比。
- `retention(u_b, mu_s, u_new, mu_t)`：Spearman。
- `change_amount(u_b, u_new, C)`：马氏平方范数；`n_changed_cols(x_src, x_new, tol=1e-9)`。
- `category(p_new, p_src, alpha_star, ref_q25)`：按下列顺序判定，先满足者为准：p_new ≥ ref_q25 为 reach；否则 p_new − p_src ≥ 0.05 为 improved；否则 alpha_star = 0 为 kept；其余为 not_improved。四类代码写成常量。
- `failure_reason(...)`：对 kept 与 not_improved，返回 `first_infeasible` 点违反的感官维度，以及这些维度的 Q2 Top-3 化合物迁移前后的原尺度取值与潜变量变化。Q2 输入按方法论 §7 取合并 Top-3、风格内 Top-3 与共变标记：合并 Top-3 读 `final_top3_with_rationality.csv` 中 `method == "main"` 的行，取 `dimension`、`rank`、`compound`、`sign`、`covariation`（不读 `method == "M3"` 的对照行）；风格内 Top-3 读 `full_controls_results.csv` 的 `Within_top3`（以 `|` 分隔，按名次排列）。两表均为 Q2 冻结产物（49 维，无 barnyard）；维度名与 Q1 的 `DIMS` 对不上时写 NaN 并给出原因，不自行映射。

### T4 对照与对象

对每个源啤酒 b 与目标风格 t：
- 主方法：终点 u*（W5a 的 `u_star`），规则 grade，取 α*；同时记 α_fit。
- T0：ũ，无约束，原尺度 `mg.inverse(ũ)`；记化学与感官违反。
- T1：u_trunc 终点，不设感官层，原尺度 `mg.inverse(u_trunc)`；记违反。
- T3：`shift_raw`；记违反。
- T4：α=0，即 x°。
- T5：路径终点 u_trunc，规则 grade，取最大可行 α。
- T2：目标风格真实啤酒中，原尺度 225 列满足 x_t[idx] ≤ b（用 b 对 x_filled 的目标啤酒比较）且 Q1 对该真实啤酒原 231 列输入的感官等级满足 `sens_feasible` 的候选；取 u_full 与 ũ 马氏距离最近者；无候选记 NaN 与原因 no_feasible。源风格与目标风格相同的对不做。
全部 22×21 个风格对做主方法、T0、T4；T1、T3、T5、T2 只做 Pils/Lager → Tripel；`--all-controls` 开关可扩到全部风格对，默认关闭。

### T5 `code/scripts/solve_q3_transfer.py`

命令行：`--rule grade|mu_cont|grade_quant|none`（默认 grade）、`--chem-suffix`（读 W5a 对应后缀的 npz，供 W6 使用）、`--all-controls`、`--smoke`。每个源风格一个中间文件。
输出 `q3_transfer{suffix}.npz` 与以下 CSV：
- `q3_transfer_pils_tripel.csv`：11 款逐款；列含 `beer`、`row_index`、`alpha_star`、`alpha_fit`、`n_active`、`binding_sens_dims`、`discontinuous`、`n_recheck_fail`、`category`、`p_src`、`p_t0`、`p_t1`、`p_main`、`p_t4`、`p_t5`、`p_t2`、`d2_*`、`retention`、`change_*`、`src_missing_diacetyl`。
- `q3_transfer_defects.csv`：每款 × 9 个化学缺陷列，含迁移前后取值、OAV（无阈值为 NaN）、上界 b、是否有效约束、是否源缺失；每款 × 6 个感官维度，含迁移前后等级与原尺度复查结果。
- `q3_transfer_top_changes.csv`：每款 |u*_j − u_bj| / sqrt(C_jj) 的前 15 个，附 Q2 模块号。
- `q3_transfer_portraits.csv`：每款迁移前、迁移后、目标风格真实啤酒（Q1 `oof_mu` 对 5 次重复取平均后取目标风格啤酒的中位数）三组 50 维画像。
- `q3_transfer_pairs.csv`：462 行，每个有序源风格/目标风格一行且 `(src, tgt)` 唯一，`src != tgt`。列含 `src`、`tgt`、`n_src`、`alpha_star_mean`、`alpha_star_q25`、`alpha_star_q75`、`frac_alpha1`、`frac_alpha0`、`frac_t0_chem_viol`、`frac_t0_sens_viol`、`p_main_median`、`p_t4_median`、`frac_ref`、`frac_improved`、`frac_unchanged`、`frac_no_improve`、`n_src_missing_defect`。四个 `frac_*` 结论列依次对应方法论 §6.4 的“达到参照范围”“改善”“保持原谱”“未改善”，按该节的先后顺序判定，逐对四列之和为 1；这些列供 W6 的论文汇总表使用。
另报：两层违约率，按全部啤酒与只按化学层可完全判定的啤酒（不含源缺失缺陷单元格）分别计算（Pils/Lager → Tripel 为 9/11 可完全判定）；感官基准与 Q1 对原输入预测等级不同的啤酒数。

## 验收标准

通用：`black --check code/` 与 `ruff check code/` 通过；`src/q3/` 无 `__main__`；脚本写 `seed` 与 `meta`。

pytest（`code/tests/test_q3_sens_search.py`，用假的线性感官模型与合成生成器，秒级）：
- `transfer_input231(sr.u)` 等于 `source_input231`（1e-9），非缺失单元格与原输入相差不超过 1e-9。
- `path_search`：α=0 总是可行；构造感官等级沿路径先升后降的假模型，`discontinuous` 为 True 且 `alpha_star` 取最大可行点。
- 化学上界在潜空间满足但原尺度复查不通过的构造样本，`n_recheck_fail` 大于 0 且该点不可行。
- `category`：四类各一个构造样本，且 reach 优先于 improved、improved 优先于 kept。
- `alpha_fit`：并列取较小 α；只在可行点中选。
- 无泄漏：判别器 `p_target(b, …)` 的训练折不含 b（断言 b 不在该折训练索引内）。

数值核对（预运行数值依赖代理模型与旧基准，按方法论 §6.8 不作核对目标；以下为一致性检查，写入 `q3_transfer.json` 的 `checks`）：
- 全部 250×21 个源啤酒与目标风格组合，α=0 可行；`transfer_input231` 在 α=0 处与源输入一致。
- T4 的两层违约率为 0；主方法、T5、T2 的原尺度复查两层违约率为 0。
- 主方法的 α* 全在网格上；`n_recheck_fail` 的总数写入回报。
- 方法论 §6.4 的预运行叙述（T0 对 11 款均违反化学约束，投影有效约束 3–5 个）作为参考，不作停止条件；偏离写入回报。
- 正式结果照实写入；不通过放宽阈值或规则消除失败，α* 小或为 0 只表示这条路径受限。

耗时估计：22 个源风格 × 全部啤酒 × 21 个目标 × 21 个 α 网格点，约 11 万次 231 列输入与 50 个模型预测；以实测回报，必要时按源风格并行。

## 处理结果
<!-- [PLACEHOLDER] | Await: 前置条件满足后 coding 回复 ack 并逐项回报 -->
