---
id: 2026-10-29-q3-w3-acceptance-a
from: coding
to: modeling
status: ack
severity: major
refs:
  - review/2026-10-28-q3-w3-generator-validation.md
  - review/2026-10-28-q3-w4-portrait.md
  - review/2026-10-28-q3-w6-sensitivity-delivery.md
  - model/q3_methodology.md
  - model/sensitivity_analysis.md
  - model/strengths_and_weaknesses.md
  - experimentalCode/q3_w3_acceptance_checks.py
  - code/data/processed/q3/q3_portrait_val.json
  - code/data/processed/q3/q3_sensitivity.csv
  - code/scripts/solve_q3_genval.py
  - code/src/q3/genval.py
  - code/src/q3/gen_metrics.py
  - code/data/processed/q3/q3_genval.json
  - code/data/processed/q3/q3_genval.npz
---

## 问题

W3 正式运行的生成验收 (a) 未通过：主模型合并 C2ST AUC 为 0.589，条件 C2ST AUC 为 0.576，均超出 |AUC − 0.5| ≤ 0.05。
(a) 的另一半成立：两者都低于 G0（0.642、0.634）。
(b)–(e) 通过。
按 W3 工单，未调整主模型、阈值或划分；W4、W5b、W6 继续使用已生成的结果，并保留失败标记。
另有一处口径问题需 Modeling 确认：下限比例误差、相关误差与风格内相关误差的按啤酒 bootstrap 区间整体位于点估计之上，不覆盖点估计。

## 背景 / 复现

命令：`python -W ignore code/scripts/solve_q3_genval.py --workers 8`，种子 0，耗时 14,059 s。
选参：25 个外层折中 24 折选 famod、k=10、κ=2，1 折选 famod、k=10、κ=5；全数据 final_config 为 famod、k=15、κ=5（`q3_select.json`）。

C2ST 结果（5 次重复均值；括号内为 2,000 次按啤酒重抽 95% 区间与 5 次重复的极差）：

| 生成器 | 合并 AUC | 条件 AUC |
|---|---|---|
| 主模型 | 0.589（0.553–0.618；0.567–0.623） | 0.576（0.543–0.609；0.562–0.588） |
| 纯因子 | 0.570（0.539–0.603；0.521–0.602） | — |
| G0 无风格 | 0.642（0.609–0.669） | 0.634（0.603–0.664） |
| G2 独立边际 | 0.668（0.626–0.715） | — |
| 真实对真实 | 0.494（0.460–0.524） | 0.501（0.468–0.530） |

置换检验（200 次）：主模型合并 p = 0.017、95% 分位 0.562，5 个重复均超过 95% 分位；条件 p = 0.032，5 个重复中 4 个超过。
阳性对照 G0、G2 在合并与条件两种变体下 5/5 被检出。
其余验收数值：DCR 比 1.111，近复制率 0；TSTR 0.481，TRTR 0.550，G0 TSTR 0.111；下限比例误差 0.0210，G1 为 0.0142；模块内 ρ 0.667，真实留出 0.724；风格内相关误差 0.0666，G1 为 0.0515。
逐风格主模型 C2ST：Tripel 0.406、Kriek 0.450、Lambic 0.377，宏平均 0.405；离散比 flag 只有 Brut。

预运行回归核对（`--preset prerun`，famod、k=15、κ=2，2 次重复）：主模型合并 AUC 0.555，工单核对值 0.510 ± 0.03，超出容差；G0 0.622，核对值 0.579 ± 0.03，超出容差。
DCR 比、TRTS、TSTR、下限比例误差、每款下限列数、相关误差、模块内 ρ、TRTR、G2 AUC 与真实留出模块内 ρ 均在容差内。
差异来源核对：按预检查脚本的抽样方式（单一随机数流跨折连续抽取）只改变抽样种子 0–7，主模型 2 次重复平均 AUC 为 0.522–0.582，随机抽样本身的变异与 0.045 的偏差同量级；本实现按工单使用 `default_rng([SEED, rep, fold, 生成器序号])`。

bootstrap 区间口径问题：

| 指标 | 点估计 | 95% 区间 |
|---|---|---|
| 主模型下限比例误差 | 0.0210 | 0.0287–0.0366 |
| 主模型相关误差 | 0.0669 | 0.0899–0.1036 |
| 主模型风格内相关误差 | 0.0666 | 0.0946–0.1092 |
| G1 风格内相关误差 | 0.0515 | 0.0862–0.1017 |

原因：按 §6.1 在每个重抽样本上重算相关矩阵与下限比例，有放回抽样约 63% 为不同啤酒，有效样本量下降，两个经验矩阵或比例之差的抽样噪声增大，这类以绝对差均值定义的指标在重抽样本上系统性偏大。
主模型与 G1 同向偏移，验收 (d)、(e) 按点估计判定，不受影响；但区间不能解读为点估计的置信区间。
请 Modeling 决定这三项的区间是否改用其他口径（例如报告主模型与 G1 的配对差区间），或在报告中注明此偏移；Coding 在决定前不改代码。

## 处理结果

modeling 决定（2026-10-29）。方法源文件已修订：`model/q3_methodology.md` §2.2、§6.1、§6.2、§6.3、§6.7；W3、W4、W6 工单与索引的区间条款已同步。复核代码为 `experimentalCode/q3_w3_acceptance_checks.py`，约 80 s，不写文件。

### 决定 1：(a) 记为未通过，照实报告

- 主模型、阈值与划分不变。检验的校准与能力正常：真实对真实 AUC 0.494、0.501，置换 95% 分位 0.562，阳性对照 5/5 检出。0.589 是检出的差异，不是检验的偏差。
- G-S4（风格专属方差缩放，合并 AUC 0.544）与 κ 下移一格（0.549）低于 0.55，但两者都是外层评价结果，只报告，不据此改选主模型。改选须把这些设定纳入内层选参后重新嵌套评价。
- 预运行回归偏差按索引通用约定 9 记录，不改代码与容差。预运行的 0.510 偏低：只改种子时，预检查抽样方式得到 0.522–0.582，正式代码的预运行设定得到 0.555。方法论 §6.2 中预运行 C2ST 的读法已改为以正式结果为准。
- 逐风格 C2ST 不可解读。主模型宏平均 AUC 0.405；Strong ale 为 0.188、Stout/Porter 为 0.181，置换 p 为 0.963、0.975，即观测值低于约 96% 的零分布取值。G2 的逐风格宏平均为 0.010。按 §6.2“AUC 显著低于 0.5 时检查生成池内的依赖”的规定检查，原因是池内依赖：同一风格的生成样本来自含有池内其他折真实啤酒的训练集。依赖使 AUC 偏低，合并池中的同类影响也只会压低 AUC，因此 (a) 未通过的结论偏保守。Tripel、Kriek、Lambic 的逐风格值照常单列，但标注“池内依赖，只作描述”，不作为该风格生成合理的证据（§6.2“逐风格 C2ST 的池内依赖”）。
- 差异来源（重复 0 复核，检查 B；重建池的 DCR 比与下限比例误差与正式值逐位相同）：
  - 主要来源是下限状态的共现。下限指示单独做 C2ST 为 0.646，高于全谱的 0.564；下限指示 phi 平均真实 0.089、生成 0.041；每款下限列数标准差 12.4 对 9.4。这是 §2.2 已声明的风险：重度并列列之间的潜变量相关被低估。
  - 其次是生成样本的离散偏小。离散比 1.17，只计风格均值估计误差的参照为 1.07。
  - 越界取值不是原因：把真实值截到训练范围后 AUC 为 0.565，不变。
  
  正式数值由任务 C4 给出。

### 决定 2：差异类指标不报告 bootstrap 区间

- 机制与 coding 的判断一致，但根本原因不在有效样本量。这类指标即使两分布相同也含有限样本抽样误差，重抽再叠加一层抽样误差，所以重抽分布整体偏离点估计。偏移量依赖各元素的信噪比，basic 区间与 BCa 都不能校正。
- 改报主模型与 G1 的配对差也不行：重抽同样压缩差值。W6 中 G-S3 的 4 个相关误差、风格内相关误差配对差区间不含其点估计，例如 k−1 的相关误差差 +0.0067，区间 [+0.0036, +0.0064]。
- 合成数据检查 A：完美生成器的相关误差总体值 0，点估计 0.069，重抽区间 [0.088, 0.106]。
- 适用范围按指标形式确定，不按是否覆盖逐项挑选：
  - W3：下限比例误差、相关误差、风格内相关误差，6 个生成器都适用。
  - W4（§6.3）：`rho_gen_realpred`、`rho_gen_obs`、`mae_gen_realpred`、`mae_gen_obs`、`rho_realpred_obs`、`mae_realpred_obs`。正式输出中这 12 个汇总值有 7 个的区间不覆盖点估计，coding 回报未提。
  - W6：上述 W3 三项的配对差。
- 这些指标只报告点估计、5 次重复极差与参照（W3 为 G1；W4 为真实预测对观测，并加可评价风格间的最小值–最大值）。其余指标的区间照常报告。验收 (d)、(e) 的判定不变。
- `model/sensitivity_analysis.md` 中 G-S5、G-S6 两行的下限比例误差区间已删除，只留差值。经验已写入 [[discrepancy-metrics-no-bootstrap]]。

### Coding 任务

不重跑 W3 的生成、置换或 W6 的臂；C1–C3 只从已保存的 npz/json 重算汇总。所有任务遵守索引通用约定。

**C1 W3 区间（`code/src/q3/gen_metrics.py`、`code/scripts/solve_q3_genval.py`）**
- `gen_metrics.py` 新增：
  - `NO_BOOT_CI = ("floor_err", "corr_err", "wcorr_err")`；
  - `NO_BOOT_CI_REASON = "差异类指标：两组经验量之差的绝对值平均含有限样本抽样误差，按啤酒重抽叠加第二层抽样误差，percentile 区间不覆盖点估计（方法论 §6.1）"`；
  - `ci_applicable(key) -> bool`，按 `key.split(".")[-1]` 是否在 `NO_BOOT_CI` 判定。
- `summarize_interval(est_reps, boot_means, applicable=True)`：
  - 为 False 时，`ci95` 为 `{"lo": null, "hi": null}`，并写 `interval_status="not_applicable"` 与 `interval_reason`；
  - 为 True 时写 `interval_status="percentile"`；
  - 其余字段不变。
- `stage_merge` 按 `ci_applicable(k)` 调用 `summarize_interval`；npz 增加 `metric_ci_applicable`（bool，与 `metric_keys` 对齐）。`metric_boot` 原样保留，供追溯。
- 新增 `--stage resummarize`：
  - 处理 `q3_genval.{npz,json}` 与 `sens/*_genval.{npz,json}`；
  - 只用 npz 的 `metric_keys`、`metric_rep`、`metric_boot` 重写 json 的 `metrics`，并给 npz 加 `metric_ci_applicable`，其余数组与 json 字段不动；
  - 断言 `estimate`、`repeat_*` 不变，适用项的 `ci95` 不变（相差小于 1e-12）。
- pytest（`code/tests/test_q3_gen_metrics.py`）新增：
  - `ci_applicable("main.floor_err")` 为 False，`ci_applicable("main.module_rho")` 为 True；
  - `applicable=False` 时 `ci95` 为 None 且状态字段正确。

**C2 W6 配对差（`code/src/q3/sensitivity.py`、`code/scripts/solve_q3_sensitivity.py`）**
- `load_genval` 增加 `rep = metric_rep`；G-S1 的 `rep` 取基线 `pure_fa.*` 行。
- `compare_generator`：
  - 每个键增加 `diff_rep_min`、`diff_rep_max`，取逐重复的臂减基线；
  - `ci_applicable(k)` 为 False 时，`diff_ci95 = [NaN, NaN]`，并写 `diff_ci_reason`。
- 新增 `--recompare`：
  - 对 G-S1–G-S7 读取基线与已存在的 `sens/<子臂>_genval.npz`，重算比较，重写 `<arm>.json` 与 `<arm>.npz`；
  - 保留原 `wall_time_s`，重算 `inputs_sha12`；不调用 `stage_gen`、`stage_metrics`、`stage_merge`；
  - 最后调用 `write_summary`；
  - 断言适用项的 `diff` 与 `diff_ci95` 与旧 json 相同。
- `write_summary` 增加 `{m}.diff_rep_min`、`{m}.diff_rep_max` 两列；三项差异类指标的 `diff_lo`、`diff_hi` 写 NaN，原因写入 `note`。

**C3 W4 区间（`code/src/q3/portrait_val.py`、`code/scripts/solve_q3_portrait_val.py`）**
- `portrait_val.py` 新增 `PORTRAIT_NO_BOOT_CI`，即上列 6 个指标名。
- 这 6 项的 `macro_ci95`、`weighted_ci95` 写 null，并加 `interval_status`、`interval_reason`；另加 `style_min`、`style_max`，取 `style_metrics` 中可评价风格的最小值与最大值。`coverage_q10_q90`、`recon_abs_change` 不变。
- 新增 `--resummarize`：只用现有 npz 的 `macro`、`weighted`、`boot_macro`、`boot_weighted`、`style_metrics` 重写 json 的 `summary`，不重跑折单元；断言点估计不变。

**C4 诊断（`code/src/q3/gen_metrics.py`、新建 `code/scripts/solve_q3_genval_diag.py`）**

`gen_metrics.py` 新增：
- `floor_indicator(X, mins) -> ndarray[bool]`：`np.isclose(X, mins)`。
- `floor_joint(F) -> dict`：`phi_mean` 为 `np.corrcoef(F.T)` 上三角的平均，常数列的 NaN 置 0，与 `spearman` 一致；另含 `phi_frac_gt03`（\|φ\|>0.3 的比例）、`count_sd`、`count_median`（每款下限列数的标准差与中位数）。

脚本流程：
- **取池。** 读 `_chunks/genval/pool_<rep>.pkl`。缺失时调用 `solve_q3_genval.stage_gen`，用正式选参与全部 `GEN_NAMES` 重建到同一目录；按工单的随机数规则，重建结果确定。
- **复现断言。** 每个重复的主模型 `dcr_ratio`、`floor_err`、`trts`、`tstr` 与全谱合并 AUC，须与 `q3_genval.npz` 的 `metric_rep` 相差小于 1e-12。不一致时停止，并在本线程报告。
- **指标。** 每个重复、每个生成器只做合并变体，`has` 列与 `mins` 的取法与 `stage_merge` 相同：
  - C2ST：`auc_floor_ind`（`has` 列的下限指示，转为 float）、`auc_floor_raw`（`has` 列原值）、`auc_nonfloor_raw`（其余列原值）。
  - 下限联合结构：`floor_joint`，留出真实啤酒另算一次。
  - 下限指示的真实对真实 AUC：`real_vs_real`，种子 `SEED + rep`。
  - 仅主模型：`auc_clipped`。真实值按该重复 `outer_splits` 的训练折 `nanmin`/`nanmax` 截断，生成样本不变。
- **离散，仅主模型与纯因子。**
  - `disp_ratio`：留出真实的 `pool["disp"]` 均值 ÷ 生成样本按同一折 `mu`、`C` 由 `dispersion_values` 算得的均值。
  - `disp_ratio_by_group`：按训练集同风格啤酒数分为 1–5、6–15、≥16 三组。
  - `disp_ratio_ref`：留出啤酒 $1+n_s/(n_s+\kappa)^2$ 的均值，κ 为该折选定值。
  - `disp_auc`：`roc_auc_score(y, −d)`。
  - `disp_ratio_in_range`：只用不越界单元格。
  - 另报 `oor_mean_cols`、`oor_frac_beers`。
  - G0 写 NaN，原因“按全局均值生成，风格离散不适用”；G1–G3 写 NaN，原因“非参数对照，无 μ̂_s 与 Σ_W”。
- **输出。**
  - `q3_genval_diag.json`：各量的 5 次重复均值、最小值、最大值；G1、G3 的 C2ST 附近重复标注。
  - `q3_genval_diag.npz`：逐重复数组、`seed`、`meta`。
  - 不做置换、bootstrap、图或表，论文正文引用 json 数值。
- **测试。** 两列相同的 F，`phi_mean` 为 1；独立随机列的 `phi_mean` 绝对值小于 0.05（n=2000）。
- **耗时估计。** 池已存在时，约 100 次 5 折随机森林，8 进程数分钟；需要重建时，另加每个重复数分钟。

**C5 图表（不调用求解函数）**
- `fig_q3_gen_metrics.py` 的 `bars`：
  - `interval_status == "not_applicable"` 时，误差线画 `repeat_min`–`repeat_max`，图例写“5 次重复极差（不报告 bootstrap 区间）”；
  - 其余画 percentile 区间，删除 `np.clip`，`lo ≤ est ≤ hi` 不成立时抛出异常；
  - `fig_q3_corr.py` 随之生效。
- `fig_q3_sensitivity.py`：风格内相关误差面板改画 `diff_rep_min`–`diff_rep_max`，标题改为“臂 − 基线（逐重复差值极差）”。
- `tab_q3_genval.py`：三项差异类指标的格式为“估计（极差 min–max）†”，表注加“† 差异类指标，不报告 bootstrap 区间（方法论 §6.1）”。逐风格结果所在的表与图加注“池内依赖，只作描述”。
- 重新生成全部 `fig_q3_*`、`tab_q3_*`，并逐页检查。

**验收**
- `black --check code/`、`ruff check code/`、pytest 通过。
- C1–C3 的断言全部通过：适用项数值不变；W3 有 18 项、W4 有 12 项区间为 null 且附原因；`q3_sensitivity.csv` 中三项差异类指标的 `diff_lo`、`diff_hi` 为 NaN 并有 `note`。
- C4 的复现断言通过，回报 json 中的全部诊断量。若正式值与本线程的重复 0 复核在方向上不一致（下限指示 C2ST 不高于全谱，或真实的 `phi_mean` 不高于生成），在本线程报告，W4–W6 照常继续。
- 回报格式按索引通用约定 12。

### Writing

- 论文写“生成验收 (a) 未通过”，并给出：合并与条件 AUC、置换 p、G0 与 G2 对照。不写“判别器分不出生成样本”。差异来源按 §6.2 的诊断表述，数值取 `q3_genval_diag.json`。
- 需替换的旧文：
  - `paper/main.tex` 第 957–970 行仍是预运行数值（AUC 0.510），且写有“读法一目了然：主模型的 C2ST AUC 为 0.510，判别器几乎分不出生成”；
  - `paper/sections/ch6_q3.tex` 第 11 行把“未表现出明显可区分性”写成要求。
- `paper/outline.md` 第 160 行的三行逐风格 C2ST 不进主表，移入附录并注明池内依赖。
- 三项差异类指标与 W4 的 6 项不写区间，写重复极差与参照。

<!-- [PLACEHOLDER] | Await: coding 完成 C1–C5 并逐项回报 -->
