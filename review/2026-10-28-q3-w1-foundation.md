---
id: 2026-10-28-q3-w1-foundation
from: modeling
to: coding
status: open
severity: major
refs:
  - review/2026-10-28-q3-index.md
  - model/q3_methodology.md
  - experimentalCode/q3_methodology_checks.py
  - experimentalCode/results/data_audit/chemical_dictionary.csv
  - code/data/processed/q2/graph_nodes.csv
  - code/src/q1/paths.py
---

## 问题

Q3 缺少包结构、数据载入、秩变换、模块读取和综合指标派生的实现。
本工单建立 `code/src/q3/` 的基础模块，并复现预检查 1、4、5 的数值。
本工单不依赖 Q1 正式产物，可立即开始。

## 背景 / 复现

先读 `review/2026-10-28-q3-index.md` 的通用约定，再读 `model/q3_methodology.md` §1、§2、§5.1。
算法参考为 `experimentalCode/q3_methodology_checks.py`：公共部分第 27–61 行，`Marg` 第 63–92 行，`within_resid` 第 95–99 行，检查 1 第 174–215 行，检查 4 第 331–351 行，检查 5 第 354–370 行。
附件在根目录 `data/raw/`；`code/data/raw/` 被忽略，Q3 一律读根目录 `data/raw/`。

## 文件与函数

### T1 `paths.py`、`config.py`

`paths.py`：`ROOT = Path(__file__).resolve().parents[3]`；`RAW_DIR = ROOT/data/raw`；`PROCESSED_DIR = ROOT/code/data/processed/q3`；`SMOKE_DIR = PROCESSED_DIR/_smoke`；`CHUNK_DIR = PROCESSED_DIR/_chunks`；`FIG_DIR = ROOT/figures/q3`；`Q1_DIR = ROOT/code/data/processed/q1`；`Q2_NODES = ROOT/code/data/processed/q2/graph_nodes.csv`；`CHEM_DICT = ROOT/experimentalCode/results/data_audit/chemical_dictionary.csv`；`SEED = 0`；`ensure_dirs()` 创建 `PROCESSED_DIR` 与 `FIG_DIR`。模块导入时不创建任何目录。
`config.py`：`TAU = 0.6`；`N_REPEATS = 5`；`N_FOLDS = 5`；`N_SYNTH = 1000`；`GRID_K = (2,5,10,15,20,25,30)`；`GRID_KAPPA = (0,1,2,5,10,20)`；`ALPHA_GRID = np.round(np.arange(0,1.0001,0.05),2)`；`FLOOR = -1.81781854`；`SMALL_STYLES` 为 Faro、Brut、Scotch、Flanders old brown，载入数据后断言四个名称都出现在风格标签中；`SRC_STYLE` 为 Pils/Lager，`TGT_STYLE` 为 Tripel；`DEFECT_CHEM` 为 9 列，顺序固定：acetaldehyde、DES、DMS、H2S、MeSH、SO2.mg.L.、X2.3.butanedione、X4VG、X.E..2.nonenal；`DEFECT_SENS` 为 dms、metallic、X4vg、t2n、diacetyl、stale_hops。

### T2 `data.py`

- `ChemData`（dataclass）：`X`（250×231 DataFrame）、`style`（长度 250 的字符串数组，来自 `tasting_category_fine`）、`beer`（`beer` 列）、`chem_cols`（`X.columns[3:]`，231 个）、`gen_cols`（225 个）、`composite_cols`（esters_sum、aroma_hops_sum、acids_sum、sulfur_sum、kcalperc）、`internal_std_col`（X3.methyl.2.butenal）。
- `load_chem() -> ChemData`：读 `RAW_DIR/1.csv`。断言：250 行；231 个化学列；`gen_cols` 恰为 `chem_cols` 去掉综合指标和内标；风格 22 类，样本量最小 3、最大 31；`DEFECT_CHEM` 的前 7 个与字典中 `off_flavors` 非空的字段集合相等，后 2 个为 X4VG 与 X.E..2.nonenal。
- `load_dictionary() -> DataFrame`：读 `CHEM_DICT`，`encoding=utf-8-sig`，以 `csv1_field` 为索引；断言 `origin_paper` 等于 Internal standard solution 的字段恰为 X3.methyl.2.butenal。
- `is_log(col) -> bool`：字典 `transform_interpretation` 以 log10 开头。
- `threshold(col, kind=threshold_mean) -> float`：阈值缺失返回 NaN；`is_log` 时返回 log10(阈值)，否则返回阈值。
- `defect_table(chem, dic) -> DataFrame`：每个缺陷列一行，列为 `field`、`scale`、`unit`、`threshold_mean`、`threshold_median`、`n_na`、`n_na_pils`，以及两种阈值口径下 OAV ≥ 1 的比例（全体、Pils/Lager、Tripel、Kriek、Lambic）。浓度取 `10**X`（log10 列）或 `X`；OAV = 浓度 / 阈值。

### T3 `marg.py`

`Marg(tie_scheme)`，`tie_scheme` 取 cond_mean（主设定）或 mean_rank（敏感性）。
- `fit(M)`：`M` 为 n×p，允许 NaN；每列只用非缺失值。取唯一值与计数，累积概率区间为 (l_k, h_k]。cond_mean：块潜变量 e_k = (pdf(ppf(l_k)) − pdf(ppf(h_k))) / (h_k − l_k)，pdf、ppf 为 `scipy.stats.norm` 的密度与分位函数，ppf 为 ±∞ 时 pdf 为 0。mean_rank：e_k = ppf((平均秩 − 0.5) / n)。逆变换结点：计数 ≥ 2 的块取两个结点 (l_k + 1e-9, v_k)、(h_k − 1e-9, v_k)；计数为 1 的块取一个结点 (cdf(e_k), v_k)。
- `transform(M)`：每列 `np.interp(M[:, j], 唯一值, e)`；输入含 NaN 时抛 `ValueError`，由调用方先填充。
- `inverse(U)`：P = cdf(U)，每列 `np.interp(P[:, j], 结点概率, 结点取值)`；超出结点范围取端点值，不外推。
参考实现为 `Marg` 第 63–92 行；mean_rank 分支为新增，cond_mean 分支与参考实现逐行等价。

### T4 `modules.py`

- `modules_from_corr(abs_corr, tau=TAU) -> ndarray[int]`：D = clip(1 − abs_corr, 0, None)，对角置 0，取 (D + Dᵀ)/2，`squareform(checks=False)`，`linkage(method=average)`，`fcluster(t=1−tau, criterion=distance)`。
- `modules_from_data(X_filled, tau=TAU)`：对均值填充后的数据取 Spearman 相关的绝对值（NaN 置 0），再调用上式；交叉验证中用训练折数据。
- `modules_from_q2(gen_cols, path=Q2_NODES) -> ndarray[int]`：读 `graph_nodes.csv` 的 `node`、`module_id`，按 `gen_cols` 顺序取出，断言 225 列全部存在；标签按首次出现顺序重编为 0 到 M−1。
- `n_multi(labels) -> int`：成员数 ≥ 2 的模块数。

### T5 `composite.py`

`CompositeDeriver`（方法论 §2.1、§5.3）：
- `fit(U_gen, X_comp)`：`X_comp` 为 n×5 的综合指标观测，先用列均值填充 NaN（均值保存在对象内），再 `Marg().fit` 与 `transform` 得潜变量 V；对每个综合指标拟合 `RidgeCV(alphas=np.logspace(-1,4,30))`，输入为 `U_gen`（225 个生成列的潜变量），目标为该列的 V。
- `predict_latent(U_gen) -> ndarray[n,5]`。
- `observed_latent(X_comp) -> ndarray[n,5]`：用保存的列均值填充后 `transform`。
- `to_original(V) -> ndarray[n,5]`：`Marg.inverse`。
参考实现为检查 4 第 331–351 行。

### T6 `code/scripts/solve_q3_data.py`

写出 `q3_data.npz`、`q3_checks.json`、`q3_defect_table.csv`。
`q3_data.npz` 字段：`gen_cols`、`chem_cols`、`style`、`beer`、`row_index`（0–249）、`na_mask_gen`（250×225 bool）、`x_filled_gen`（列均值填充）、`u_full`（全数据 `Marg` 对填充数据的潜变量）、`x_src`（`Marg.inverse(u_full)`，源表示）、`module_q2`（225）、`seed`、`meta`。
`q3_checks.json`：每项含 `name`、`expected`、`observed`、`tol`、`pass`。

## 验收标准

通用：`black --check code/` 与 `ruff check code/` 通过；`src/q3/` 无 `__main__`，导入时不读写文件；`solve_q3_data.py` 写 `seed` 与 `meta`。

pytest（`code/tests/test_q3_marg.py`、`test_q3_data.py`，全部秒级）：
- `Marg` 往返：非缺失训练值满足 `|inverse(transform(x)) − x| ≤ 1e-9`。
- 下限原子：构造含两个以上并列最小值的列，`inverse` 对落在该块概率区间内的潜变量返回该最小值。
- `inverse` 输出落在训练观测的 [min, max] 内；`transform` 对含 NaN 的输入抛 `ValueError`。
- `modules_from_corr` 对两块的块对角相关矩阵（块内 |ρ| 为 0.9，块间 0.1，τ 为 0.6）返回两个模块。
- `defect_table` 返回 9 行。

数值核对（写入 `q3_checks.json`，容差见括号，均取自预检查输出与方法论正文）：
- 生成列 225；风格 22 类，样本量 3–31。
- 均值填充后，最小值并列（≥ 2 款）的列 156 个、7,592 个单元格；共同下限 −1.81781854 出现在 142 列、6,807 个单元格。
- 全数据潜变量方差：最小 0.46、10% 分位 0.87（±0.01）。
- 非缺失训练值往返最大误差 ≤ 1e-9。
- amyl.acetate 的 NA 为 27 个。
- 风格 η²：中位 0.250、四分位 0.178 与 0.347（±0.005）；置换零分布中位 0.081（±0.003；200 次，`default_rng(0)`，逐次 `rng.permutation(style)`）；理论值 (22−1)/(250−1) = 0.084；超过逐列置换 95% 分位的列 198（±2）。η² 公式：Σ_g n_g (mean_g − mean)² / Σ (u − mean)²，参考第 197–203 行。
- 合并风格内残差协方差的累计方差占比，前 1、10、30 个主成分：0.18、0.50、0.72（±0.01）。
- 综合指标岭回归 5 折 R²（`KFold(5, shuffle=True, random_state=0)`）：aroma_hops_sum 0.78，esters_sum 0.94，其余三项介于二者之间（±0.01）。
- 双乙酰：NA 共 9 个，其中 Pils/Lager 2 个，记录这 2 款的 `beer` 值与行号，代码中不写死；非缺失 241 个；列均值 2.7458238779128634（±1e-9）；列均值位于非缺失观测的 93.8% 分位（±0.5 个百分点）；阈值均值 0.299，列均值为阈值的 9.2 倍（±0.1）。
- 60 个 NA 生成单元格往返后全部偏离填充值，最大偏差 9.3e-4（±1e-4）；双乙酰 9 个 NA 的往返值 2.746749（±1e-6）。
- 缺陷表，均值口径超阈比例（±0.01）：DES 全体 0.18；DMS 全体 0.67、Pils/Lager 0.27、Tripel 0.89、Kriek 0.10、Lambic 0.82；H2S 0.92、1.00、1.00、0.90、0.91；双乙酰 0.45、0.09、0.46、0.80、0.36；X4VG 全体 0.06，其余 0；acetaldehyde 与 SO2 全为 0。中位口径全体：双乙酰 0.69、DMS 0.62、SO2 0.004、X4VG 0.07（±0.01）。
- Q2 模块表：使用 Q2 正式运行的 τ=0.6 `graph_nodes.csv` 标签，按 `gen_cols` 顺序对齐并断言 225 列齐全；记录 `n_multi(module_q2)` 的实际值。预运行中的 33 个多成员模块只作回归核对。若正式值不同，照实记录并继续使用 Q2 正式标签；不得为匹配 33 而改模块算法、阈值或标签。模块数不一致本身不构成失败或待 Modeling 决策。

## 处理结果
<!-- [PLACEHOLDER] | Await: coding 回复 ack 并逐项回报 -->
