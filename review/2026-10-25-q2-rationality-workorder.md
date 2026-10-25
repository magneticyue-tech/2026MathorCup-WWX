---
id: 2026-10-25-q2-rationality-workorder
from: modeling
to: coding
status: open
severity: major
refs:
  - model/q2_methodology.md
  - experimentalCode/q2_methodology_checks.py
  - code/src/q2/rationality.py
  - code/src/q2/pipeline_final.py
  - code/src/q2/config.py
  - experimentalCode/results/data_audit/chemical_dictionary.csv
---

## 问题
`model/q2_methodology.md` §6 已改写：删除预期表、命中率、最佳预期化合物排名、来源富集和 X/D/M/S/U 分类。
化学合理性改为两步：先逐项输出附件 8 原始标签、阈值证据和共变标记（§6.1–6.3），再由 Modeling 在结果产生后撰写化学解释（§6.4）。
§4.5 的 A 级改为“Top-1 显著且 $\pi^{c}\ge0.6$”，不再要求合理性类别；这是 §5.3 运行前声明中维度等级的改动，以本线程为记录。
`code/src/q2/rationality.py` 仍按旧 §6 实现，其输出 `final_top3_with_rationality.csv` 中的 `category`、`strict_hit`、`module_hit`、`exp_D`、`exp_M` 列及打印的命中率不再有效。
本工单只规定代码结构、产物和验收；与方法论冲突处以方法论为准，并在本线程回报。

## 背景 / 复现

### 前置：路径
2026-10-25 文件整理（`cdafbb0`）后，原始附件在 `code/data/raw/`，Q2 输出在 `code/data/processed/q2/`。
`code/src/q2/config.py` 仍指向根目录的 `data/raw` 与 `data/processed/q2`，两处已不存在；先改 `DATA_RAW`、`Q2_DIR`，再做下列修改。

### 已由 Modeling 完成
- `experimentalCode/q2_methodology_checks.py` 检查 6 已删除 `EXPECT`、`hits()`、`best_rank`、维度标签置换和来源富集，只保留 OAV 与“进入 ≥5 个维度 Top-3”两段输出。

### 修改 `code/src/q2/rationality.py`
删除 `EXPECTED_TABLE`、严格/模块命中、最佳预期排名和 X 类名单。
删除 `config.py` 中的 `NO_PRIOR_DIMS`（只服务于命中率）。
新增以下函数，`run_rationality()` 依次调用并写出结果：

- `attachment8_labels(compounds: list[str]) -> pd.DataFrame`：从 `experimentalCode/results/data_audit/chemical_dictionary.csv` 按 `csv1_field` 取 `origin_paper`、`off_flavors`、`aging`、`chemical_name`，原样返回，空值保留为空，不改写、不合并（§6.1）。
- `oav_share(compounds: list[str], X_raw: pd.DataFrame) -> pd.Series`：OAV≥1 的啤酒比例（§6.3）。阈值用 `threshold_mean`；`transform_interpretation` 以 `log10` 开头的列先取 $10^x$，其余按线性处理；无阈值返回 NaN。口径与 `experimentalCode/q2_methodology_checks.py` 检查 6 一致。
- `breadth(top3: pd.DataFrame) -> pd.Series`：每个化合物进入多少个维度的 Top-3，barnyard 不计。
- `covariation_flag(top3: pd.DataFrame, oav: pd.Series, n_dims: pd.Series) -> pd.DataFrame`：§6.2 的三项判据各一列（`flag_style_ci0`、`flag_breadth5`、`flag_oav_lt1`）及其“或” `covariation`。风格内 $r^{(s)}$ 的 95% CI 取 §4.6 的输出；若现有流程尚未产出该 CI，在本线程报告，暂填 NaN，不自行定义。

barnyard 行不再映射到 stale_hops 的任何判定，只保留“同 stale_hops”注明（§1 交付范围）。

### 修改 `code/src/q2/pipeline_final.py`
`run_oav_analysis()` 现在按名称列模糊匹配附件 8、按“中位数 < 0 或存在负值”猜测 log10，口径与 §6.3 不一致。
改为直接使用 `rationality.py` 输出中的 `oav_ge_1_ratio`，删除本函数的阈值读取与变换判断。

### 输出产物
- `code/data/processed/q2/final_top3_with_rationality.csv`（`config.Q2_DIR`），每个维度 3 行，列为：`dimension`、`rank`、`compound`、`chemical_name`、`module_id`、`sign`（入选符号）、`origin_paper`、`off_flavors`、`aging`、`threshold_mean`、`oav_ge_1_ratio`、`n_dims_top3`、`flag_style_ci0`、`flag_breadth5`、`flag_oav_lt1`、`covariation`、`interpretation`。
- `interpretation` 列留空，由 Modeling 按 §6.4 填写；代码不生成任何解释文本或类别。
- 主方法与 M3 Top-1 不同的维度，M3 的 Top-1 另写一行，增加 `method` 列区分 `main` / `M3`（§5.3）。
- 维度等级 A/B/C 按新的 §4.5 计算，写入 `grade` 列。

## 验收
- `black` 与 `ruff` 通过；`SEED` 取自 `config.py`。
- 代码中不出现任何“维度 → 化合物”的手写映射；`grep -n "EXPECTED_TABLE\|strict_hit\|module_hit\|NO_PRIOR_DIMS" code/` 无结果。
- 对主 Top-3 中有阈值的化合物，`oav_ge_1_ratio` 与检查 6 的输出逐项一致（预运行值见 §6.3，例如 ethanol 0.98、linalool 0.98、DMS 0.67；acetaldehyde、ethyl decanoate、furfural、isobutanol、phenethyl acetate 为 0）。
- `n_dims_top3 ≥ 5` 的化合物集合与检查 6 一致（color 8、ethyl octanoate 7、beta-phellandrene 6、ethanol 6、protein 6、n-hexanol 5、IBU 5）。
- `origin_paper`、`off_flavors`、`aging` 与 `chemical_dictionary.csv` 逐元素相同。
- C 级维度为 A_esters_flower 与 t2n（§4.5 预运行）。

## 处理结果
