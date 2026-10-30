# 啤酒酿造化合物分析与啤酒风味预测：支撑材料代码

2026 年 MathorCup 数学应用挑战赛大数据竞赛赛道 A 初赛支撑材料。
本目录包含问题 1–3 的全部源代码，以及复现论文结果所需的运行说明。
目录不含数据和运行产物；按第 3 节放入数据后，可按第 5 节的顺序从头复现全部实验。

## 1. 项目总览

| 问题 | 任务 | 方法要点 | 主要交付 |
|---|---|---|---|
| 问题 1 | 由附件 1 化学数据给出 50 个感官维度的 0–6 级评分 | 每维一个梯度提升回归（HistGradientBoosting）拟合品评小组均分 x̄；嵌套交叉验证（外层 5 次重复 × 5 折，按品评日期分组、按风格分层）；用内层折外残差构造等级预测分布，取分布中位数作为等级 | `q1_final_scores.csv`（250 × 50 等级）、`q1_scores_long.csv`（含等级分布与 90% 区间） |
| 问题 2 | 构建化合物共表达图，给出每个感官维度的 Top-3 关键化合物并论证稳定性与化学合理性 | 以附件 2 的 \|ρ\| 做平均连接层次聚类（τ = 0.6）得到模块；图约束前向选择（Top-2、Top-3 不与已选化合物同模块）；置换门槛与按日期组的 bootstrap 给出显著性、入选频率 π^c 与 Jaccard；M0–M3 与风格内选择作对照 | `final_top3_with_rationality.csv`（49 维 Top-3、显著性、稳定性、合理性证据） |
| 问题 3 | 风格条件生成化合物谱，给出感官画像，并做保持缺陷不恶化的跨风格迁移 | 逐列秩变换到潜在正态空间；潜空间风格条件高斯（风格均值收缩 + 因子协方差 + 按问题 2 模块分块的特异协方差），嵌套交叉验证选参；生成谱经问题 1 模型得到画像；迁移为潜空间平移加缺陷约束投影，沿路径检查感官缺陷维度 | `q3_synthetic_profiles.npz`（22 风格 × 1000 个生成谱）、`q3_portraits_long.csv`、`q3_transfer_pils_tripel.csv` |

代码注释中的 `§x.y` 指论文各问方法章节的编号。
各求解脚本运行时会逐项打印数值核对结果（`[PASS]`/`[FAIL]` 或 `PASS`/`FAIL`）；任一核对失败时脚本仍写出产物，随后以非零退出码结束并列出失败项。

## 2. 环境依赖

| 项目 | 要求 |
|---|---|
| Python | 3.10（在 3.10.22 下验证） |
| 第三方库 | 见 `requirements.txt`；`scikit-learn` 必须为 1.7.2，求解脚本启动时强制检查 |
| 中文字体 | 绘图使用 Microsoft YaHei、SimHei 或 Noto Sans CJK SC 之一；缺少时中文标签显示为方框，不影响数值结果 |
| CPU | 并行脚本默认 8 个进程（`--workers` 可调） |

安装：

```bash
pip install -r requirements.txt
```

运行前设置以下环境变量。
线程变量已在各求解脚本内部设为 1，这里只需设置编码与 MKL 线程层：

```bash
export PYTHONIOENCODING=utf-8
export MKL_THREADING_LAYER=SEQUENTIAL
```

Windows PowerShell 中写作 `$env:PYTHONIOENCODING="utf-8"`、`$env:MKL_THREADING_LAYER="SEQUENTIAL"`。
使用基于 MKL 的 numpy 时，未设置 `MKL_THREADING_LAYER=SEQUENTIAL` 的进程可能在首次调用 BLAS 时直接退出（返回码 127，无错误信息）。

**所有命令都在本目录（`sprtMaterial/`）下执行。**

## 3. 数据入口

全部输入放在 `data/raw/`，文件名保持不变：

| 文件 | 来源 | 用途 |
|---|---|---|
| `1.csv`–`7.csv`、`3.xlsx`、`8.xlsx` | 赛题附件 1–8 | 全部问题 |
| `raw_sensory_data_anonymized.csv` | Zenodo 记录 23624915（DOI `10.5281/zenodo.23624915`）中的 `Raw sensory data anonymized.csv`，即附件来源论文发布的逐品评员原始评分（2,197 条，cp1252 编码） | 问题 1 训练目标 x̄、品评日期分组；问题 2、3 经问题 1 产物使用 |
| `Supplemental Files and Figure source files.xlsx` | Zenodo 记录 10653704（DOI `10.5281/zenodo.10653704`），md5 应为 `52645e38d90aafda272532e19e2fea1c` | 问题 1 外部检查 `solve_q1_external.py` 与问题 2 加标实验表 `solve_q2_spike.py`；也可用环境变量 `ZENODO_XLSX` 指定其路径 |

附件 1–8 与 Zenodo 10653704 中 Schreurs et al. (2024, *Nature Communications* 15:2368) 发布的数据一致；原始评分由同一作者在该数据记录的新版本中发布。

第 0 步数据审计由附件 1–8 生成两份派生输入，写入 `experimentalCode/results/data_audit/`：

| 派生文件 | 内容 | 使用者 |
|---|---|---|
| `chemical_dictionary.csv` | 化学列字典：变换类型、阈值（附件 8 *Sensory thresholds*）、缺陷标签、单位 | 问题 1 OAV 实验、问题 2 合理性、问题 3 缺陷约束 |
| `calibrated_discordances.csv` | 附件 1 与附件 8 校准表冲突的 80 个单元格 | 问题 1、2、3 的附件 8 敏感性臂 |

## 4. 目录结构

```
sprtMaterial/
├── README.md
├── requirements.txt
├── data/raw/                          输入数据入口（见第 3 节）
├── experimentalCode/
│   ├── dataAudit/                     第 0 步：数据审计
│   └── results/data_audit/            审计输出（运行后生成）
├── code/
│   ├── src/q1/  src/q2/  src/q3/      各问的可导入模块
│   ├── scripts/                       求解、绘图、制表与导出入口（问题 2 的论文图表也在此）
│   ├── tests/                         单元测试与产物检查（pytest）
│   └── data/processed/q1|q2|q3/       各问求解产物（运行后生成）
└── figures/q1|q2|q3/                  论文图表（运行后生成）
```

问题 1、3 采用“`src` 模块只定义计算、不读写文件；`scripts` 入口负责读写”的划分：求解脚本 `solve_*` 写 `.npz`/`.json`/`.csv` 到 `code/data/processed/`，图表脚本 `fig_*`/`tab_*` 只读取这些产物，不重新求解。
问题 2 的各步骤是 `code/src/q2/` 下可直接运行的模块，由 `run_all.py` 统一调度；问题 2 的论文图表由 `code/scripts/` 中的 `solve_q2_*`、`fig_q2_*`、`tab_q2_*` 在 `run_all.py` 之后生成。
所有图表数值都从产物文件读取，脚本中不写入任何数据值。

## 5. 运行顺序与问题间依赖

```
第 0 步  数据审计 ─────────────┬──────────────┬──────────────┐
                               ▼              ▼              ▼
问题 1   数据 → 主模型 → 评分 → 对照 → 指标 → 敏感性/压力/外部/OAV → 导出 → 图表
                    │                            │
                    │ q1_data、q1_main_oof       │ q1_models、q1_scores、q1_metrics、
                    ▼                            │ q1_models_drop_sum
问题 2   run_all.py（15 步）                     │
                    │ graph_nodes、Top-3 表、    │
                    │ 附件 8 替换表              │
                    ▼                            ▼
问题 3   数据 → 选参 → 生成验证 / 化学层迁移 / 画像 / 画像验证 → 感官层迁移 → 敏感性 → 图表
```

必须按 **第 0 步 → 问题 1 → 问题 2 → 问题 3** 的顺序运行：

- 问题 2 读取问题 1 的 `q1_data.npz`（x̄、日期组、风格）与 `q1_main_oof.npz`（外层划分 `fold_id` 与逐折参数，供 M3 对照复用），并导入 `src/q1` 的维度分组、单调约束和模型构造函数。
- 问题 3 读取问题 1 的全数据模型 `q1_models/`、`q1_scores.npz`（残差与分位数切点）、`q1_main_oof.npz`、`q1_data.npz`、`q1_metrics.npz`，敏感性臂 G-S8 另读 `q1_models_drop_sum/` 与 `q1_drop_sum_final.json`；还读取问题 2 的模块表 `graph_nodes.csv`、Top-3 表与附件 8 替换表。
- 问题 2、3 复用问题 1 的划分时读取 `fold_id`，不重新调用划分函数；`StratifiedGroupKFold` 的打乱随 scikit-learn 版本变化，这是固定 1.7.2 的原因。

### 第 0 步：数据审计

```bash
python experimentalCode/dataAudit/run_data_audit.py
```

| 脚本 | 功能 |
|---|---|
| `run_data_audit.py` | 入口，依次调用下面三个脚本；日志写入 `audit_run_log.txt` |
| `audit_data.py` | 附件 1–8 的逐列概况、缺失单元、工作簿结构与 SHA-256 清单 |
| `audit_deep.py` | 字段与样本编号核对、变换底数验证、附件 1/8 冲突单元、附件 2/6 相关复算、下限与并列统计 |
| `audit_supplement.py` | 化学、感官、论坛三份字典与极值记录 |

耗时约 20 秒。审计只读取 `data/raw/`，不修改原始数据。

## 6. 问题 1：缺失条件下的感官评分

### 6.1 模块（`code/src/q1/`）

| 文件 | 功能 |
|---|---|
| `paths.py` | 路径常量与全局种子 `SEED=0` |
| `meta.py` | 运行元信息；scikit-learn 版本检查 |
| `data.py` | 维度与维度组常量；读取原始评分并构造 x̄、等级真值 `G = floor(clip(x̄,0,6)+0.5)`、日期组；化学缺失编码 `ChemNAEncoder`（主设定为训练折列均值） |
| `panels.py` | 由附件 4 推断品评团（两阶段对照 C1、C2 使用） |
| `features.py` | 化合物–感官单调约束对 |
| `splits.py` | 外层/内层划分、留一风格划分、70/30 划分 |
| `model.py` | 24 项超参数网格、模型构造、内层选参、外层折拟合、嵌套交叉验证、全数据重拟合 |
| `discretize.py` | 预测分布、分布中位数与区间、分位数切点、QWK 切点、六种离散化规则 |
| `baselines.py` | 对照 B0、B1、lin10、ElasticNet、随机森林、化学加风格、B2（复现原论文流程） |
| `ordinal.py` | 有序二分类对照 K2 |
| `twostage.py` | 两阶段对照 C1（只用附件）与 C2（C1 加校准） |
| `metrics.py` | 等级 MAE、完全一致率、QWK、平衡准确率、RPS、对数损失等；按日期组的配对 bootstrap |
| `sensitivity.py` | 8 个敏感性臂的设定与合并 |
| `external.py` | Zenodo 验证实验对照杯检查 |
| `oav.py` | 气味活度值（OAV）特征实验 |
| `export.py`、`report_beers.py` | 交付宽表、长表；报告用的 12 款啤酒 |
| `plotstyle.py`、`tables.py` | 绘图样式；TeX 表输出 |

### 6.2 求解脚本（`code/scripts/`）

| 步骤 | 命令 | 输入 | 输出（`code/data/processed/q1/`） | 参考耗时 |
|---|---|---|---|---|
| 1 | `python code/scripts/solve_q1_data.py` | `1.csv`、`4.csv`、原始评分 | `q1_data.npz`：x̄、G、日期组、风格、数据核对 | 数秒 |
| 2 | `python code/scripts/solve_q1_main.py --workers 8` | 步骤 1 | `q1_main_oof.npz`（折外与内层预测、`fold_id`、逐折参数）、`q1_models/`（全数据模型） | 约 2.4 小时 |
| 3 | `python code/scripts/solve_q1_scores.py` | 步骤 2 | `q1_scores.npz`：预测分布、六种规则等级、`G_hat`、区间、`final_resid`、`final_qcuts` | 约 30 秒 |
| 4a | `python code/scripts/solve_q1_baselines.py --workers 8` | 步骤 1–3 | `q1_baselines.npz` | 最长，约 1 天 CPU 时间 |
| 4b | `python code/scripts/solve_q1_ordinal.py --workers 8` | 步骤 1–3 | `q1_ordinal.npz` | 约 2 分钟 |
| 4c | `python code/scripts/solve_q1_twostage.py --workers 8` | 步骤 1–3、`4.csv` | `q1_twostage.npz` | 约 2.2 小时 |
| 5 | `python code/scripts/solve_q1_metrics.py --workers 8` | 步骤 3–4 | `q1_metrics.json`/`.npz`：全部指标、配对比较、预先声明的判定 | 数秒 |
| 6a | `python code/scripts/solve_q1_sensitivity.py --workers 8` | 步骤 2–5、审计输出 | `q1_sensitivity.*`；问题 3 用的 `q1_models_drop_sum/`、`q1_drop_sum_final.json` | 约 25 分钟 |
| 6b | `python code/scripts/solve_q1_stress.py --workers 8` | 步骤 2–5 | `q1_stress.*`：留一风格、70/30 划分 | 约 2 小时 |
| 6c | `python code/scripts/solve_q1_external.py` | 步骤 5、Zenodo 工作簿、`8.xlsx` | `q1_external.*` | 数秒 |
| 6d | `python code/scripts/solve_q1_oav.py --workers 8` | 步骤 3、审计输出 | `q1_oav.*`（可选实验） | 约 5 分钟 |
| 7 | `python code/scripts/export_q1_scores.py` | 步骤 3–5 | `q1_delivery_scores.*`、`q1_final_scores.csv`、`q1_scores_long.csv` | 数秒 |
| 8a | `python code/scripts/solve_q1_panels.py` | `4.csv`、步骤 1 | `q1_panels.npz`：对照 C1 的精确并列连边、品评团、组维零点与地板堆积 | 数秒 |
| 8b | `python code/scripts/solve_q1_spearman.py` | 步骤 2、4a、4c | `q1_spearman.npz`：各模型平均 Spearman 及按日期组 bootstrap 的 95% 区间（2,000 次） | 约 1 分钟 |
| 8c | `python code/scripts/solve_q1_mscale.py` | 步骤 2、4c | `q1_mscale.npz`：C1 尺度参数 m ∈ [2, 4.5] 的参照还原误差，原始评分直接估计的 m | 数秒 |

步骤 8 只为论文图准备数据，复用正式划分 `fold_id`；`solve_q1_mscale.py` 在 m = 3 时核对与 `q1_twostage.npz` 的参照逐格一致，不一致时以非零退出码结束。

步骤 4a 可用 `--only rf`、`--only chemstyle` 等（取值 `b0 b1 lin10 enet rf chemstyle`）分批运行，最后不带 `--only` 运行一次以合并。
步骤 6a 可用 `--arm NAME` 单独运行某个臂（`na_median_ind na_native_nan na_floor_ind drop_sum att8 mono_on mono_off alpha`）。
`export_q1_scores.py` 按 `q1_metrics.json` 中的判定 `success` 选择交付来源；正式结果为主模型（`model_key=main`）。

### 6.3 图表脚本

| 命令 | 输出（`figures/q1/`） |
|---|---|
| `python code/scripts/fig_q1_rules.py` | `fig_q1_rules.pdf`：六种离散化规则对比 |
| `python code/scripts/fig_q1_confusion.py` | `fig_q1_confusion.pdf`：等级混淆矩阵 |
| `python code/scripts/fig_q1_paired_diff.py` | `fig_q1_paired_diff.pdf`：与各对照的配对差及置信区间 |
| `python code/scripts/fig_q1_reliability.py` | `fig_q1_reliability.pdf`：预测分布校准 |
| `python code/scripts/fig_q1_reference_error.py` | `fig_q1_reference_error.pdf`：只用附件构造参照的误差 |
| `python code/scripts/fig_q1_sensitivity.py` | `fig_q1_sensitivity.pdf`：敏感性臂 |
| `python code/scripts/fig_q1_stress.py` | `fig_q1_stress.pdf`：压力测试 |
| `python code/scripts/tab_q1_{metrics,rules,paired,external,report_beers}.py` | 同名 `.csv` 与 `.tex`；`tab_q1_report_beers` 依赖步骤 7 |

论文正文图（依赖步骤 7、8）：

| 命令 | 输出（`figures/q1/`） | 数据来源 |
|---|---|---|
| `python code/scripts/fig_q1_panel.py` | `fig_q1_panel.pdf`：品评团与零点三子图 | `q1_panels.npz` |
| `python code/scripts/fig_q1_performance.py` | `fig_q1_performance.pdf`：模型对比双子图（平均 Spearman 与配对差区间；主模型与 B1 逐维） | `q1_spearman.npz` |
| `python code/scripts/fig_q1_confusion_anchor.py` | `fig_q1_confusion_anchor.pdf`：有锚 / 无锚单元的等级混淆热力图 | `q1_scores.npz`、`q1_twostage.npz` |
| `python code/scripts/fig_q1_scores.py` | `fig_q1_scores.pdf`：250 × 50 交付等级热力图（行按风格聚类） | `q1_delivery_scores.npz` |
| `python code/scripts/fig_q1_radar.py` | `fig_q1_radar.pdf`：代表性啤酒雷达图（预测、真值与外部对照杯等级） | `q1_delivery_scores.npz`、`q1_external.npz` |
| `python code/scripts/fig_q1_m_sensitivity.py` | `fig_q1_m_sensitivity.pdf`：尺度参数 m 的敏感性 | `q1_mscale.npz` |

### 6.4 主要结果（供核对）

| 项目 | 数值 |
|---|---|
| 等级 MAE / 完全一致率 / QWK（中位数规则） | 0.479 / 0.581 / 0.290 |
| 连续 MAE / R² / Spearman | 0.473 / 0.184 / 0.360 |
| 90% 等级区间覆盖率 / 平均宽度 | 0.972 / 1.97 |

## 7. 问题 2：化合物–风味分析

### 7.1 运行

```bash
python code/src/q2/run_all.py
```

`run_all.py` 按下表顺序运行 15 个步骤，检查每步返回码，任一步失败即停止下游；控制台输出写入 `run_log.txt`，输入与全部产物的 SHA-256 写入 `run_manifest.json`。
`--start STEP`（如 `--start stability.py`）从指定步骤续跑。
全量运行约十几分钟，M3 对照最耗时。

### 7.2 模块与步骤（`code/src/q2/`）

| 顺序 | 文件 | 功能 | 主要输出（`code/data/processed/q2/`） |
|---|---|---|---|
| — | `config.py` | 路径、种子与全部参数（τ=0.6，置换、bootstrap、模拟各 1000 次） | — |
| — | `core.py`、`labels.py` | 秩与偏相关、模块、前向选择、置换门槛、bootstrap、敏感性方案框架；附件 8 标签 | — |
| 1 | `data.py` | 读取附件 1/2/4 与问题 1 产物，对齐 250 款啤酒，确定 225 个候选与 6 个注释列，输入口径断言 | — |
| 2 | `graph.py` | 共表达图（\|ρ\| ≥ 0.6 连边）与 τ=0.6 模块 | `graph_nodes.csv`、`graph_modules.csv`、`graph_edges.csv` |
| 3 | `association.py` | 全样本图约束 Top-3 | `top3_results.csv`、`q2_selection.npz` |
| 4 | `inference.py` | 置换门槛 | `top3_results_with_sig.csv`、`q2_permutation.npz` |
| 5 | `stability.py` | 按日期组 bootstrap（每次重建图与模块）、π^c、π^m、Jaccard、维度等级 | `final_top3_with_stability.csv`、`q2_bootstrap_main.npz` |
| 6 | `rationality.py` | 附件 8 标签、OAV、进入维度数、风格内相关、共变标记 | `final_top3_with_rationality.csv` |
| 7 | `controls_sensitivity.py` | 对照 M0（无图约束）、M1（图约束边际选择） | `controls_sensitivity_results.csv` |
| 8 | `full_controls.py` | 对照 M2（ElasticNet）、M3（梯度提升置换重要度，复用问题 1 划分与参数）、风格内选择 | `full_controls_results.csv`、`m3_fold_records.csv`、`q2_m3_importance.npz` |
| 9 | `simulation.py` | 模拟还原（6 种设置 × 1000 次） | `simulation_recovery_results.csv` |
| 10 | `sensitivity_runs.py` | 9 个敏感性方案与逐对删除诊断 | `sensitivity/` |
| 11 | `check_attachment6.py` | 附件 2 与附件 6 复算（误差 ≤ 1e-12） | — |
| 12 | `advanced_audits.py` | 复算验收；C 级维度零膨胀检查 | `zero_inflation_c_grade.csv` |
| 13 | `extra_figures.py` | 网络图、方法对比、π^c 分布、模块大小 | `figures/q2/*.png` |
| 14 | `pipeline_final.py` | Top-1 热力图 | `figures/q2/top1_heatmap.png` |
| 15 | `acceptance.py` | 结构化最终验收；附录 147 行视图 | `acceptance_log.csv`、`appendix_main_147.csv`、`handoff_modeling.md` |

敏感性方案：`tau_0.5`、`tau_0.7`、`cand_plus_composites`、`cand_plus_internal_std`、`na_lowest`、`attachment8_conflict`、`boot_fixed_graph`、`boot_beer_unit`、`sensory_z`。
`sensitivity_runs.py` 支持 `--only NAME ...` 与 `--force`；已完成的方案默认跳过。

### 7.3 论文图表（`code/scripts/`，在 `run_all.py` 之后运行）

技术路线图 `figures/q2_methodology_flowchart.pdf` 由 `fig_q2_methodology_flowchart.py` 从 TikZ 源文件编译，不在本包内。

| 命令 | 输出（`figures/q2/`） | 数据来源 |
|---|---|---|
| `python code/scripts/solve_q2_layout.py` | `code/data/processed/q2/q2_layout.npz`：共表达图 MDS 布局（距离 1−\|ρ\|，种子 0） | 附件 2、`graph_nodes.csv` |
| `python code/scripts/solve_q2_spike.py` | `code/data/processed/q2/q2_spike.csv`：加标杯与对照杯的同日同品评员配对差及 t 值 | Zenodo 工作簿（md5 校验） |
| `python code/scripts/fig_q2_recovery.py` | `fig_q2_recovery.pdf`：前向选择与边际选择的找回率对比 | `simulation_recovery_results.csv` |
| `python code/scripts/fig_q2_network.py` | `fig_q2_network.pdf`：τ=0.6 共表达图（颜色示来源、圈示模块、点大小示模块内连接度） | `q2_layout.npz`、图三表 |
| `python code/scripts/fig_q2_stability.py` | `fig_q2_stability.pdf`：49 维 × 3 位置 π^c 热力图与三档分布 | `final_top3_with_rationality.csv` |
| `python code/scripts/fig_q2_jaccard.py` | `fig_q2_jaccard.pdf`：方法间 Jaccard（随机基线按候选数解析计算） | `full_controls_results.csv` |
| `python code/scripts/fig_q2_oav.py` | `fig_q2_oav.pdf`：Top-3 中有阈值化合物的 OAV≥1 啤酒比例 | `final_top3_with_rationality.csv` |
| `python code/scripts/tab_q2_recovery.py` | `tab_q2_recovery.{csv,tex}`：植入–还原找回率（按植入顺序） | `simulation_recovery_results.csv` |
| `python code/scripts/tab_q2_jaccard.py` | `tab_q2_jaccard.{csv,tex}`：主方法与对照的 Jaccard（49 维平均） | `full_controls_results.csv` |
| `python code/scripts/tab_q2_spike.py` | `tab_q2_spike.{csv,tex}`：加标杯 − 对照杯配对差（括号内为 t 值） | `q2_spike.csv` |
| `python code/scripts/tab_q2_top3.py` | `tab_q2_top3.{csv,tex}`：全部 49 维 Top-3 完整结果（longtable） | `final_top3_with_rationality.csv` |
| `python code/scripts/tab_q2_sensitivity.py` | `tab_q2_sensitivity.{csv,tex}`：敏感性运行汇总 | `sensitivity/sensitivity_summary.csv` |

`solve_q2_layout.py` 须在 `fig_q2_network.py` 之前运行，`solve_q2_spike.py` 须在 `tab_q2_spike.py` 之前运行；其余脚本相互独立。

### 7.4 主要结果（供核对）

49 个交付维度（barnyard 与 stale_hops 评分近似重复，barnyard 不单独建模）的 Top-1 等级为 A 级 26 维、B 级 22 维、C 级 1 维。
Top-1/2/3 的平均 bootstrap 入选频率 π^c 为 0.60 / 0.36 / 0.22。
附件 2、附件 6 的复算最大误差分别为 1.05e-15、6.66e-16。

## 8. 问题 3：风味与风味指纹生成

### 8.1 模块（`code/src/q3/`）

| 文件 | 功能 |
|---|---|
| `paths.py`、`config.py`、`meta.py` | 路径、全部参数（网格、样本量、重抽次数、源与目标风格、9 个化学缺陷列与 6 个感官缺陷维度）、运行元信息 |
| `data.py` | 读取附件 1、化学字典、阈值与缺陷表；NA 与附件 8 变体；读取问题 2 Top-3 |
| `marg.py` | 逐列秩变换到潜在正态空间及逆变换（并列块与检测下限按原子处理） |
| `modules.py` | 读取问题 2 的 τ=0.6 模块 |
| `composite.py` | 5 个综合指标由潜变量岭回归派生 |
| `generator.py` | 风格条件生成器 `StyleGen`（风格均值收缩 κ、k 因子协方差、模块分块特异协方差） |
| `controls.py` | 对照生成器 G1（复制）、G2（逐列独立）、G3（风格内插值） |
| `selection.py` | 96 个设定（84 个参与选择）、嵌套划分、内层留出高斯评分与选参 |
| `gen_metrics.py`、`genval.py` | C2ST 与置换零分布、DCR、TRTR/TSTR、下限比例、相关误差、模块内 ρ、离散比、验收判定 |
| `fullmodel.py` | 全数据生成器与派生器 |
| `q1_bridge.py` | 接入问题 1 模型：预测 μ̂、等级分布、折内模型（画像验证） |
| `portrait.py`、`portrait_val.py` | 感官画像、化学指纹 δ 与感官指纹；画像验证 |
| `defects.py`、`transfer.py`、`source_input.py`、`sens_search.py` | 缺陷约束、潜空间平移与马氏投影、输入组装、感官路径搜索与 α* / α_fit |
| `evaluate.py` | 交叉拟合风格判别器 P(t)、d²、保留度、改动量、结论类别 |
| `sensitivity.py` | 敏感性臂定义与配对比较 |
| `plotting.py`、`tables.py` | 绘图与 TeX 表 |

### 8.2 求解脚本（`code/scripts/`）

| 步骤 | 命令 | 依赖 | 输出（`code/data/processed/q3/`） | 参考耗时 |
|---|---|---|---|---|
| 1 | `python code/scripts/solve_q3_data.py` | 附件 1、审计字典、问题 2 模块 | `q3_data.npz`、`q3_checks.json`、`q3_defect_table.csv` | 数秒 |
| 2 | `python code/scripts/solve_q3_select.py --workers 8` | 步骤 1 | `q3_select.npz`/`.json`（`final_config`） | 约 1 分钟 |
| 3 | `python code/scripts/solve_q3_genval.py --workers 8` | 步骤 2 | `q3_genval.npz`/`.json`：生成验证指标、bootstrap 区间、验收 (a)–(e) | 约 4 小时 |
| 4 | `python code/scripts/solve_q3_transfer_chem.py` | 步骤 2 | `q3_transfer_chem.npz`/`.json`：化学层约束与 462 个风格对统计 | 数秒 |
| 5 | `python code/scripts/solve_q3_portrait.py` | 步骤 2、问题 1 | `q3_portraits.*`、`q3_fingerprint.csv`、`q3_sensory_fingerprint.csv`、`q3_synthetic_profiles.npz` | 约 1 分钟 |
| 6 | `python code/scripts/solve_q3_portrait_val.py --workers 6` | 问题 1 | `q3_portrait_val.*` | 约 4 分钟 |
| 7 | `python code/scripts/solve_q3_transfer.py --workers 8` | 步骤 4、问题 1、问题 2 | `q3_transfer.*`、`q3_transfer_pils_tripel.csv`、`q3_transfer_defects.csv`、`q3_transfer_pairs.csv` 等 | 约 2 分钟 |
| 8 | `python code/scripts/solve_q3_sensitivity.py --arm all --workers 5` | 步骤 3、5、7 | `sens/<臂>.json`/`.npz`、`q3_sensitivity.csv` | 约 1 小时 |
| 9 | `python code/scripts/export_q3_outputs.py` | 步骤 5 | `q3_portraits_long.csv` | 数秒 |
| 10 | `python code/scripts/solve_q3_pca.py` | 步骤 2、4、5、7 | `q3_pca.npz`/`.json`：潜空间 PCA（只在真实啤酒上拟合）下的真实、生成样本与迁移轨迹投影 | 数秒 |

步骤 10 以相同种子重抽生成样本的潜变量，并核对其反变换与 `q3_synthetic_profiles.npz` 一致、轨迹终点 α* 与 `q3_transfer.npz` 一致，任一不一致时以非零退出码结束。

步骤 3 可用 `--stage gen|metrics|merge` 分阶段运行。
步骤 8 的 `--arm` 也接受逗号分隔的臂名，`--list` 列出全部臂，`--summary` 只重写汇总表；迁移臂在内部以子进程调用步骤 4、7 的脚本，变体产物带后缀（如 `q3_transfer_strict.json`）。
敏感性臂：生成器 G-S1–G-S7（协方差结构、相邻网格点、风格专属方差、并列块得分、NA 编码、附件 8）、画像 G-S8–G-S9（问题 1 删除 `_sum` 列模型、内标抽样）、迁移 T-S1–T-S7（综合指标派生、候选选择、阈值口径、化学层规则、感官层规则、问题 1 等级映射、改用 G-S5/6/7 生成器）。

### 8.3 图表脚本

| 命令 | 输出（`figures/q3/`） |
|---|---|
| `python code/scripts/fig_q3_select_heatmap.py` | 内层评分热图与外层选定频数 |
| `python code/scripts/fig_q3_c2st.py` | 合并与条件 C2ST |
| `python code/scripts/fig_q3_gen_metrics.py` | DCR 比、近复制率、TSTR 对 TRTR、下限比例误差 |
| `python code/scripts/fig_q3_corr.py` | 相关误差、风格内相关误差、模块内 ρ |
| `python code/scripts/fig_q3_portrait.py` | Tripel、Kriek、Lambic 感官画像 |
| `python code/scripts/fig_q3_fingerprint.py` | 三个风格的化学指纹前 10 项 |
| `python code/scripts/fig_q3_portrait_val.py` | 画像验证：逐风格 Spearman 对样本量 |
| `python code/scripts/fig_q3_transfer_alpha.py` | Pils/Lager → Tripel 的可行路径与 α* |
| `python code/scripts/fig_q3_transfer_pt.py` | 各迁移方法的 P(Tripel) |
| `python code/scripts/fig_q3_pairs.py` | 22 × 22 风格对热图 |
| `python code/scripts/fig_q3_sensitivity.py` | 敏感性臂相对基线的变化 |
| `python code/scripts/tab_q3_{select,genval,transfer,pairs,sensitivity}.py` | 同名 `.csv` 与 `.tex` |

论文正文图（依赖步骤 3、5、7、10；技术路线图 `figures/q3_methodology_flowchart.pdf` 由 TikZ 源文件编译，不在本包内）：

| 命令 | 输出（`figures/q3/`） | 数据来源 |
|---|---|---|
| `python code/scripts/fig_q3_gen_eval.py` | `fig_q3_gen_eval.pdf`：生成器 C2ST AUC 与潜空间 PCA 双子图 | `q3_genval.json`、`q3_pca.npz` |
| `python code/scripts/fig_q3_radar.py` | `fig_q3_radar.pdf`：Tripel、Kriek、Lambic 画像雷达图（中位数、10%–90% 带、真实折外预测） | `q3_portraits.npz`、问题 1 折外预测 |
| `python code/scripts/fig_q3_fingerprint_pair.py` | `fig_q3_fingerprint_pair.pdf`：化学指纹前 10 项与感官指纹 | `q3_fingerprint.csv`、`q3_sensory_fingerprint.csv` |
| `python code/scripts/fig_q3_transfer.py` | `fig_q3_transfer.pdf`：迁移轨迹、逐款 α* 与目标风格概率、代表款画像三子图 | `q3_pca.npz`、`q3_transfer_pils_tripel.csv`、`q3_transfer_portraits.csv` |
| `python code/scripts/fig_q3_defect.py` | `fig_q3_defect.pdf`：缺陷不恶化核查表（化学层 OAV 变化与感官层等级） | `q3_transfer_defects.csv` |

### 8.4 主要结果（供核对）

| 项目 | 数值 |
|---|---|
| 全数据选定设定 | 模块分块因子协方差，k = 15，κ = 5 |
| 合并 / 条件 C2ST AUC（G0 不分风格对照） | 0.589 / 0.576（0.642） |
| DCR 比 / TSTR / TRTR | 1.111 / 0.481 / 0.550 |
| Pils/Lager → Tripel（11 款） | 7 款达到 Tripel 参照范围，2 款改善，2 款未改善；化学层与感官层缺陷违约均为 0 |

## 9. 测试

```bash
cd code
python -m pytest tests -q
```

共 145 项。`test_q1_outputs.py`、`test_q1_deliverables.py` 检查问题 1 正式产物，须在问题 1 全部运行后执行；其余为不依赖正式产物的单元测试。

## 10. 复现注意事项

1. **固定版本**：scikit-learn 必须为 1.7.2。外层划分与逐折超参数依赖该版本的 `StratifiedGroupKFold` 打乱结果；问题 2、3 通过 `q1_main_oof.npz` 中保存的 `fold_id` 复用划分。问题 2 的 M2 对照（`ElasticNetCV`）的选定正则强度也随 scikit-learn 版本变化；在其他版本下运行时，主方法的 Top-3、显著性、稳定性与敏感性结果不变，M2 的 Top-3 会有差异。
2. **随机种子**：全局 `SEED=0`。问题 1 外层第 r 次重复用 `SEED+r`，内层 `SEED+100+r`，全数据重拟合内层 `SEED+200`；树模型 `random_state=0`；bootstrap 用 `numpy.random.default_rng(0)`。问题 2、3 的种子在各自 `config.py` 中设定。相同环境下重复运行，结果逐位一致。
3. **断点续跑**：耗时脚本按计算单元写入 `code/data/processed/q*/_chunks/`，已存在的单元直接跳过。修改代码或参数后，须先删除对应的 `_chunks` 子目录再运行，否则会复用旧结果。
4. **冒烟模式**：问题 1、3 的求解脚本与问题 2 的 `full_controls.py`、`simulation.py`、`sensitivity_runs.py` 接受 `--smoke`，只运行少量折与重抽，结果写入 `_smoke/`，用于检查流程与结构，数值不作为结果。冒烟链与正式链相互独立，但须按第 5 节相同的顺序运行；问题 3 的冒烟模式读取正式的 `q3_select.json` 与问题 1 正式模型。问题 1 冒烟仍要完成 2 个外层折的完整选参，全链约 1.5 小时（主模型约 40 分钟、对照约 27 分钟、两阶段约 23 分钟）；问题 2、3 的冒烟各在数分钟内完成。
5. **模型文件**：`q1_models/`、`q1_models_drop_sum/` 是 joblib 格式的全数据模型，由问题 1 步骤 2、6a 生成，问题 3 依赖它们。
6. **计算量**：完整复现的主要耗时为问题 1 对照（约 1 天 CPU 时间）、问题 1 主模型与两阶段对照（各约 2 小时）、问题 3 生成验证（约 4 小时）；其余步骤在数分钟内完成。

## 11. 在同格式新数据上测试泛化

`code/scripts/predict_q1_new.py` 用问题 1 的全数据模型（`q1_models/`）对新啤酒评分，不重新训练。
新化学表须包含附件 1 的 231 个化学列（列名相同，NA 允许）；`beer` 列存在时作为编号输出。
若同时提供观测评分（列名同原始评分文件：`beer_name` 为该款在化学表中的 1 起行号，另有 50 个维度的 0–6 分），脚本按与正式评价相同的口径计算指标。

```bash
python code/scripts/predict_q1_new.py --chem path/to/new_chem.csv --ratings path/to/new_ratings.csv --out path/to/out
```

| 输出 | 内容 |
|---|---|
| `q1_new_scores.csv` | 宽表：每款 50 维中位数等级 `G_hat` |
| `q1_new_scores_long.csv` | 长表：连续预测 `mu_hat`、`G_hat`、分位数切点等级 `G_hat_Q`、等级分布 `p0`–`p6`、90% 区间 `lo`/`hi`；给出评分时另含 `xbar_obs`、`G_obs` |
| `q1_new_report.json` | 样本数与运行环境；给出评分时另含等级 MAE、完全一致率、±1 以内、平均 QWK、连续 MAE、R²、Spearman、区间覆盖率与宽度 |

前提为问题 1 步骤 2、3 已完成。
对本数据集的 250 款运行时，这些啤酒都参与了全数据模型训练，指标是样本内结果（等级 MAE 0.275），不代表泛化性能；第 6.4 节的嵌套交叉验证折外结果才是本数据上的泛化估计。

整套流程（第 5 节）的求解脚本含针对本数据集的断言（样本数 250、231 个化学列、22 个风格、附件 2 为 231 × 231 等），用于防止输入错配。
样本数或风格组成不同的数据在这些断言处中止，因此在新数据上重训问题 1–3 需相应修改这些常量；问题 3 的源与目标风格、缺陷列在 `code/src/q3/config.py` 中设定。
