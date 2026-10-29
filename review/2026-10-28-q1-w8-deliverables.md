---
id: 2026-10-28-q1-w8-deliverables
from: modeling
to: coding
status: ack
severity: major
refs:
  - model/q1_methodology.md
  - review/2026-10-28-q1-w1-data.md
  - review/2026-10-28-q1-w3-scores.md
  - review/2026-10-28-q1-w6-metrics.md
  - review/2026-10-28-q1-w7-sensitivity.md
  - code/src/q1/report_beers.py
  - code/scripts/export_q1_scores.py
---

## 问题

- 生成 Q1 的全部图、表和交付 CSV，供论文直接引用。
- 旧实现的错误：图由求解脚本顺带画出，规则散点图、敏感性图、参照误差图用的是写死的占位坐标（`fix_q1_t7_final.py:51-67`）；`fig_q1_reliability` 画的是 x̄ 对 G_hat，不是折外 R² 对信度；`tab_q1_metrics.csv` 没有生成脚本，数字与 `q1_metrics.json` 对不上；宽表缺 `beer_row`，长表缺 `barnyard_flag`；删除 `figures/q1/` 后无法重新生成。W1 任务 0 已把旧图表移入 `figures/q1/legacy_z/`。
- 本工单不求解、不拟合、不计算指标：所有数从 W1、W3、W6、W7 的产物读取。
- 依赖：W1 的 `q1_data.npz`；W2 的 `q1_main_oof.npz`；W3 的 `q1_scores.npz`；W4 的 `q1_baselines.npz`（`b2_r2` 与 `chemstyle_final_*`）；W6 的 `q1_metrics.json`、`q1_metrics.npz`；W7 的 `q1_sensitivity.json`、`q1_stress.json`、`q1_external.json`（含 `model_key`）。
- 只创建或修改：`code/src/q1/plotstyle.py`、`code/src/q1/tables.py`、`code/src/q1/export.py`、`code/src/q1/report_beers.py`；`code/scripts/` 下 `fig_q1_rules.py`、`fig_q1_confusion.py`、`fig_q1_paired_diff.py`、`fig_q1_reliability.py`、`fig_q1_reference_error.py`、`fig_q1_sensitivity.py`、`fig_q1_stress.py`、`tab_q1_metrics.py`、`tab_q1_rules.py`、`tab_q1_paired.py`、`tab_q1_external.py`、`tab_q1_report_beers.py`、`export_q1_scores.py`；`code/tests/test_q1_deliverables.py`、`code/tests/test_q1_outputs.py`。另在 `paper/main.tex` 导言区加入缺少的 `longtable` 与 `pdflscape` 宏包，使生成的分页横向附录表可直接编译；不改正文内容。
- 运行环境与通用约定见本工单末尾「通用约定」。

## 共同规则

1. 绘图与表脚本的数值及啤酒元数据只从 `code/data/processed/q1/` 产物读取；唯一原始数据例外是 `report_beers.py` 为确定多日期名单而调用 `load_raw_ratings()`。允许导入：`src.q1.paths`、`src.q1.plotstyle`、`src.q1.tables`、`src.q1.export`、`src.q1.report_beers`，以及 `src.q1.data` 中的常量 `DIMS`、`DIM_GROUPS`、`ZERO_HEAVY_DIMS`、`load_raw_ratings`。不得导入 `model`、`baselines`、`ordinal`、`twostage`、`discretize`、`sensitivity`、`external`、`metrics`、`oav`。用 `ast` 在测试中检查。
2. 图和表里的每个数都来自文件；脚本中不写任何结果数值、阈值判定或状态字符串。
3. 输入文件缺失时 `FileNotFoundError` 退出，不画空图、不写空表。
4. 每个脚本接受 `--src DIR`（默认 `PROCESSED_DIR`）和 `--out DIR`；`fig_*`、`tab_*` 的默认 `--out` 为 `FIG_DIR`，`export_q1_scores.py` 的默认 `--out` 为 `PROCESSED_DIR`。导出脚本把 `q1_delivery_scores.npz/.json`、`q1_final_scores.csv`、`q1_scores_long.csv` 写在同一个 `--out` 目录；表脚本从 `--src/q1_scores_long.csv` 等正式产物读取，故默认执行时先运行导出脚本，再运行全部表脚本。测试用临时目录运行。脚本名与产物名一致：`fig_q1_rules.py` 写 `fig_q1_rules.pdf`，`tab_q1_metrics.py` 写 `tab_q1_metrics.csv` 与 `tab_q1_metrics.tex`。
5. 图只写 PDF（矢量，`pdf.fonttype=42`），论文引用 PDF。表同时写 CSV 与 tex。
6. 文字为中文。图内标注、坐标轴标签、图例都用中文；数学符号用 mathtext，例如 `$\\hat{G}$`。

## 任务 1：公共模块

`plotstyle.py`（导入时不得建目录或写文件）：
- `apply() -> None`：设置 `font.family=sans-serif`，`font.sans-serif=['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']`，`axes.unicode_minus=False`，`pdf.fonttype=42`，`font.size=9`，`axes.spines.top=False`，`axes.spines.right=False`，`axes.grid=True`，`grid.alpha=0.25`，`savefig.bbox='tight'`。
- `PALETTE`：字典，色盲友好的 Okabe–Ito 色：`main='#0072B2'`、`ctrl='#999999'`、`worse='#0072B2'`、`better='#D55E00'`、`neutral='#999999'`、`accent='#009E73'`、`warn='#E69F00'`。主输出始终用 `main`，对照用 `ctrl`，不用彩虹色。
- `save(fig, out_dir, name) -> Path`：`out_dir.mkdir(parents=True, exist_ok=True)` 后写 `out_dir / name`，名称必须以 `.pdf` 结尾，返回路径。
- `forest(ax, labels, point, lo, hi, colors) -> None`：横向森林图，自上而下按给定顺序，点加 CI 横线，`x=0` 处画虚线。
- `significance_color(lo, hi) -> str`：`lo > 0` 取 `PALETTE['worse']`，`hi < 0` 取 `PALETTE['better']`，否则 `neutral`；函数只依赖 CI 与 0 的位置，图注说明含义（对照减主输出为正表示对照更差）。

`tables.py`：
- `latex_table(df, colfmt, caption, label, float_fmt='{:.3f}', longtable=False, bold_mask=None) -> str`：手写 booktabs 表（不用 `DataFrame.to_latex`，避免依赖 jinja2）；转义 `_`、`%`、`&`、`#`；NaN 与 None 写 `--`；`bold_mask` 为与 `df` 同形状的布尔数组，真值单元加粗；`longtable=True` 时用 longtable 环境并重复表头。
- `write_table(df, out_dir, stem, caption, label, **kw) -> None`：写 `stem.csv`（`utf-8-sig`，NaN 写空串）与 `stem.tex`。

`report_beers.py`：
- `REPORT_ROWS: tuple[int, ...]`：12 个附件行号（1–250），升序；由数据计算出后写成字面量（见测试）。
- `multi_date_rows(ratings) -> list[int]`：`ratings` 为 `load_raw_ratings()` 的结果；返回在多于一个日期品评的啤酒的 `row`，升序。
- `REPORT_ROWS` 必须等于 `multi_date_rows(load_raw_ratings())`；按附件行号升序生成，不按预测结果筛选，不接受手工名单。

`export.py`（纯函数，不读写文件）：
- `build_wide(beer_row, beer, G_hat, dims) -> DataFrame`：列 `beer_row`、`beer`、50 个维度，值为 `G_hat`（int）。
- `build_long(beer_row, beer, style, dims, G_hat, G_hat_Q, p_final, lo, hi, G_obs, xbar, mu_hat) -> DataFrame`：每行一款啤酒 × 一个维度，共 12,500 行，先按 `beer_row` 再按 `dims` 顺序。列顺序：`beer_row`、`beer`、`style`、`dim`、`G_hat`、`G_hat_Q`、`p0`…`p6`、`lo`、`hi`、`G_obs`、`xbar_obs`、`mu_hat`、`barnyard_flag`。`barnyard_flag` 在 `dim == 'barnyard'` 的行写 `疑似 stale_hops 副本`，其余行为空串（方法论 §2.3）。

## 任务 2：交付 CSV `export_q1_scores.py`（重写）

- 读 `q1_metrics.json` 的 `success` 并据方法 §5.2 选择交付来源：True 时用 W3 `q1_scores.npz` 和 W2 `q1_main_oof.npz` 的 `oof_mu`（5 次平均）；False 时用 W4 `q1_baselines.npz` 的 `chemstyle_final_*`。`q1_scores.npz` 始终保留化学主模型结果供评估与 Q3 使用，不因交付回退而覆盖。`beer_row`、`beer`、`style`、`G`、`xbar`、`dims`、`n_ratings` 全部读 W1 `q1_data.npz`；断言 `beer_row` 为 1..250，且所有数组首轴与它逐行对齐。`beer` 为附件 `beer` 列取值，二者同时给出，避免混淆两种编号（方法论 §1.2）。
- 先写 `PROCESSED_DIR/q1_delivery_scores.npz` 与 `.json`：NPZ 键为 `model_key`（`main` 或 `chemstyle`）、`G_hat`、`G_hat_Q`、`p_final`、`lo`、`hi`、`mu_hat`、`seed`、`meta`；JSON 写 `model_key`、选择依据（`success`）、输入产物摘要。缺少 `success` 或所选来源字段时以 `FileNotFoundError`/`KeyError` 退出，不默认为主模型。
- `q1_final_scores.csv`（宽表，250 行 × 52 列）和 `q1_scores_long.csv`（长表，12,500 行）均从 `q1_delivery_scores.npz` 写出，编码 `utf-8-sig`、`index=False`。论文表注读取 `model_key` 并说明本次交付所用模型；主模型对照指标仍按 `q1_metrics.json` 报告。
- 宽表中的值为选定交付来源的 `G_hat`（预测分布中位数），第二输出 `G_hat_Q` 只在长表中，列名已标明。

## 任务 3：图（每个脚本一张图，尺寸为单栏 3.4 × 2.6 英寸或整页宽 7.0 × 2.8 英寸，各图自行选定）

| 脚本 | 产物 | 内容与数据来源 |
|---|---|---|
| `fig_q1_rules.py` | `fig_q1_rules.pdf` | 横轴等级 MAE，纵轴 QWK，6 个离散规则各一点（`q1_metrics.json` 的 `models` 中 `main.const`、`main.round`、`main.median`、`main.quant`、`main.qwk_cut`、`main.median_knn`）；`main.median` 用主色并加粗标注「主输出」，其余灰色；标注规则名；图注说明左上更好。不画 CI。 |
| `fig_q1_confusion.py` | `fig_q1_confusion.pdf` | 3 个并排 7×7 混淆矩阵：`main.median`、`k2.median`、`c1.sel`（`q1_metrics.npz` 的 `conf_*`）；颜色为行归一化比例（单色蓝，0–1，3 个子图共用色标），格内写计数（5 次重复平均，取整），深色格用白字；行为真实等级、列为预测等级，子图标题写模型名。 |
| `fig_q1_paired_diff.py` | `fig_q1_paired_diff.pdf` | 两个子图共用纵轴：9 个「对照减主模型」的配对差（`q1_metrics.json` 的 `paired` 中 `b0`、`b1`、`lin10`、`enet`、`rf`、`chemstyle`、`k2`、`c1.sel`、`c2` 减 `main`），左为等级 MAE，右为 QWK；点为点估计，线为 95% CI；按 MAE 点估计排序；颜色由 `significance_color` 决定。 |
| `fig_q1_reliability.py` | `fig_q1_reliability.pdf` | 50 个点：横轴信度 ρ_d，纵轴折外 R²（`q1_metrics.json` 的 `reliability`、`r2_main`），画 y=x 线；按 `DIM_GROUPS` 的 6 组用不同标记形状，零分为主的 22 个维度用空心点；标注信度最低的 4 个维度和 `leak_dims` 中的所有维度。 |
| `fig_q1_reference_error.py` | `fig_q1_reference_error.pdf` | C1 的混合参照 R 对真值 G 的误差（`q1_metrics.json` 的 `reference_error`）：左子图为 `all`、`A`、`L` 三类单元的完全一致率与 ±1 以内比例（分组柱），右子图为平均偏差与等级 MAE；每类标注单元数。 |
| `fig_q1_sensitivity.py` | `fig_q1_sensitivity.pdf` | 森林图：W7 的各臂相对主设定的等级 MAE 配对差与 95% CI（左）和 QWK 配对差（右）；`alpha` 臂用强调色；右侧文字标注 `changed_delivered`（交付等级变化比例）。数据 `q1_sensitivity.json` 的 `arms`。 |
| `fig_q1_stress.py` | `fig_q1_stress.pdf` | 留一风格：22 个风格的等级 MAE 点图（主模型），叠加 b0、b1 的标记；`small`（款数 ≤ 5）的风格用空心点并在图注说明只作定性解读；按主模型 MAE 排序；数据 `q1_stress.json`。 |

## 任务 4：表

| 脚本 | 产物 | 内容 |
|---|---|---|
| `tab_q1_metrics.py` | `tab_q1_metrics.csv`、`.tex` | 行：`main`、`b0`、`b1`、`lin10`、`enet`、`rf`、`chemstyle`、`k2`、`c1.sel`、`c2`，规则取预测分布中位数（`c1.sel` 为自身选定映射）；列：等级 MAE、完全一致率、±1 以内、QWK、平衡准确率、RPS、对数损失、连续 MAE、R²、Spearman（`q1_metrics.json` 的 `models`）。不适用的单元写 `--`。另加一行 `B2（对 z，75 款测试）`，只填 R² 列，值取 `q1_baselines.npz` 的 `b2_r2` 的 50 维平均；最后一行为气候学基线的 RPS 与对数损失。 |
| `tab_q1_rules.py` | `tab_q1_rules.csv`、`.tex` | 方法论 §4.3 的规则比较：行为 6 个规则与真值；列为等级 MAE、完全一致、±1 以内、QWK、平衡准确率、5–6 级占比；真值行只填 5–6 级占比（`grade_freq`）。分 22 个零分为主维度与其余 28 维两个子表（`subset_report`）。 |
| `tab_q1_paired.py` | `tab_q1_paired.csv`、`.tex` | `paired` 中全部 18 对：`a`、`b`，MAE、完全一致、QWK 的点估计与 95% CI（格式 `点估计 [lo, hi]`），`wins_a`。 |
| `tab_q1_external.py` | `tab_q1_external.csv`、`.tex` | W7 的外部检查：两款啤酒各 11 行：`ctrl_mean`、`ctrl_se`、`ctrl_grade`、`G_hat`、`G_obs`；表尾为汇总（`G_hat` 对 `ctrl_grade` 的 MAE、完全一致数、平均有符号误差；`G_obs` 对 `ctrl_grade` 的同样几项）；表注读取 `model_key`，说明使用的交付模型、仅两款啤酒、啤酒 223 只有 13 个评分，结果只报告。 |
| `tab_q1_report_beers.py` | `tab_q1_report_beers.csv`、`.tex` | 从 `--src/q1_scores_long.csv` 按 `REPORT_ROWS` 与 `dims` 取 600 行，CSV 列顺序为 `beer_row`、`beer`、`style`、`n_ratings`、`dim`、`G_hat`、`lo`、`hi`、`G_obs`、`score_cell`；`n_ratings` 按 `beer_row` 查 W1 产物。`score_cell` 精确写为整数文本 `G_hat[lo,hi]/G_obs`，预测与实测不同只在 TeX 中加粗，不在 CSV 中加入 TeX 命令。TeX 为附录用转置表：维度作行、12 款啤酒作列，按 `DIM_GROUPS` 的定义顺序拆成六张表；每组的维度保持 `DIMS` 相对顺序。列头用附件 `beer` 编号，每组表前给出 `beer_row`、`beer`、`style`、`n_ratings` 的编号对照。表注解释单元为预测中位数及 90% 等级区间/实测等级、加粗含义，以及从 `--src/q1_delivery_scores.json` 读取的 `model_key` 来源；用 `landscape`、`\scriptsize`、重复表头的 `longtable`。全部 250 款的 12,500 行明细只作为机器可读补充 CSV，不排入论文。 |

## 任务 5：测试

`code/tests/test_q1_deliverables.py`（不读真实产物，用人造数据）：
- `latex_table`：特殊字符转义、NaN 写 `--`、`bold_mask` 加粗位置、longtable 开关。
- `build_wide`：形状、列名、`beer_row` 与 `beer` 同时存在且不相等（人造 `beer` 与行号错开）。
- `build_long`：2 款 × 3 维的人造数组上，行数 6、列顺序与任务 1 一致、`p0`…`p6` 与输入逐元素相同、`barnyard_flag` 只在 barnyard 行有值。
- 报告表：六组维度的行集合互不相交且并集等于 `DIMS`；TeX 每张表按维度为行、啤酒编号为列，编号映射覆盖 12 款且 `n_ratings` 与 W1 产物一致；单元格编码、预测不一致的加粗位置、表注来源和重复表头正确；CSV 恰有 600 行、列顺序与任务 4 一致且 `(beer_row, dim)` 唯一。
- 表格排版冒烟：用人造数据生成六组 TeX 表，在临时 `article` 包装中加载 `ctex`、`booktabs`、`longtable`、`pdflscape` 后用 XeLaTeX 编译并渲染；逐页检查无裁切、重叠、空白页或缺字，且横向页方向正确。临时包装与渲染图不交付。
- `significance_color`：三种情形。
- AST 检查：每个 `fig_*`、`tab_*`、`export_*` 脚本只导入允许的模块（共同规则 1）；没有包含 `\\_\\_main\\_\\_` 以外的求解调用（即不调用 `fit`、`apply_rules`、`nested_cv_predict`）。
- 每个绘图脚本用 `--src` 指向 `tests/` 下一个由 pytest fixture 临时生成的小型人造产物目录运行一次，写入临时 `--out`，断言 PDF 文件存在、大小大于 0、以 `%PDF` 开头。人造产物的键与 W6、W7 的输出一致。

`code/tests/test_q1_outputs.py`（读真实产物，产物缺失即失败，不得跳过）：
- `q1_delivery_scores.npz`：`model_key` 与 `q1_metrics.json.success` 一致；若为 `main`，数组逐元素对应 W3/W2 主模型产物；若为 `chemstyle`，数组逐元素对应 W4 的 `chemstyle_final_*`。
- `q1_external.json` 与 `.npz` 的 `model_key` 与交付来源一致；`tab_q1_external.tex` 表注写出该来源。
- `q1_final_scores.csv`：形状 (250, 52)；`beer_row` 为 1..250；与 `q1_delivery_scores.npz` 的 `G_hat` 逐元素相同。
- `q1_scores_long.csv`：12,500 行；`p0`…`p6` 每行和为 1（1e-12）；`lo ≤ G_hat ≤ hi`；与宽表在对应单元一致；`xbar_obs` 取整后等于 `G_obs`。
- `REPORT_ROWS` 等于 `multi_date_rows(load_raw_ratings())`，长度 12。
- `tab_q1_metrics.csv` 中各单元与 `q1_metrics.json` 相应字段在 1e-9 内一致（防止手改）。
- `tab_q1_paired.csv` 的行数为 `len(paired)`。

## 验收

命令（仓库根目录）：
```
PY=/e/anaconda3/python.exe
$PY code/scripts/export_q1_scores.py
for f in code/scripts/fig_q1_*.py code/scripts/tab_q1_*.py; do case $f in *methodology_flowchart*) continue;; esac; $PY $f || exit 1; done
cd code && $PY -m pytest tests -q
```

再做一次再生成检查：删除 `figures/q1/` 下除 `legacy_z/` 以外的全部文件，只运行上述 `fig_*`、`tab_*`、`export_*` 脚本，确认全部文件重新出现，且 `git diff --stat` 显示 `code/src/q1/` 下的求解模块没有改动。

结构类检查（写入各脚本的终端输出，失败则非零退出；这里没有数值预期）：

| 名称 | 预期 |
|---|---|
| `figures/q1/` 下有 7 个 `fig_q1_*.pdf`、5 组 `tab_q1_*.csv/.tex`（`tab_q1_report_beers` 含在内） | 是 |
| 所有 PDF 以 `%PDF` 开头、大小大于 0 | 是 |
| 宽表 (250, 52)，长表 12,500 行 | 是 |
| `REPORT_ROWS` 与由数据算出的多日期品评名单相同 | 是 |
| `tab_q1_metrics.csv` 与 `q1_metrics.json` 一致 | 是 |
| 删除 `figures/q1/` 后只运行绘图与导出脚本即可全部重生成 | 是 |

回报中另列：每张图的路径与 SHA-256 前 12 位；`REPORT_ROWS` 的 12 个行号及其 `beer` 列取值；每个脚本的读入文件清单。

其余：Q1 范围 `black --check`、`ruff check` 通过；`code/src/q1/` 无 `__main__`；`figures/q1/` 下没有非本工单脚本产生的文件。

## 禁止事项

- 不得在绘图或表脚本中调用求解函数、读取原始评分以外的原始数据，或写入任何结果数值。
- 不得用占位坐标、随机数或示意数据画图；输入缺失时直接失败。
- 不得手改 CSV 或 tex；所有产物由脚本生成。
- 不得新建本工单未列出的脚本；需要改图时修改对应脚本。
- 不得把 `fig_q1_methodology_flowchart.py` 或 `figures/q1_methodology_flowchart.*` 改动（流程图的更新属于 modeling）。

## 通用约定

执行顺序：W1 → W2 → W3 → W4 与 W5（可并行）→ W6 → W7 → W8；W9（OAV，可选）在 W2 之后随时可做。每个工单单独领取、单独验收。

### 方法论中已定的决定

- OAV 类别名单见 `model/q1_methodology.md` §3.2；B2 的设定与网格见 §5.3；正文报告啤酒的选取规则见 §5.2。这些不在工单中另行决定，工单与方法论冲突时以方法论为准，并在回报中写明。
- 敏感性对照臂和 K2 沿用主运行在各外层折、各维度组选定的超参数，不单独做嵌套 CV。
- 5 次重复先平均 `p` 再取中位数；`G_hat_Q` 取 5 次分位数切点等级的中位数。

### 运行环境

- 正式运行用 `PY=/e/anaconda3/python.exe`：Python 3.13.9、scikit-learn 1.7.2、numpy 2.3.5、pandas 2.3.3，已装 pytest、black、ruff。`E:/anaconda3/envs/mathorcup`（scikit-learn 1.9.1）和 PATH 上默认的 `python`（3.14、scikit-learn 1.9.0、无 pytest）不用于正式运行。
- 每个求解脚本开头调用 `require_sklearn('1.7.2')`，版本不符即退出；`meta` 记录实际版本。
- `HistGradientBoostingRegressor` 在 `warm_start` 下，`max_iter` 取 100、200、400 与分别拟合的结果一致（本机核对，最大差 0.0）。一次拟合到 400 轮约 1 秒（单线程）。
- Q1 范围（black、ruff 只检查这些文件，不动 Q2、Q3 代码）：`code/src/q1/`、`code/scripts/solve_q1_*.py`、`code/scripts/export_q1_*.py`、`code/scripts/fig_q1_*.py`（不含 `fig_q1_methodology_flowchart.py`）、`code/scripts/tab_q1_*.py`、`code/tests/`。
- 不要求逐元素复现预运行。完整执行实验，各项数值与方法论中的预运行数字在工单给定容差内一致即可。

### 执行规则

1. 一次只领一个工单；领取时把 `status` 改为 `ack`。
2. 只创建或修改本工单列出的文件。不得新建 `final_*`、`fix_*`、`gen_*`、`patch_*`、`perfect_*` 一类补丁脚本；需要改动时修改原文件。
3. 代码中不得写入任何结果数值，包括指标、MAE、R²、状态字符串和 `checks` 的布尔值。输出的每个数由本次运行算出。`checks` 为列表，每项含 `name`、`observed`、`expected`、`tol`、`passed`，`passed` 由代码比较得出。
4. 预运行数值仅作回归核对。超出容差时写明观测值与预期值，不修改方法或容差；偏差超过容差 3 倍时自动开 `review/` 线程并附复现材料，但仅数值偏差不阻断当前任务或独立下游任务。只有方法预先定义的结构不变量失败、产物不完整或方法无法解释的实质偏离，才暂停依赖该产物的任务并在对应线程定位问题。
5. 不吞异常：不用 `try/except` 返回 0 或 NaN 代替失败；文件缺失即抛错退出。指标确实不适用时写 NaN，并在同一输出中写原因字符串。
6. `code/src/q1/` 不得有 `__main__`，导入时不得读写文件、建目录或绘图。目录由脚本调用 `paths.ensure_dirs()` 创建。
7. 求解脚本不绘图，绘图脚本不求解；绘图脚本只读 `code/data/processed/q1/`。
8. 先写 pytest 再写实现。测试放 `code/tests/`，秒级完成。
9. 每个求解脚本接受 `--smoke`：1 次重复、2 个外层折、bootstrap 20 次，输出写入 `code/data/processed/q1/_smoke/`。冒烟和测试通过后才启动正式运行。
10. 耗时任务按单元把中间结果写入 `code/data/processed/q1/_chunks/<任务名>/<单元>.npz`；已存在的单元跳过；最终 npz 由中间文件合并，合并前断言单元个数。`_smoke/` 与 `_chunks/` 已加入 `.gitignore`（W1 任务 0.7）。
11. 并行：`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 设为 1；用 joblib 进程池，进程数不超过 8，并行只在外层折、重复或维度组上展开。
12. 种子：`SEED = 0`。外层第 r 次重复（r=0..4）用 `SEED + r`，内层用 `SEED + 100 + r`；树模型 `random_state=SEED`；bootstrap 用 `np.random.default_rng(SEED)`，2,000 次；每个 npz 存 `seed`。
13. 格式：`PY -m black --check` 与 `PY -m ruff check` 对 Q1 范围通过（配置见 W1 任务 0.5）。git commit 须先征得用户同意。
14. 卡住时，在回报中写「卡点」：文件、命令、报错全文、已尝试的做法。不绕过、不降级、不写占位结果。
15. 回报格式：运行命令、耗时、产物路径与 SHA-256 前 12 位、验收项逐条的观测值与通过状态、偏离工单之处。

## 处理结果

coding 回报（2026-10-29，状态 ack，待 modeling 验收）。

运行命令（`PY=C:/Users/11600/anaconda3/envs/mathorcup_new/python.exe`）：按「验收」三条命令执行；再生成检查已做（删除 `figures/q1/` 下除 `legacy_z/` 以外的全部文件后只运行导出、绘图与表脚本，全部文件重新出现）。

结构类检查全部通过：7 个 `fig_q1_*.pdf`、5 组 `tab_q1_*.csv/.tex`；PDF 均以 `%PDF` 开头且非空；宽表 (250, 52)，长表 12500 行；`REPORT_ROWS` 与多日期名单相同；`tab_q1_metrics.csv` 与 `q1_metrics.json` 在 1e-9 内一致；pytest 全部 123 项通过（含 XeLaTeX 编译与逐页渲染检查）。

交付来源：`success` = True，`model_key` = `main`。

图（SHA-256 前 12 位）：`fig_q1_confusion.pdf` `4e905014e264`、`fig_q1_paired_diff.pdf` `95dd339e773b`、`fig_q1_reference_error.pdf` `4c40de97db9d`、`fig_q1_reliability.pdf` `964e16c73d71`、`fig_q1_rules.pdf` `73fa672367be`、`fig_q1_sensitivity.pdf` `740845776d59`、`fig_q1_stress.pdf` `eae670a10153`。

交付文件：`q1_delivery_scores.npz` `e9b10302dd7d`、`q1_delivery_scores.json` `7bbed123f65a`、`q1_final_scores.csv` `b33c69e8a1ce`、`q1_scores_long.csv` `b4e4b23497e1`。

`REPORT_ROWS`（行号 → `beer`）：21→202、37→185、42→139、49→224、91→219、104→37、112→238、130→23、166→127、190→75、201→214、206→114。

各脚本读入文件：`export_q1_scores.py`：`q1_metrics.json`、`q1_data.npz`、`q1_scores.npz`、`q1_main_oof.npz`（`success=False` 时改读 `q1_baselines.npz`）；`fig_q1_rules`、`fig_q1_paired_diff`、`fig_q1_reliability`、`fig_q1_reference_error`、`tab_q1_rules`、`tab_q1_paired`：`q1_metrics.json`；`fig_q1_confusion`：`q1_metrics.npz`；`fig_q1_sensitivity`：`q1_sensitivity.json`；`fig_q1_stress`：`q1_stress.json`；`tab_q1_metrics`：`q1_metrics.json`、`q1_baselines.npz`；`tab_q1_external`：`q1_external.json`；`tab_q1_report_beers`：`q1_scores_long.csv`、`q1_data.npz`、`q1_delivery_scores.json`，常量 `REPORT_ROWS`。

`paper/main.tex` 与 `paper/main.pdf` 按用户要求保持队友远端版本，未加入 `longtable`、`pdflscape`；`tab_q1_report_beers.tex` 需要这两个宏包，引用前由 writing 在论文导言区自行决定是否加入。测试中的编译检查使用独立的临时包装文档，不依赖 `paper/main.tex`。

偏离工单之处：
- `git diff --stat` 显示 `code/src/q1/` 下的求解模块相对 HEAD 有改动，均来自 W2–W7 的重写；W8 未修改任何求解模块。
- `plotstyle.py` 另含模型、规则、敏感性臂的中文标签字典与读入辅助函数（`load_json`、`load_npz`、`script_args`），供各绘图与表脚本共用。
- `tables.write_table` 增加可选参数 `tex_df`、`colfmt`，使 CSV 保留机器可读列名而 TeX 使用中文表头。
- `tab_q1_external` 的表尾汇总行与逐行表共用列，汇总行中 `ctrl_mean`、`ctrl_se`、`G_hat` 三列依次为等级 MAE、完全一致数、平均有符号误差，表注已说明。
- Q1 范围 black/ruff：`code/src/q1/panels.py` 不通过；W5 规定该文件不改，因此未格式化。其余全部通过。
