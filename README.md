# 数学建模

团队模板仓库。面向每个 agent 和团队成员的共享标准位于 `COLLABORATION.md`；各 agent 适配器为 `CLAUDE.md`、`AGENTS.md`（Codex）和 `.workbuddy/memory/MEMORY.md`（Workbuddy）。持续演进的决策记录在 `memory/` 中。

## 目录结构
- `COLLABORATION.md` — 共享标准；所有 agent 的唯一事实来源
- `review/` — 跨团队沟通线程；见 `review/README.md`
- `model/` — 持续维护的文档。按问题的方法论文档 `q<N>_methodology.md`    （来自 `_METHODOLOGY_TEMPLATE.md`）、`assumptions.md`、
  `summary_sheet.md`、`sensitivity_analysis.md`、`result_interpretation.md`、
  `strengths_and_weaknesses.md`
- `code/` — 实现（`src/`、`scripts/`），数据（`data/raw`、`data/processed`）
- `figures/` — 论文读取的生成产物
- `paper/` — LaTeX 源文件
- `memory/` — 分段式项目记忆；见 `memory/MEMORY.md`

## 代码
Python 3.11。提交前用 `black` 格式化，用 `ruff` 做 lint。
每个图/表都由 `code/scripts/` 中的脚本生成，并与其输出一起提交到 `figures/`。

## 论文
`paper/` 中的 LaTeX 源文件是权威版本。按以下两种方式构建：

本地构建：
```
cd paper
latexmk -pdf main.tex
```
需要本地 TeX 发行版——Windows 上的 MiKTeX 或 TeX Live。

CI 构建：`.github/workflows/build-pdf.yml` 会在每次涉及 `paper/` 的推送时编译 `paper/main.tex`，并将 PDF 作为构建产物上传。
