# 数学建模

团队模板仓库。面向每个 agent 和团队成员的共享标准位于 `COLLABORATION.md`；各 agent 适配器为 `CLAUDE.md`、`AGENTS.md`（Codex）和 `.workbuddy/memory/MEMORY.md`（Workbuddy）。持续演进的决策记录在 `memory/` 中。

## 目录结构
- `COLLABORATION.md` — 共享标准；
- `review/` — 团队沟通线程；见 `review/README.md`
- `model/` — 持续维护的文档。按问题的方法论文档 `q<N>_methodology.md`    （来自 `_METHODOLOGY_TEMPLATE.md`）、`assumptions.md`、
  `summary_sheet.md`、`sensitivity_analysis.md`、`result_interpretation.md`、
  `strengths_and_weaknesses.md`
- `code/` — 实现（`src/`、`scripts/`），数据（`data/raw`、`data/processed`）
- `figures/` — 论文读取的生成产物
- `paper/` — LaTeX 源文件
- `memory/` — 分段式项目记忆；见 `memory/MEMORY.md`

## 代码
Python 3.11。
每个图/表都由 `code/scripts/` 中的脚本生成，并与其输出一起提交到 `figures/`。
