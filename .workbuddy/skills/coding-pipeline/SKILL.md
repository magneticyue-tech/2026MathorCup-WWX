---
name: coding-pipeline
description: 本仓库中求解器/模型代码、求解驱动脚本、绘图脚本以及图表/表格的编写规范。在新增或编辑 `code/` 下的任何内容时调用——包括 `src/` 模块、`scripts/` 入口、`.npz` 输出或图表。涵盖 `src` 与 `scripts` 的划分、solve→npz→plot 流水线、随机种子设置以及图表命名。
---

# 编码流水线规范

> `.claude/skills/coding-pipeline/SKILL.md` 的适配镜像（权威来源）。保持本文件与
> 源文件同步——先在那里改动规则，再镜像到此处。`COLLABORATION.md` 和 `CLAUDE.md` 指向同一
> 来源。不要直接在此适配文件中编辑规则文本。

`code/` 目录的权威来源。

- Python 3.11。格式化工具 `black`，检查工具 `ruff`；提交前两者都要运行。
- 每个随机过程都固定其随机种子，并在脚本中记录该值。
- `src/` 与 `scripts/` 的划分依据是导入时与运行时，而不是代码是否“可复用”：
  - `code/src/` 存放供 `import` 的代码——以函数和类形式组织的求解器/模型逻辑，导入时不得有 `__main__`、不得写文件、不得绘图。
  - `code/scripts/` 存放可直接运行的入口（每个问题或每张图一个）。脚本从 `src/` 导入，执行求解和/或绘图。针对单个问题的 `main()` 属于这里，不属于 `src/`。
- 在 `scripts/` 内，求解驱动脚本与绘图脚本要分开，使图表无需重新求解即可重生成：求解脚本写出 `.npz`，绘图脚本读取它。单个脚本不得同时求解和绘图。
- 求解器结果以 `.npz` 文件形式保存在 `code/data/processed/` 下。绘图脚本加载这些 `.npz` 文件，而不是重新运行求解器。流水线为：求解器核心（`src/`）-> 求解脚本（`scripts/`）-> `.npz`（`data/processed/`）-> 绘图脚本（`scripts/`）-> 图表（`figures/`）。
- 每张图/每个表都由 `code/scripts/` 中的脚本生成，并与输出一并提交到 `figures/`；脚本名与产物名一致，例如 `scripts/fig_03_sensitivity.py` -> `figures/fig_03_sensitivity.pdf`。
- `data/raw/` 为只读；处理结果写入 `data/processed/`。

## Workbuddy 执行说明

本项目面向 Python 3.11 语法与特性。请使用受管 Python 运行时（参见当前会话可用的运行时）运行 `code/scripts/` 入口，而不要使用裸 `python`。将 `black`、`ruff` 及任何求解器依赖安装到受管运行时目录下的隔离 venv 中——绝不全局安装，也绝不安装到系统解释器。受管的 3.13 运行时可以运行 3.11 兼容的代码；不要将 3.12+ 独有的语法引入 `code/`。
