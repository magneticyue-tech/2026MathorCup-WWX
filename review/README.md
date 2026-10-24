# 评审模块

本仓库的跨团队沟通机制。每个线程一个 markdown 文件，命名为 `YYYY-MM-DD-short-slug.md`，由 `review/_TEMPLATE.md` 创建。

## 工作方式
- 当一个团队需要另一个团队行动或决策时，提交一个线程（例如 Coding 向 Modeling 报告求解器异常）。
- 在 frontmatter 中设置 `from`、`to`、`severity` 和 `refs`，以便对应的团队能找到它并打开所引用的产物。
- 生命周期：`open` -> `ack`（接收方已看到）-> `resolved`。
- 所有 agent 在会话开始时扫描 `review/`，查找发给本团队的 open 线程（见 `COLLABORATION.md`）。
- 提交任何内容前，务必先与用户沟通。

## 沉淀为记忆
当已解决的线程承载了应当长期保留的经验教训（新的约束、约定）时，为其创建一个 `memory/` 文件，并在该线程的 Resolution 部分链接该文件。线程是对话；`memory/` 才是持久记录。
