# Workbuddy 项目记忆

## 每个任务先读
所有智能体共用的团队标准位于仓库根目录的 `COLLABORATION.md`。
在每个任务开始时阅读该文件以及仓库根目录的 `memory/MEMORY.md`，然后遵循这两份文件。此 `.workbuddy/` 记忆是 Workbuddy 本地记忆；仓库文件才是权威。

## 团队：编码
你是编码团队智能体。遵循 `COLLABORATION.md` 中的编码标准。尤其注意：
- 将绘图代码与求解器代码分离：求解器/模型逻辑放在 `code/src/`，绘图放在 `code/scripts/`。一个脚本不得同时求解和绘图。
- 将求解器结果保存为 `code/data/processed/` 下的 `.npz` 文件；绘图脚本加载这些 `.npz` 文件，而不是重新运行求解器。
- 通过 `review/` 线程向建模团队报告求解器问题（发散、不稳定、运行时间）。

## 记忆桥接
- `.workbuddy/memory/YYYY-MM-DD.md` 每日日志是仅追加的本地草稿：记录你当天做了什么，绝不覆盖。
- 当每日日志产生持久约定、决策、约束或参考时，将其提升为仓库根目录 `memory/` 中的单事实文件，并在 `memory/MEMORY.md` 中添加一行，使 Claude 和 Codex 也能看到。不要让持久事实只存在于 `.workbuddy/` 中。
- 将本文件保持在 3000 词以内；将细节移入仓库 `memory/` 文件。

## 跨团队同步
你不会收到自动的文件变更通知。在每个任务开始时重新阅读：仓库提交日志（`git log` / `git diff`）、更新后的 `model/` 活文档、`figures/` 下更新后的求解器输出，以及 `code/data/processed/` 下更新后的派生数据；并扫描 `review/` 中发给你的团队的未关闭线程。参见 `COLLABORATION.md`。

## 技能
所有成员共享的项目技能位于 `.workbuddy/skills/`。三套 `.claude/skills/` 规则集在此以适配器副本形式镜像——权威源仍为 `.claude/skills/`，变更后重新镜像：
- `coding-pipeline` — src/scripts 拆分、solve→.npz→plot、随机种子设置、图件命名。
- `latex-paper` — 在 .tex 中每行一句，构建产物由 git 忽略。
- `review-thread` — review/ 生命周期与编码工单字段。
