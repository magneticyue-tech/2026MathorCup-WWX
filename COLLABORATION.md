# 协作规范

本文件是所有 agent（Claude Code、Codex、Workbuddy）及每位团队成员的唯一权威来源。
Agent 配置文件只是指向本文件的轻量适配层；不要把这些规则复制进 agent 文件，应更新本文档。

## 团队角色与职责归属
- Modeling——负责假设、推导、模型选择，以及评审结果解读。负责 `model/`，但 `result_interpretation.md` 除外。
- Coding——负责实现、求解器运行、图和表。负责 `code/`。
- Writing——负责论文草稿、排版和结果解读。负责 `paper/` 与 `model/result_interpretation.md`。

每个团队负责自己的目录；跨团队请求通过评审模块流转。

## model 目录中的文档
`model/` 存有两类文档，均由 Modeling 负责。

按问题编排的方法论文档，每个问题一份，命名为 `q<N>_methodology.md`
（`q1_methodology.md`、`q2_methodology.md`、……），由
`model/_METHODOLOGY_TEMPLATE.md` 创建。每份文档说明该问题每个子部分（Q<N>.1、Q<N>.2、……）的完整模型与方法：符号与单位，以及各子部分的方程/估计量及其选择理由。只写方法：与代码相关的要求（输出文件、图）属于 Coding 工作单，结果解读属于 `result_interpretation.md`。

持续维护的活文档：`q<N>_methodology.md`、`assumptions.md`、
`summary_sheet.md`、`sensitivity_analysis.md`、`result_interpretation.md`、
`strengths_and_weaknesses.md`。论文中对应的章节由这些文档生成。将实测结果、结果解读和评价（优势/不足）分开。

假设按作用范围拆分。`assumptions.md` 只记录全局假设——即约束整个问题的数据、范围或环境的假设（例如单种群抽样、SI 单位、静风飞行）；它生成论文的 Assumptions 章节，并保存共享符号表。方法局部假设——与某一方法或方程绑定的函数形式、估计量或范围选择（例如状态的指数衰减、特定插补方案、不剔除异常值）——在相关的 `q<N>_methodology.md` 中使用处说明，并明确表述为假设，以便 `sensitivity_analysis.md` 和 `strengths_and_weaknesses.md` 引用；这些假设不列入 `assumptions.md`。引用每条假设时按内容，不按编号。按内容分类：若某假设约束整个问题，则为全局假设；若只附属于单一方法，则为方法局部假设；不要按预判评分人希望看到什么来分类。

## 产物状态标记
工作并行推进，并会回退到较早阶段，因此任何文档都可能在部分内容尚未完成或已知有误时被阅读。应在原处标记这些部分，而不是让读者从无标记处推断完整性。

- `[PLACEHOLDER]`——此处尚未写入内容，但应写什么已知。用于等待初稿的 `.tex` 章节、尚未产出数值的表格行、尚未明确的方法论子部分。
- `[PENDING]`——内容已存在，但已被取代或尚未验证，并且有具体动作会替换它。用于模型变更后需要重写的 `.tex` 段落、需要修改的代码、等待重新运行的结果。

示例：
<!-- [PLACEHOLDER] | Await: <what‑to‑produce> -->
<!-- [PENDING] | Await: <trigger‑action> -->

每个标记都要写明它在等待什么：`review/` 线程、工作单任务、脚本或将其清除的决定。裸标记只说明有内容缺失，却不告诉读者由谁清除、何时清除。

下游团队可以依据已标记的产物写作。这样做是一种选择，不是违规：把标记带入你产出的内容；如果你打算清除标记，要确保使用者知晓并复核该标记。不要默默把已标记内容当作最终内容。

## 跨团队同步（行动前阅读）
模型文件和求解器结果经常变化，基于过期副本行动会浪费工作。
开始任务前，以及再次向下游写入任何内容前：
- Modeling 和 Writing：重新阅读 `model/` 的活文档（`assumptions.md`、
  `summary_sheet.md`、`sensitivity_analysis.md`、`result_interpretation.md`、
  `strengths_and_weaknesses.md`），以及 `figures/` 和 `code/data/processed/` 中最新的求解器输出。
- Coding：在运行求解器前重新阅读 `model/q<N>_methodology.md`，使实现与当前模型一致。
- 所有 agent：检查 `git log` / `git diff` 中 `model/` 和 `figures/` 的近期变更，并扫描 `review/` 中发给本团队的未关闭线程。

Codex 和 Workbuddy 尤其不会收到文件变更自动通知。每个任务开始时，在依赖任何缓存上下文之前，它们必须重读：
- 仓库提交日志（`git log`，以及相关范围的 `git diff`）；
- 更新后的 `model/` 活文档；
- `figures/` 下更新后的求解器输出文件；
- `code/data/processed/` 下更新后的派生数据。

## 解读求解器结果
Coding 产出数值，并通过评审模块标记异常——发散、不稳定、运行时长，或与方法论文档相矛盾的实现。

Writing 负责 `result_interpretation.md`，并在起草对应论文章节前填写：一列是实测值，下一列是它的含义。Modeling 评审解读列，当读数错误或结果不合理时发起评审线程，并将确认后的读数带入 `sensitivity_analysis.md` 和 `strengths_and_weaknesses.md`。

Writing 是起草前必须说明结果含义的团队，因此解读记录在产生它的地方，而不是先向 Modeling 索取。Modeling 负责正确性检查，因为它拥有方法，以及两份引用这些读数的文档。

## 任务范围内的规则（Claude Code skills）
三套任务范围内的规则以 Claude Code skills 的形式位于
`.claude/skills/` 下，权威文本保存在那里。Codex 和 Workbuddy
读取这些 `SKILL.md` 文件，并将其镜像到各自的适配层。
- 评审模块——`.claude/skills/review-thread/SKILL.md`：`review/` 中跨团队沟通如何运作（每个线程一个文件，生命周期
  `open -> ack -> resolved`，将持久有效的经验沉淀到 `memory/`），以及 Coding 工作单的必填字段。
- 编码规范——`.claude/skills/coding-pipeline/SKILL.md`：`src/` 与
  `scripts/` 的分工，solve -> `.npz` -> plot 流程，随机种子设置，图命名，`data/raw/` 只读。
- LaTeX 约定——`.claude/skills/latex-paper/SKILL.md`：`.tex` 中每行一句，构建产物由 git 忽略。

## 文档语言
仓库的工作语言为简体中文。

- `model/`、`review/`、`memory/`、协作规则与技能说明均以中文撰写，中文版本即权威版本，不再维护英文原稿或派生副本。
- 文件名、路径、代码标识符、YAML 键名与枚举取值、状态标记 `[PLACEHOLDER]`/`[PENDING]` 保持英文。
- 代码注释保持英文。
- 论文语言不受此规定影响；`paper/` 不在本节范围内。

## 输出规范
适用于所有 agent 产出的全部文字内容——论文正文、评审线程、提交信息和给团队的状态报告。

内容约束：
- 每项性能结论都要引用图、表或数字。
- 将客观陈述与评价分开：结果说明测量了什么；优势/不足（以及任何判断）要按其性质标注。
- 不使用填充性强调词（“显著”“惊人地”“尤其”），除非有数字支撑。
- 不为修辞效果使用先否定后肯定的句式（“不仅仅是 X，而是 Y”；“远非 X，实为 Y”）。
- 将内容融入句子，而不是塞进括号：如果材料属于论证，就应放进分句。括号只保留给确实次要的内容——单位、符号定义、来源或随机种子、交叉引用。每个括号都应承载这类内容；面向评分人或复述题目要求的插入语不承载任何此类内容：“（无需额外假设）”、“（详细展开）”、“（按要求）”、“（满足要求）”。直接陈述实质内容；如果不需要假设，就不要添加。
- 每个选择都要根据其对模型、数据或结果的影响来论证：它假设了什么、换来了什么、代价是什么。“评分人会希望这里这样写”和“多列一条假设几乎不花成本”不是理由——它们从预期接受度而非实质内容出发论证，会产生通过评审却没有存在价值的内容。

报告语气：
- 用朴素的名词短语说明创建或更改了什么。删去摘要中的卖点式括号和为可读性而做的铺陈。
  不推荐：“model/：四份持续维护的文档（全程同步，而非项目收尾时才写）”。推荐：“新建的 model/ 目录”。
- 平实陈述结果：如果某事失败或被跳过，就说明；不要弱化或夸大。宁可诚实纠正，也不要附和。
- 直接讨论模型和方法，并在文件变更发生时将其作为平实事实说明。过程叙述只保留给会改变读者下一步行动的内容：阻塞项、需要做出的决定、偏离要求之处。整句都在讲自己工作状态的句子——“还没有写文件；这就是你要的细节检查”、“只是开始前确认一下”、“如你所见”——只是在报告“报告”本身。
- 回复开头不要奉承请求或用户。
