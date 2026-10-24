# Project: Mathematical Modeling

`COLLABORATION.md` is the shared source of truth for all agents; the
sections below mirror it so the rules load into Claude Code's context
automatically. Change a shared rule in COLLABORATION.md and mirror it
here to keep the two in sync. Also read `memory/MEMORY.md` at session
start.

## Claude Code specifics
- Repo `memory/` is the durable, agent-neutral record (one fact per
  file, index in `memory/MEMORY.md`, template in `_TEMPLATE.md`).
  Prefer it over Claude-local memory so Codex and Workbuddy see the
  same facts.
- Editor/session config under `.claude/` is Claude-local, not
  authoritative for the team.

## Model directory
- Ownership: modeling owns `model/` except
  `result_interpretation.md`, which writing owns and fills in before
  drafting; modeling reviews the interpretation column there.
- Per-problem methodology docs `q<N>_methodology.md` (from
  `model/_METHODOLOGY_TEMPLATE.md`) hold the model and method for each
  sub-part: notation, equations/estimators, why each was chosen.
- Living docs maintained continuously: `assumptions.md`,
  `summary_sheet.md`, `sensitivity_analysis.md`,
  `result_interpretation.md`, `strengths_and_weaknesses.md`.
- Assumptions split by scope: `assumptions.md` holds only global
  assumptions (data/scope/environment) plus the shared notation table;
  method-local assumptions (functional form, estimator, preprocessing
  choice) are stated in the relevant `q<N>_methodology.md` at the point
  of use, so sensitivity and strengths/weaknesses can cite them.
  Reference assumptions by content, never by serial number.

## Artifact status markers
- `[PLACEHOLDER]` — nothing written yet, content known (a `.tex`
  section awaiting a draft, a table row whose value is not produced).
- `[PENDING]` — content exists but is superseded or unverified, and a
  specific action replaces it (a `.tex` passage to rewrite after a
  model change, code to modify, a result awaiting a re-run).
- Every marker names what clears it: the `review/` thread, work-order
  task, script, or decision.
- Writing against a marked artifact is allowed: carry the marker
  forward, and open a `review/` thread if clearing it would change your
  work. Never treat marked content as final silently.

## Task-scoped rules (skills)
Coding, review, and LaTeX rules are Claude Code skills under
`.claude/skills/`; invoke the matching skill when the task touches its
area (they load on demand instead of always-on).
- `coding-pipeline` — writing anything under `code/` (src/scripts
  split, solve -> `.npz` -> plot pipeline, seeding, figure naming).
- `review-thread` — opening/advancing/closing `review/` threads, and
  writing a coding work order (files, signatures, artifacts, acceptance
  criteria).
- `latex-paper` — editing `.tex` (one sentence per line, artifacts
  git-ignored).
- `translate-docs` — regenerating the Chinese copy of a `model/` doc
  after its English source is finished or changed (naming, full vs
  update mode, what is never translated).

## Document language
- `model/` modeling and analysis docs are authoritative in English;
  Claude reasons over the English, and no Chinese copy is fed back into
  its reasoning context. A copy may read as translationese without
  degrading the model, because the model never reads it back.
- Each English `model/` doc has a Chinese copy beside it named
  `<english-stem>_<中文名>.md`, regenerated from the English by a relay
  model (qwen/glm) via the `translate-docs` skill, not hand-translated
  or edited in place. `_METHODOLOGY_TEMPLATE.md` is never translated.
- Skills under `.claude/skills/` and code comments stay in English.
- The paper's language is unchanged; `paper/` is out of scope here.

## Output and writing style
Applies to all prose — paper, review threads, commits, status reports.
- Every performance claim cites a figure, table, or number.
- Separate objective statement from evaluation; label judgment as such.
- No filler intensifiers (significantly, remarkably, notably) unless
  backed by a number.
- No negation-then-affirmation for effect ("not merely X, but Y").
- Work content into the sentence instead of parking it in parentheses.
  Reserve parentheses for genuinely secondary material: a unit, a
  symbol definition, a source or seed, a cross-reference. Grader-facing
  asides carry no content: "(no additional assumptions required)",
  "(as required)".
- Justify choices by effect on model/data/result — what it assumes,
  buys, costs. "A grader would expect this" is not a reason.
- Report plainly: plain noun phrase for what changed, no selling-point
  parentheticals; state failures/skips honestly; prefer correction
  over agreement; no opening flattery.
- Process narration only when it changes the reader's next action
  (blocker, decision needed, deviation). Not "just to confirm before I
  start" or "no files written yet".
