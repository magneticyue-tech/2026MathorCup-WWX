# Collaboration standards

Single source of truth for all agents (Claude Code, Codex, Workbuddy) and every
team member.
Agent config files are thin adapters that point here; do not duplicate these
rules into an agent file, update this document instead.

## Team roles and ownership
- Modeling — assumptions, derivations, model selection, and review of result
  interpretation. Owns `model/`, except `result_interpretation.md`.
- Coding — implementation, solver runs, figures, tables. Owns `code/`.
- Writing — paper draft, typesetting, and interpretation of results. Owns
  `paper/` and `model/result_interpretation.md`.

Each team owns its directory; cross-team requests go through the review module.

## Model directory documents
`model/` holds two kinds of document, both owned by modeling.

Per-problem methodology documents, one per problem, named `q<N>_methodology.md`
(`q1_methodology.md`, `q2_methodology.md`, ...), created from
`model/_METHODOLOGY_TEMPLATE.md`. Each states the complete model and method for
every sub-part of that problem (Q<N>.1, Q<N>.2, ...): notation and units, and
the equations/estimators for each sub-part with why they were chosen. Methods
only: code-related requirements (output files, figures) belong in the coding
work order, and interpretation of results in `result_interpretation.md`.

Living documents, maintained continuously: `q<N>_methodology.md`, `assumptions.md`,
`summary_sheet.md`, `sensitivity_analysis.md`, `result_interpretation.md`,
`strengths_and_weaknesses.md`. The paper's matching sections are generated from
these. Keep measured results, interpretation, and evaluation
(strengths/weaknesses) separate.

Assumptions are split by scope. `assumptions.md` records only global
assumptions — those conditioning the data, scope, or environment of the whole
problem (e.g. one-population sampling, SI units, still-air flight); it generates
the paper's Assumptions section and holds the shared notation table.
Method-local assumptions — functional-form, estimator, or scope choices tied to
one method or equation (e.g. exponential decay of a state, a particular
imputation scheme, no outlier removal) — are stated inside the relevant
`q<N>_methodology.md` at the point of use, worded explicitly as assumptions so
`sensitivity_analysis.md` and `strengths_and_weaknesses.md` can cite them; they
are not listed in `assumptions.md`. Reference every assumption by content, not
by serial number. Classify by content — global if it conditions the whole
problem, method-local if it attaches to a single method — not by anticipating
what a grader expects to see.

## Artifact status markers
Work runs in parallel and returns to earlier stages, so any document may be read
while parts of it are unfinished or known-wrong. Mark those parts in place rather
than leaving the reader to infer completeness from silence.

- `[PLACEHOLDER]` — nothing written here yet, and what belongs here is known.
  Use for a `.tex` section awaiting a first draft, a table row whose value has
  not been produced, a methodology sub-part not yet specified.
- `[PENDING]` — content exists but is superseded or unverified, and a specific
  action will replace it. Use for a `.tex` passage that needs rewriting after a
  model change, code that needs modification, a result awaiting a re-run.
example:
<!-- [PLACEHOLDER] | Await: <what‑to‑produce> -->
<!-- [PENDING] | Await: <trigger‑action> -->
Every marker names what it is waiting on: the `review/` thread, work-order task,
script, or decision that will clear it. A bare marker states that something is
missing without telling the reader who clears it or when.

A downstream team may write against a marked artifact. Doing so is a choice, not
a violation: carry the marker forward into what you produce, and ensure users
are aware of and review the marker if you intend to clear it. Do not silently
treat marked content as final.

## Cross-team synchronization (read before acting)
Model files and solver results change often, and acting on a stale copy wastes
work.
Before starting a task, and again before writing anything downstream:
- Modeling and Writing: re-read the `model/` living docs (`assumptions.md`,
  `summary_sheet.md`, `sensitivity_analysis.md`, `result_interpretation.md`,
  `strengths_and_weaknesses.md`) and the latest solver outputs in `figures/`
  and `code/data/processed/`.
- Coding: re-read `model/q<N>_methodology.md` before a solver run so the
  implementation matches current model.
- All agents: check `git log` / `git diff` for recent changes to `model/` and
  `figures/`, and scan `review/` for open threads addressed to your team.

Codex and Workbuddy in particular do not receive automatic file-change
notifications. At the start of every task, before relying on any cached context,
they must re-read:
- the repository commit log (`git log`, and `git diff` for the relevant range);
- the updated `model/` living documents;
- the updated solver output files under `figures/`;
- the updated derived data under `code/data/processed/`.

## Interpreting solver results
Coding produces the numbers and flags anomalies via the review module —
divergence, instability, runtime, or an implementation that contradicts the
methodology document.

Writing owns `result_interpretation.md` and fills it in before drafting the
corresponding paper section: the measured value in one column, what it means in
the next. Modeling reviews the interpretation column and raises a review thread
when a reading is wrong or a result is implausible, and carries the confirmed
readings into `sensitivity_analysis.md` and `strengths_and_weaknesses.md`.

Writing is the team that must state what a result means in order to draft, so
interpretation is recorded where it is produced instead of being requested from
modeling first. Modeling holds the correctness check because it owns the methods
and the two documents that cite the readings.

## Task-scoped rules (Claude Code skills)
Three task-scoped rule sets live as Claude Code skills under
`.claude/skills/`, which hold their authoritative text. Codex and Workbuddy
read the `SKILL.md` files and mirror them into their own adapters.
- Review module — `.claude/skills/review-thread/SKILL.md`: how cross-team
  communication works in `review/` (one file per thread, lifecycle
  `open -> ack -> resolved`, promote durable lessons to `memory/`), and the
  required fields of a coding work order.
- Coding standards — `.claude/skills/coding-pipeline/SKILL.md`: `src/` vs
  `scripts/` split, the solve -> `.npz` -> plot pipeline, seeding, figure
  naming, `data/raw/` read-only.
- LaTeX conventions — `.claude/skills/latex-paper/SKILL.md`: one sentence per
  line in `.tex`, build artifacts git-ignored.
- Translate docs — `.claude/skills/translate-docs/SKILL.md`: regenerating the
  Chinese reader-facing copy of a `model/` document after its English source is
  finished or changed (naming, full vs update mode, what is never translated).

## Document language
Reasoning-dense authoritative documents are written in English; the reader-
facing Chinese copies are derived from them.

- `model/` modeling and analysis documents are authoritative in English. Claude
  reasons over the English version, and no Chinese copy is fed back into its
  reasoning context. Keeping the reasoning context English-only is the reason
  the copies exist separately: a copy may read as translationese without
  degrading the model, because the model never reads it back.
- Each English `model/` document has a Chinese copy beside it named
  `<english-stem>_<中文名>.md`, regenerated from the English by a relay model
  (qwen/glm) through the `translate-docs` skill, not hand-translated or edited
  in place. `_METHODOLOGY_TEMPLATE.md` is a scaffold and is never translated.
- Skills under `.claude/skills/` and code comments stay in English.
- The paper's language is unchanged by this split; `paper/` is out of scope
  here.

## Output specifications
Applies to all prose every agent produces — paper text, review threads, commit
messages, and status reports to the team.

Content constraints:
- Every performance claim cites a figure, table, or number.
- Separate objective statement from evaluation: results state what was measured;
  strengths/weaknesses (and any judgment) are labeled as such.
- No filler intensifiers (significantly, remarkably, notably) unless backed by a
  number.
- No negation-then-affirmation for rhetorical effect ("not merely X, but Y";
  "far from being X, it is Y").
- Work content into the sentence rather than parking it in parentheses: if the
  material belongs in the argument, it belongs in the clause. Reserve
  parentheses for what reads as genuinely secondary — a unit, a symbol
  definition, a source or seed, a cross-reference. Every parenthetical carries
  content of that kind; an aside that addresses the grader or restates the
  prompt's requirement carries none: "(no additional assumptions required)",
  "(detailed elaboration)", "(as required)", "(meeting the requirement)". State
  the substance directly; if no assumption is needed, add none.
- Justify every choice by its effect on the model, the data, or the result:
  what it assumes, what it buys, what it costs. "A grader would expect this
  here" and "listing an extra assumption is nearly free" are not reasons — they
  argue from anticipated reception rather than from the substance, and they
  produce content that survives review without earning its place.

Reporting tone:
- Name what was created or changed with a plain noun phrase. Drop selling-point
  parentheticals and readability-tailored elaboration from summaries.
  Rejected: "model/: four continuously-maintained docs (kept in sync all along,
  not written at project close)". Preferred: "the new model/ directory".
- State outcomes plainly: if something failed or was skipped, say so; do not
  soften or oversell. Prefer honest correction over agreement.
- Discuss the model and method directly, and state file changes as plain facts
  at the point they happen. Reserve process narration for what changes the
  reader's next action: a blocker, a decision needed, a deviation from what was
  asked. Sentences whose whole content is the state of your own work —
  "no files written yet; this is the detail-check you asked for", "just to
  confirm before I start", "as you can see" — report on the reporting.
- Do not open a reply with flattery of the request or of the user.
