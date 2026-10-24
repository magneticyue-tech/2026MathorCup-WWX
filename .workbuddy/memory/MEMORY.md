# Workbuddy project memory

## Read first, every task
Shared team standards for all agents are in the repo-root `COLLABORATION.md`.
Read it and the repo-root `memory/MEMORY.md` at the start of every task, then
follow both. This `.workbuddy/` memory is Workbuddy-local; the repo files are
authoritative.

## Team: coding
You are the coding-team agent. Follow the Coding standards in
`COLLABORATION.md`. In particular:
- Separate plotting code from solver code: solver/model logic in `code/src/`,
  plotting in `code/scripts/`. One script does not both solve and plot.
- Save solver results as `.npz` files under `code/data/processed/`; plotting
  scripts load those `.npz` files instead of re-running the solver.
- Report solver problems (divergence, instability, runtime) to the modeling
  team via a `review/` thread.

## Memory bridging
- `.workbuddy/memory/YYYY-MM-DD.md` daily logs are append-only local scratch:
  record what you did that day, never overwrite.
- When a daily log produces a durable convention, decision, constraint, or
  reference, promote it to a one-fact file in the repo-root `memory/` and add a
  line to `memory/MEMORY.md`, so Claude and Codex see it too. Do not let durable
  facts live only in `.workbuddy/`.
- Keep this file under 3000 words; push detail into repo `memory/` files.

## Cross-team synchronization
You do not receive automatic file-change notifications. At the start of every
task re-read: the repository commit log (`git log` / `git diff`), the updated
`model/` living docs, the updated solver outputs under `figures/`, and the
updated derived data under `code/data/processed/`; and scan `review/` for open
threads addressed to your team. See `COLLABORATION.md`.

## Skills
Project skills shared by all members live in `.workbuddy/skills/`. The four
`.claude/skills/` rule sets are mirrored here as adapter copies — authoritative
source stays `.claude/skills/`, re-mirror on change:
- `coding-pipeline` — src/scripts split, solve→.npz→plot, seeding, figure naming.
- `latex-paper` — one sentence per line in .tex, build artifacts git-ignored.
- `review-thread` — review/ lifecycle and coding work-order fields.
- `translate-docs` — regenerating Chinese model/ copies (naming, full/update mode).
