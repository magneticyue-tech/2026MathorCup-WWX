---
name: coding-pipeline
description: Standards for writing solver/model code, solve drivers, plotting scripts, and figures/tables in this repo. Invoke when adding or editing anything under code/ — src/ modules, scripts/ entry points, .npz outputs, or figures. Covers the src-vs-scripts split, the solve→npz→plot pipeline, seeding, and figure naming.
---

# Coding pipeline standards

> Adapter mirror of `.claude/skills/coding-pipeline/SKILL.md` (authoritative
> source). Keep this file in sync with the source — change a rule there first,
> then mirror it here. `COLLABORATION.md` and `CLAUDE.md` point to the same
> source. Do not edit the rule text directly in this adapter.

Authoritative source for the `code/` directory.

- Python 3.11. Formatter `black`, linter `ruff`; run both before committing.
- Every random process fixes its seed and records the value in the script.
- `src/` vs `scripts/` is decided by import-time vs run-time, not by whether
  code is "reusable":
  - `code/src/` holds code you `import` — solver/model logic as functions and
    classes, with no `__main__`, no file writes, and no plotting at import.
  - `code/scripts/` holds entry points you run directly (one per problem or
    figure). A script imports from `src/`, runs the solve, and/or plots. A
    per-problem `main()` belongs here, not in `src/`.
- Within `scripts/`, keep the solve driver separate from the plotting script so
  figures regenerate without re-solving: a solve script writes the `.npz`, a
  plotting script reads it. A single script does not both solve and plot.
- Save solver results as `.npz` files under `code/data/processed/`. Plotting
  scripts load those `.npz` files instead of re-running the solver. The pipeline
  is solver core (`src/`) -> solve script (`scripts/`) -> `.npz`
  (`data/processed/`) -> plotting script (`scripts/`) -> figure (`figures/`).
- Every figure/table is produced by a script in `code/scripts/`, committed with
  its output in `figures/`; script name matches the artifact, e.g.
  `scripts/fig_03_sensitivity.py` -> `figures/fig_03_sensitivity.pdf`.
- `data/raw/` is read-only; processing writes to `data/processed/`.

## Workbuddy execution note

The project targets Python 3.11 syntax and features. Run `code/scripts/` entry
points with the managed Python runtime (see the session's available runtimes),
not a bare `python`. Install `black`, `ruff`, and any solver dependencies into
the isolated venv under the managed runtime directory — never globally and never
into the system interpreter. The managed 3.13 runtime runs 3.11-compatible code;
do not introduce 3.12+ only syntax into `code/`.
