---
name: review-thread
description: How cross-team communication works in this repo via the review/ module, including the required fields of a coding work order. Invoke when opening, acknowledging, resolving, or referencing a review thread — flagging a solver anomaly, raising an implausible-result question, handing a methodology document to the coding team, or promoting a resolved lesson to memory/.
---

# Review module

> Adapter mirror of `.claude/skills/review-thread/SKILL.md` (authoritative
> source). Keep this file in sync with the source — change a rule there first,
> then mirror it here. `COLLABORATION.md` and `CLAUDE.md` point to the same
> source. Do not edit the rule text directly in this adapter.

- Cross-team communication lives in `review/`, one markdown file per thread
  (`review/_TEMPLATE.md`), lifecycle `open -> ack -> resolved`.
- Set `from`, `to`, `severity`, and `refs` in the frontmatter so the recipient
  team can find the thread and open the referenced artifacts.
- When a resolved thread carries a durable lesson, promote it to a `memory/`
  file and close the thread with a `[[link]]` to it.

Who raises what (recommended assignment; see `COLLABORATION.md`):
- Coding produces the numbers and flags anomalies here — divergence,
  instability, runtime — and raises a thread when implementation contradicts the
  methodology document.
- Writing interprets results in `model/result_interpretation.md` and raises a
  thread when a model or result reads as implausible or inconsistent with the
  paper's claims.
- Modeling reviews that interpretation and raises a thread when a reading is
  wrong, then carries confirmed readings into `sensitivity_analysis.md` and
  `strengths_and_weaknesses.md`.

## Coding work order

A work order is a `review/` thread from modeling to coding that translates a
methodology document into implementation tasks. It states the code structure,
not just the math:

- The files to add or change and where each belongs — `code/src/` for imported
  logic, `code/scripts/` for run-time entry points, per the `coding-pipeline`
  skill.
- For each file, the functions/classes to implement: name, signature (arguments
  and return), and what each computes. Reference the methodology section that
  specifies the equation or method.
- The output artifacts each script writes (`.npz`/CSV under
  `code/data/processed/`, figures under `figures/`) and the figure/script name
  pairing.
- Acceptance criteria: `black` and `ruff` pass, seeds recorded, figures
  regenerate from processed data without re-solving, plus any per-task numerical
  or sign checks.
