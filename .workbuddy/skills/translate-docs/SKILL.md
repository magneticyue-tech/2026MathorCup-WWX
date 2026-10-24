---
name: translate-docs
description: How to regenerate the Chinese reader-facing copy of a model/ document after the English authoritative version is finished or changed. Invoke when a model/ modeling or analysis document has been drafted, finalized, or edited and its Chinese copy needs to be created or brought back in sync. Covers the English-name_Chinese-name naming rule, full vs update mode, and which files are never translated.
---

# Translate model/ docs to Chinese

> Adapter mirror of `.claude/skills/translate-docs/SKILL.md` (authoritative
> source). Keep this file in sync with the source — change a rule there first,
> then mirror it here. `COLLABORATION.md` and `CLAUDE.md` point to the same
> source. Do not edit the rule text directly in this adapter.

The English documents under `model/` are the authoritative version the agent
reasons over. Each has a Chinese copy for human readers, regenerated from the
English by a relay model (qwen/glm), not by the agent. Keeping the copies out of
the agent's reasoning context is the point: a copy may read as translationese
without degrading the model, because the model never reads it back.

Regenerate the Chinese copy after the English document is finished or changed.
Do not hand-translate, and do not edit a Chinese copy directly — a direct edit
is lost the next time the copy is regenerated from its English source.

## Naming

The Chinese copy sits beside its English source in `model/`, named
`<english-stem>_<中文名>.md`:

- `assumptions.md` -> `assumptions_假设.md`
- `summary_sheet.md` -> `summary_sheet_摘要表.md`
- `sensitivity_analysis.md` -> `sensitivity_analysis_敏感性分析.md`
- `result_interpretation.md` -> `result_interpretation_结果解读.md`
- `strengths_and_weaknesses.md` -> `strengths_and_weaknesses_优缺点.md`
- `q<N>_methodology.md` -> `q<N>_methodology_q<N>方法.md`
  (e.g. `q1_methodology.md` -> `q1_methodology_q1方法.md`)

`_METHODOLOGY_TEMPLATE.md` is a scaffold, not a document, and is never
translated. `code/scripts/translate_doc.py` rejects any stem with no Chinese
name rather than guessing one; add a new document's name to `ZH_NAME` in that
script before translating it.

## Two modes

The driver script `code/scripts/translate_doc.py` imports the relay logic from
`code/src/transcribe_for_communication.py` and runs one of two modes. The
relay model is OpenAI-protocol only; `qwen3.7-plus` is the default, `glm-5.2`
the alternative via `--model`. `kimi-k3` is not usable here — it speaks the
Anthropic `/messages` protocol the call layer does not implement.

- `full` — regenerate the whole Chinese copy from a finished English document.
  Use after a first draft or a large rewrite. Overwrites the copy.

      python code/scripts/translate_doc.py full model/assumptions.md

- `update` — rewrite only the Chinese passages matching a changed English line
  range, leaving the rest of the copy untouched. Use after a small edit to an
  English document that already has a Chinese copy. Requires `--lines`, the
  1-based inclusive line range of the change in the English file.

      python code/scripts/translate_doc.py update model/assumptions.md --lines 40-58

Pick `update` for a localized edit and `full` for a rewrite that touches most of
the document; `update` sends the whole English file and the current Chinese copy
for context but only rewrites the corresponding region, so it preserves
hand-reviewed Chinese elsewhere in the file.

## Prerequisite

The relay key comes from `RIGHTAPI_API_KEY` in the environment, or from
`~/.workbuddy/models.json` if the env var is unset. If neither is present the
script exits with that message rather than calling the relay.

## Workbuddy execution note

Run the driver with the managed Python runtime (see the session's available
runtimes), not a bare `python`, so the relay library imports resolve against the
isolated venv:

    <managed-python> code/scripts/translate_doc.py full model/assumptions.md

Install the relay client and any dependencies into that venv first — never
globally. Run the command from the repo root so the `code/src/` and
`code/scripts/` relative imports resolve.

## After regenerating

The status markers `[PLACEHOLDER]` and `[PENDING]` are carried into the Chinese
copy by the prompt. A Chinese copy inherits the state of its English source: if
the English document is marked, the copy is too. Do not treat a Chinese copy as
a place to resolve a marker — resolve it in the English source and regenerate.
