# Mathematical Modeling

Team template repository. Shared standards for every agent and team member are
in `COLLABORATION.md`; per-agent adapters are `CLAUDE.md`, `AGENTS.md` (Codex),
and `.workbuddy/memory/MEMORY.md` (Workbuddy). Evolving decisions are recorded
in `memory/`.

## Layout
- `COLLABORATION.md` — shared standards; single source of truth for all agents
- `review/` — cross-team communication threads; see `review/README.md`
- `model/` — living documents. per-problem methodology docs `q<N>_methodology.md`    (from`_METHODOLOGY_TEMPLATE.md`), `assumptions.md`,
  `summary_sheet.md`, `sensitivity_analysis.md`, `result_interpretation.md`,
  `strengths_and_weaknesses.md`
- `code/` — implementation (`src/`, `scripts/`), data (`data/raw`, `data/processed`)
- `figures/` — generated artifacts the paper reads from
- `paper/` — LaTeX source 
- `memory/` — segmented project memory; see `memory/MEMORY.md`

## Code
Python 3.11. Format with `black`, lint with `ruff` before committing.
Every figure/table is produced by a script in `code/scripts/` and committed
with its output in `figures/`.

## Paper
The LaTeX source in `paper/` is authoritative. Build it two ways:

Local build:
```
cd paper
latexmk -pdf main.tex
```
Requires a local TeX distribution — MiKTeX or TeX Live on Windows.

CI build: `.github/workflows/build-pdf.yml` compiles `paper/main.tex` on every
push touching `paper/` and uploads the PDF as a build artifact.
