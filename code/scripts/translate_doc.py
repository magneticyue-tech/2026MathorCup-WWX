"""Regenerate the Chinese reader-facing copy of a model/ English document.

Run-time entry point per the coding-pipeline convention: this script does the
reading and writing; ``code/src/transcribe_for_communication.py`` holds the
import-only relay logic.

The English documents under ``model/`` are authoritative. Each has a Chinese
copy named ``<english-stem>_<中文名>.md`` in the same directory, regenerated
from the English by a relay model (qwen/glm) so Claude's reasoning context stays
English-only. ``_METHODOLOGY_TEMPLATE.md`` is a scaffold, not a document, and is
never translated.

Two modes:

- ``full``: regenerate the whole Chinese copy from the finished English doc.
      python code/scripts/translate_doc.py full model/assumptions.md
- ``update``: rewrite only the Chinese passages matching a changed English line
  range, leaving the rest of the copy untouched.
      python code/scripts/translate_doc.py update model/assumptions.md --lines 40-58

The Chinese-name suffix for each English stem is fixed in ``ZH_NAME`` so the
copy's filename is stable across runs. An English stem with no entry there is
rejected rather than guessed, so a new document gets a deliberate Chinese name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``code/src`` importable when this script is run directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import transcribe_for_communication as tfc  # noqa: E402

# The Chinese suffix appended to each English stem to name its copy, e.g.
# ``assumptions.md`` -> ``assumptions_假设.md``. Per-problem methodology files
# are matched separately below because their stem carries a problem number.
ZH_NAME = {
    "assumptions": "假设",
    "summary_sheet": "摘要表",
    "sensitivity_analysis": "敏感性分析",
    "result_interpretation": "结果解读",
    "strengths_and_weaknesses": "优缺点",
}

# A scaffold copied to create each q<N>_methodology.md, not a document itself.
# Translating it would produce a Chinese copy nothing reads.
NEVER_TRANSLATE = {"_METHODOLOGY_TEMPLATE"}


def chinese_stem(english_stem: str) -> str:
    """``<english-stem>_<中文名>`` for a model/ document, or raise.

    ``q<N>_methodology`` is handled by pattern because the problem number is
    part of the stem; the fixed docs come from ``ZH_NAME``. An unknown stem is
    an error, not a guess, so a new document is given a deliberate Chinese name
    in ``ZH_NAME`` before it can be translated.
    """
    if english_stem in NEVER_TRANSLATE:
        raise ValueError(f"{english_stem} is a scaffold and is never translated")
    if english_stem in ZH_NAME:
        return f"{english_stem}_{ZH_NAME[english_stem]}"
    if english_stem.endswith("_methodology") and english_stem[: -len("_methodology")]:
        problem = english_stem[: -len("_methodology")]
        return f"{english_stem}_{problem}方法"
    raise ValueError(
        f"no Chinese name for stem {english_stem!r}; add it to ZH_NAME in this "
        "script before translating a new document"
    )


def chinese_path(english_path: Path) -> Path:
    """Path of the Chinese copy beside its English source."""
    return english_path.with_name(f"{chinese_stem(english_path.stem)}.md")


def parse_line_range(spec: str) -> tuple[int, int]:
    """``"40-58"`` or a single ``"40"`` to a 1-based inclusive ``(start, end)``."""
    parts = spec.split("-")
    try:
        if len(parts) == 1:
            start = end = int(parts[0])
        elif len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
        else:
            raise ValueError
    except ValueError:
        raise SystemExit(f"--lines must be N or N-M, got {spec!r}")
    if start < 1 or end < start:
        raise SystemExit(f"--lines range is not 1-based ascending: {spec!r}")
    return start, end


def run_full(english_path: Path, model: str) -> None:
    """Regenerate the whole Chinese copy from the English document."""
    english = english_path.read_text(encoding="utf-8")
    out_path = chinese_path(english_path)
    base_url, api_key = tfc.relay_credentials(model)
    messages = tfc.build_full_messages(english)
    result = tfc.translate_with_retry(
        messages, "full", model=model, base_url=base_url, api_key=api_key
    )
    out_path.write_text(result.text, encoding="utf-8")
    print(
        f"full: {english_path} -> {out_path} "
        f"({result.model}, {result.attempts} attempt(s), {result.elapsed_s:.1f}s)"
    )


def run_update(english_path: Path, line_range: tuple[int, int], model: str) -> None:
    """Rewrite only the Chinese passages matching a changed English line range."""
    english = english_path.read_text(encoding="utf-8")
    out_path = chinese_path(english_path)
    if not out_path.is_file():
        raise SystemExit(f"no Chinese copy at {out_path} to update; run 'full' first")
    current_chinese = out_path.read_text(encoding="utf-8")
    base_url, api_key = tfc.relay_credentials(model)
    messages = tfc.build_update_messages(english, line_range, current_chinese)
    result = tfc.translate_with_retry(
        messages, "update", model=model, base_url=base_url, api_key=api_key
    )
    out_path.write_text(result.text, encoding="utf-8")
    start, end = line_range
    print(
        f"update: {english_path} lines {start}-{end} -> {out_path} "
        f"({result.model}, {result.attempts} attempt(s), {result.elapsed_s:.1f}s)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("full", "update"))
    parser.add_argument("english_doc", type=Path, help="path to the English model/ doc")
    parser.add_argument(
        "--lines",
        help="changed English line range N or N-M, required for update mode",
    )
    parser.add_argument(
        "--model",
        default=tfc.DEFAULT_MODEL,
        help=f"relay model (OpenAI-protocol only; default {tfc.DEFAULT_MODEL})",
    )
    args = parser.parse_args(argv)

    english_path = args.english_doc
    if not english_path.is_file():
        raise SystemExit(f"no such file: {english_path}")

    if args.mode == "full":
        run_full(english_path, args.model)
    else:
        if not args.lines:
            raise SystemExit("update mode requires --lines N-M")
        run_update(english_path, parse_line_range(args.lines), args.model)


if __name__ == "__main__":
    main()
