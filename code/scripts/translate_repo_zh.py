"""Translate the repo's English working documents to Chinese, in place.

One-off switch of the working language to Chinese. Each file in ``FILES`` is
sent whole to DeepSeek (credentials from ``~/.workbuddy/models.json``), the
English original is copied to ``--backup`` first, and the translation
overwrites the file. ``report.json`` in the backup directory lists protected
tokens (code spans, paths, markers, math, numbers) that did not survive, for
review before committing.

    python code/scripts/translate_repo_zh.py
    python code/scripts/translate_repo_zh.py --only model/assumptions.md --dry-run

Code files are out of scope; ``paper/`` and documents already in Chinese are
not listed. The ``translate-docs`` skill is retired by this switch rather than
translated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import repo_zh  # noqa: E402
import transcribe_for_communication as tfc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS_JSON = Path.home() / ".workbuddy" / "models.json"
# The DeepSeek entry in models.json stores the full endpoint, not a base URL.
CREDENTIAL_ID = "deepseek-v4-pro"

FILES = [
    "COLLABORATION.md",
    "CLAUDE.md",
    "AGENTS.md",
    "README.md",
    "review/README.md",
    "review/_TEMPLATE.md",
    "memory/MEMORY.md",
    ".workbuddy/memory/MEMORY.md",
    ".workbuddy/memory/_TEMPLATE.md",
    ".workbuddy/skills/coding-pipeline/SKILL.md",
    ".workbuddy/skills/latex-paper/SKILL.md",
    ".workbuddy/skills/review-thread/SKILL.md",
    "model/_METHODOLOGY_TEMPLATE.md",
    "model/assumptions.md",
    "model/summary_sheet.md",
    "model/sensitivity_analysis.md",
    "model/result_interpretation.md",
    "model/strengths_and_weaknesses.md",
    "model/q1_methodology.md",
]


def deepseek_credentials() -> tuple[str, str]:
    """``(endpoint, api_key)`` from the Workbuddy model config."""
    for entry in json.loads(MODELS_JSON.read_text(encoding="utf-8")):
        if entry.get("id") == CREDENTIAL_ID and entry.get("apiKey"):
            return str(entry["url"]), str(entry["apiKey"])
    raise SystemExit(f"no {CREDENTIAL_ID} key in {MODELS_JSON}")


def _stream(endpoint: str, key: str, payload: dict) -> str:
    """POST with ``stream: true`` and join the content deltas.

    Long documents take minutes to generate; a non-streamed response sits idle
    on the socket that long and was cut mid-read in testing.
    """
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({**payload, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    parts: list[str] = []
    finish = None
    with urllib.request.urlopen(request, timeout=600.0) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            choice = json.loads(data)["choices"][0]
            parts.append((choice.get("delta") or {}).get("content") or "")
            finish = choice.get("finish_reason") or finish
    if finish != "stop":
        raise tfc.TranslationError(f"stream ended with finish_reason={finish!r}")
    text = "".join(parts).strip()
    if not text:
        raise tfc.TranslationError("empty content")
    return text


def translate(english: str, model: str, endpoint: str, key: str) -> str:
    """One document, with deterministic backoff on transient failures."""
    payload = {
        "model": model,
        "max_tokens": 64000,
        "temperature": 0.2,
        "messages": repo_zh.build_messages(english),
    }
    last: Exception | None = None
    for attempt in range(3):
        try:
            return repo_zh.strip_wrapping_fence(_stream(endpoint, key, payload))
        except (tfc.TranslationError, OSError, ValueError) as exc:
            # OSError covers HTTP, SSL and connection resets mid-read.
            last = exc
            time.sleep(4.0 * 2**attempt)
    raise tfc.TranslationError(f"failed after 3 attempts: {last}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=repo_zh.DEFAULT_MODEL)
    parser.add_argument("--only", nargs="*", help="subset of FILES")
    parser.add_argument(
        "--backup", type=Path, default=ROOT.parent / f"{ROOT.name}_en_backup"
    )
    parser.add_argument("--dry-run", action="store_true", help="write to backup only")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    files = args.only or FILES
    endpoint, key = deepseek_credentials()
    args.backup.mkdir(parents=True, exist_ok=True)

    def one(rel: str) -> tuple[str, dict]:
        src = ROOT / rel
        english = src.read_text(encoding="utf-8")
        orig = args.backup / "en" / rel
        orig.parent.mkdir(parents=True, exist_ok=True)
        if not orig.exists():
            shutil.copy2(src, orig)
        out = args.backup / "zh" / rel if args.dry_run else src
        if args.dry_run and out.exists():
            chinese = out.read_text(encoding="utf-8")
            return rel, repo_zh.preservation_diff(english, chinese)
        started = time.monotonic()
        chinese = translate(english, args.model, endpoint, key)
        if not chinese.endswith("\n"):
            chinese += "\n"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(chinese, encoding="utf-8")
        print(f"{rel}: {time.monotonic() - started:.0f}s", flush=True)
        return rel, repo_zh.preservation_diff(english, chinese)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        report = dict(pool.map(one, files))

    report_path = args.backup / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flagged = [f for f, d in report.items() if d]
    print(f"done: {len(files)} file(s), {len(flagged)} with token drift -> {report_path}")


if __name__ == "__main__":
    main()
