"""Regenerate a Chinese reader-facing copy of an English authoritative doc.

Import-only per the coding-pipeline convention: every function returns data,
none writes to disk. ``scripts/translate_doc.py`` does the reading and writing.

The English documents under ``model/`` are the authoritative context Claude
reasons over; the Chinese copies are derived, human-facing, and never fed back
into Claude's reasoning context, so translationese in a copy does not degrade
the model. The translation is therefore delegated to a relay model (qwen/glm)
rather than produced by Claude.

Only the OpenAI ``/chat/completions`` protocol is supported here. ``kimi-k3`` on
this relay speaks the Anthropic ``/messages`` protocol, whose request and
response shapes differ, so it is deliberately out of scope; ``qwen3.7-plus`` is
the default and ``glm-5.2`` an alternative, both OpenAI-protocol on the same
``ali`` prefix.

Two modes share one call path:

- ``full`` regenerates a whole Chinese copy from a finished English document.
- ``update`` rewrites only the Chinese passages that correspond to a changed
  English line range, leaving the rest of the copy untouched.

Each mode has its own system prompt (``FULL_SYSTEM_PROMPT`` /
``UPDATE_SYSTEM_PROMPT``). Retries use deterministic backoff, so no seed is
involved.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# qwen3.7-plus is the default translator: text-only, OpenAI-protocol on the
# ``ali`` prefix. glm-5.2 is the alternative on the same prefix. kimi-k3 is
# excluded because it speaks the Anthropic /messages protocol, not the
# /chat/completions protocol this module posts to.
DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_RELAY_BASE = "https://www.rightapi.ai/ali/v1"
MODELS_JSON = Path.home() / ".workbuddy" / "models.json"
API_KEY_ENV = "RIGHTAPI_API_KEY"

# Translation is a generation task over a whole document of prose, equations,
# and tables, so the ceiling is higher than a single transcribed page.
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_ATTEMPTS = 3
BACKOFF_BASE_S = 4.0

# Regeneration should reproduce the source's content and structure, not rephrase
# for variety: a low temperature keeps terminology and numbers stable across the
# document while still reading as native Chinese rather than a gloss.
DEFAULT_TEMPERATURE = 0.2

# Regenerate a finished English document as a native-Chinese reader-facing copy.
# The instruction is to rewrite from the source's meaning, not to translate
# sentence by sentence, because a sentence-by-sentence gloss is exactly the
# translationese the split-language setup exists to avoid.
FULL_SYSTEM_PROMPT = """\
你是一名数学建模论文的中文技术编辑。下面给你一份英文的建模/分析文档，它是权威版本。
请依据它的内容，用地道的学术中文重新撰写一份面向中文读者的对应文档。

要求：
- 依据英文文档的含义重写，而不是逐句直译；产出要读起来像中文母语者直接写成的技术文档，不能有翻译腔。
- 完整覆盖英文文档的全部内容，不增补原文没有的结论，不省略原文有的内容。
- 保留原有的章节结构、标题层级和列表层级，与英文文档一一对应。
- 数学符号、变量名、公式、数值、单位一律照抄，不改动有效位数，不改写 LaTeX。
- Markdown 结构照旧：表格保持表格，代码块保持代码块，行内公式 $...$、独立公式 $$...$$。
- 保留状态标记 [PLACEHOLDER] 和 [PENDING] 原样，连同其后说明由谁清除的文字一起译出。
- 数学建模的固定术语用学界通行的中文译名，全篇保持一致；首次出现的关键术语在中文后用括号附上英文原词。
- 只输出重写后的中文文档本身，不要任何前言、说明或“以下是翻译”之类的话。
"""

# Update only the Chinese passages matching a changed English line range. The
# whole current Chinese copy is given for context so headings and terminology
# stay consistent, but only the corresponding region is rewritten and the rest
# is returned verbatim.
UPDATE_SYSTEM_PROMPT = """\
你是一名数学建模论文的中文技术编辑。现有一份英文权威文档和它对应的中文版。
英文文档的某一段发生了修改，需要你在中文版中对应的位置做同步修改，其余部分保持原样。

我会给你三样东西：
1. 修改后的英文文档全文（带行号）。
2. 发生修改的英文行号范围。
3. 当前的中文版全文。

要求：
- 只改写中文版中与所给英文行号范围对应的内容，其他所有部分逐字保持不变。
- 改写依据修改后英文的含义重写，用地道学术中文，不要翻译腔，与中文版其余部分的术语和文风保持一致。
- 数学符号、变量名、公式、数值、单位照抄，不改动有效位数。
- 保留 [PLACEHOLDER]、[PENDING] 标记及其说明文字。
- 保持整篇的 Markdown 结构、标题层级和术语一致。
- 输出修改后的完整中文文档，不要只输出被改的片段，也不要任何前言或说明。
"""


class TranslationError(RuntimeError):
    """The relay call failed, or returned something unusable."""


@dataclass(frozen=True)
class Translation:
    """One regenerated Chinese document and what it cost to get it."""

    text: str
    model: str
    mode: str
    elapsed_s: float
    attempts: int


def relay_credentials(
    model: str = DEFAULT_MODEL,
    models_json: Path = MODELS_JSON,
) -> tuple[str, str]:
    """``(base_url, api_key)`` for a relay model, from the env or models.json.

    The env var wins so a run can use a different key without editing the shared
    config. The key is never returned anywhere it would be printed: callers pass
    it straight to ``translate_text``.
    """
    env_key = os.environ.get(API_KEY_ENV)
    base_url = os.environ.get("RIGHTAPI_BASE_URL", DEFAULT_RELAY_BASE)
    if env_key:
        return base_url.rstrip("/"), env_key

    if not models_json.is_file():
        raise TranslationError(
            f"no {API_KEY_ENV} in the environment and no config at {models_json}"
        )
    entries = json.loads(models_json.read_text(encoding="utf-8"))
    for entry in entries:
        if entry.get("id") == model and entry.get("apiKey"):
            # The relay keys are scoped per prefix, so take the url that shipped
            # with the key rather than pairing one entry's key with another's
            # base.
            url = str(entry.get("url") or base_url).rstrip("/")
            return url, str(entry["apiKey"])
    for entry in entries:
        if str(entry.get("url", "")).rstrip("/") == base_url and entry.get("apiKey"):
            return base_url, str(entry["apiKey"])
    raise TranslationError(
        f"no API key for {model!r} in {models_json}; set {API_KEY_ENV} instead"
    )


def number_lines(text: str) -> str:
    """Prefix each line with a 1-based number, for the update prompt.

    The update mode addresses a change by English line range, so the model has
    to see the same numbering the caller used to describe the range.
    """
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(text.splitlines(), 1))


def build_full_messages(english: str) -> list[dict]:
    """Chat messages for a whole-document regeneration."""
    return [
        {"role": "system", "content": FULL_SYSTEM_PROMPT},
        {"role": "user", "content": english},
    ]


def build_update_messages(
    english: str,
    line_range: tuple[int, int],
    current_chinese: str,
) -> list[dict]:
    """Chat messages for an update confined to a changed English line range."""
    start, end = line_range
    user = (
        f"修改后的英文文档全文（带行号）：\n\n{number_lines(english)}\n\n"
        f"发生修改的英文行号范围：第 {start} 行到第 {end} 行。\n\n"
        f"当前的中文版全文：\n\n{current_chinese}"
    )
    return [
        {"role": "system", "content": UPDATE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_payload(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """OpenAI-protocol chat body for a text-only translation call."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }


def _post(url: str, api_key: str, payload: dict, timeout: float) -> dict:
    """POST JSON and parse the response, or raise ``TranslationError``."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise TranslationError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TranslationError(f"connection failed: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        # Dropping /v1 from the base makes the relay serve its frontend HTML
        # with status 200, which lands here rather than as an HTTP error.
        raise TranslationError(
            f"response was not JSON (first 200 chars): {body[:200]!r}"
        ) from exc
    if "error" in parsed and not parsed.get("choices"):
        raise TranslationError(f"relay error: {parsed['error']}")
    return parsed


def _content_of(parsed: dict) -> str:
    """Pull the assistant text out of a chat completion, or raise."""
    try:
        choice = parsed["choices"][0]
    except (KeyError, IndexError) as exc:
        raise TranslationError(f"no choices in response: {parsed}") from exc
    text = (choice.get("message") or {}).get("content") or ""
    if isinstance(text, list):
        # Some relays mirror the request shape back and return content parts.
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    text = text.strip()
    if not text:
        raise TranslationError(
            "empty content "
            f"(finish_reason={choice.get('finish_reason')!r}); "
            "raise max_tokens if the reason is length"
        )
    return text


def translate_text(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str,
    api_key: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """One relay call for one translation. Raises on any unusable response."""
    payload = build_payload(
        messages, model=model, max_tokens=max_tokens, temperature=temperature
    )
    parsed = _post(f"{base_url}/chat/completions", api_key, payload, timeout)
    return _content_of(parsed)


def translate_with_retry(
    messages: list[dict],
    mode: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str,
    api_key: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: float = DEFAULT_TIMEOUT_S,
    attempts: int = DEFAULT_ATTEMPTS,
    sleep=time.sleep,
) -> Translation:
    """Translate one document, retrying transient relay failures.

    This relay returns 502/503, timeouts, and ``channel_all_disabled`` rate
    limits often enough that a first failure is not a config error. Backoff is
    deterministic (4 s, 8 s, ...) so a run is reproducible.
    """
    started = time.monotonic()
    last: TranslationError | None = None
    for attempt in range(1, attempts + 1):
        try:
            text = translate_text(
                messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except TranslationError as exc:
            last = exc
            if attempt < attempts:
                sleep(BACKOFF_BASE_S * 2 ** (attempt - 1))
            continue
        return Translation(
            text=text,
            model=model,
            mode=mode,
            elapsed_s=time.monotonic() - started,
            attempts=attempt,
        )
    raise TranslationError(f"translation failed after {attempts} attempt(s): {last}")
