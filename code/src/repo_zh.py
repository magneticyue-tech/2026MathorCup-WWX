"""Convert the repo's English working documents to Chinese, in place.

Import-only per the coding-pipeline convention: every function returns data,
none writes to disk. ``scripts/translate_repo_zh.py`` does the reading,
writing, and API calls.

This is a one-off language switch, unlike ``transcribe_for_communication``,
which regenerates a derived reader copy: the Chinese output replaces the
English source and becomes authoritative, so the prompt asks for a faithful
translation rather than a free rewrite, and ``preservation_diff`` checks that
every machine-relevant token survived.
"""

from __future__ import annotations

import re
from collections import Counter

DEFAULT_MODEL = "deepseek-flash"

SYSTEM_PROMPT = """\
你是一名数学建模团队的技术文档译者。下面给你一份英文的项目工作文档（协作规则、建模方法、技能说明、评审模板等）。
请把它完整翻译成简体中文。译文将直接替换英文原文，成为团队今后唯一的权威版本。

要求：
- 忠实完整：逐段翻译全部内容，不增补、不省略、不总结，不改变任何规则的含义和强弱（必须/应当/可以要区分）。
- 用地道、简洁的技术中文，避免翻译腔。
- 以下内容一律原样保留，不翻译、不改动：
  - 反引号内的行内代码和代码块（`...`、```...```），包括文件路径、文件名、命令、函数名、变量名；
  - URL、链接目标、[[双方括号链接]]；
  - 数学符号、公式、LaTeX（$...$、$$...$$）、数值、单位、有效位数；
  - 状态标记 [PLACEHOLDER]、[PENDING] 及 HTML 注释中的 Await: 关键字；
  - YAML frontmatter 的键名，以及枚举取值（如 open、ack、resolved、coding、modeling、writing、blocker、major、minor、decision、constraint、reference）；frontmatter 中的 name: 取值保持原样，description: 的取值翻译成中文；
  - 人名、产品名（Claude Code、Codex、Workbuddy、qwen、glm 等）。
- 保留原有 Markdown 结构：标题层级、列表层级、表格、引用块、空行分段一一对应。
- 写作规范类条款若举了英文措辞示例（如 significantly、"not merely X, but Y"），译成对应的中文措辞示例（如“显著”“不仅仅是 X，而是 Y”），使规则适用于中文写作。
- 只输出译文本身，不要任何前言、说明或代码围栏包裹。
"""

# Tokens that must survive translation unchanged. Code spans carry paths and
# identifiers; the rest are markers, links, math, and numbers.
_FENCE = re.compile(r"```.*?```", re.S)
_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"https?://[^\s)>\]]+")
_WIKI = re.compile(r"\[\[[^\]]+\]\]")
_MARKER = re.compile(r"\[(?:PLACEHOLDER|PENDING)\]")
_MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.S)
_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?")


def build_messages(english: str) -> list[dict]:
    """Chat messages for translating one whole document."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": english},
    ]


def strip_wrapping_fence(text: str) -> str:
    """Drop a ```markdown fence the model put around the whole answer."""
    m = re.fullmatch(r"```(?:markdown|md)?\n(.*)\n```\s*", text, re.S)
    return m.group(1) if m else text


def _tokens(text: str) -> dict[str, Counter]:
    fences = _FENCE.findall(text)
    body = _FENCE.sub("", text)
    return {
        "code_fence": Counter(fences),
        "code_span": Counter(_CODE.findall(body)),
        "url": Counter(_URL.findall(body)),
        "wiki_link": Counter(_WIKI.findall(body)),
        "marker": Counter(_MARKER.findall(body)),
        "math": Counter(_MATH.findall(_CODE.sub("", body))),
        "number": Counter(_NUMBER.findall(_CODE.sub("", body))),
        "heading": Counter(
            line.split(" ", 1)[0] for line in body.splitlines() if re.match(r"#+ ", line)
        ),
    }


def preservation_diff(english: str, chinese: str) -> dict[str, dict[str, list]]:
    """Tokens lost from or added to the translation, per category.

    Empty dict means every protected token survived. Numbers are reported but
    are the noisiest category: a translator may legitimately turn "two" into
    "2" or drop an ordinal, so number drift is reviewed by hand.
    """
    before, after = _tokens(english), _tokens(chinese)
    out: dict[str, dict[str, list]] = {}
    for key in before:
        lost = sorted((before[key] - after[key]).elements())
        added = sorted((after[key] - before[key]).elements())
        if lost or added:
            out[key] = {"lost": lost, "added": added}
    return out
