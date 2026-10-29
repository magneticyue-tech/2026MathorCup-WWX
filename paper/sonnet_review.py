# -*- coding: utf-8 -*-
"""用 Sonnet（effort medium）逐段检查论文正文的独立可读性。

每个章节作为一次 API 调用，结果写入 paper/sonnet_review/ 下同名 txt。
凭据来自环境变量 ANTHROPIC_AUTH_TOKEN 与 ANTHROPIC_BASE_URL。
"""
import os
import sys
import time
import requests

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://www.rightapi.ai").rstrip("/")
AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
MODEL = "claude-sonnet-5-5"

SECTIONS = [
    ("ch1_problem", "问题重述与分析"),
    ("ch2_assumptions", "模型假设与符号"),
    ("preprocess", "数据预处理"),
    ("ch4_q1", "问题一"),
    ("ch5_q2", "问题二"),
    ("ch6_q3", "问题三"),
    ("ch7_evaluation", "模型评价与推广"),
]

SYSTEM = (
    "你是一名啤酒风味代谢组学与机器学习方向的审稿人。"
    "我写一篇竞赛论文，目标是让领域研究者能顺畅、快速理解并认可方法与结论，"
    "每一段文字、每一张图表都能被独立理解。"
    "请逐段检查我给出的 LaTeX 章节，只关注可读性与独立可理解性，不评价方法本身的优劣。"
    "输出中文。"
)

TASK = (
    "请逐段检查下面的章节，指出任何一段无法被领域研究者独立理解的地方。"
    "具体关注："
    "(1) 段落里出现未在本段或前文定义、又不在章节符号表中的符号、代号或缩写；"
    "(2) 引用了表格或图却没有足够上下文，读者不知道在看什么；"
    "(3) 段落依赖后文或另一章才成立的表述；"
    "(4) 破折号、比喻、口语化、不必要的括号等语言问题；"
    "(5) 表格或图的标题、坐标、图例使用了简称、代号或内部术语。"
    "对每个问题给出：位置（小节名或段首几个字）、问题、以及一行修改建议。"
    "如果没有问题，明确写“无问题”。不要重写整章，只列问题清单。"
)


def call_section(name, title, tex):
    url = BASE_URL + "/v1/messages"
    headers = {
        "authorization": "Bearer " + AUTH_TOKEN,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": MODEL,
        "max_tokens": 4000,
        "output_config": {"effort": "medium"},
        "system": SYSTEM,
        "messages": [
            {"role": "user", "content": f"章节：{title}\n\n{TASK}\n\n=== 章节正文开始 ===\n{tex}\n=== 章节正文结束 ==="}
        ],
    }
    r = requests.post(url, headers=headers, json=body, timeout=600)
    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text[:2000]}"
    data = r.json()
    out = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            out.append(block["text"])
    return "\n".join(out).strip()


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonnet_review")
    os.makedirs(outdir, exist_ok=True)
    for name, title in SECTIONS:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sections", name + ".tex")
        if not os.path.exists(path):
            print(f"[skip] {name}: 文件不存在", flush=True)
            continue
        tex = open(path, encoding="utf-8").read()
        print(f"[start] {name} ({title})", flush=True)
        try:
            result = call_section(name, title, tex)
        except Exception as e:
            result = f"EXCEPTION: {e!r}"
        with open(os.path.join(outdir, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[done ] {name} -> {len(result)} chars", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
