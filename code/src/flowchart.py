"""方法流程图的编译：XeLaTeX 编译 TikZ 源文件，并导出 SVG 与 300 dpi PNG。"""

from pathlib import Path
import shutil
import subprocess

import fitz

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "paper/figures"
OUT_DIR = ROOT / "figures"
MAX_WIDTH_MM = 159
MAX_HEIGHT_MM = 244


def compile_flowchart(name):
    """编译 paper/figures/<name>.tex，输出 figures/<name>.{pdf,png,svg}，返回图幅（毫米）。"""
    compiler = shutil.which("xelatex")
    if compiler is None:
        raise RuntimeError("需要安装 XeLaTeX 并将其加入 PATH。")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={OUT_DIR}",
            f"{name}.tex",
        ],
        cwd=SOURCE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(result.stdout.decode("utf-8", errors="replace"))
    for suffix in (".aux", ".log"):
        (OUT_DIR / f"{name}{suffix}").unlink(missing_ok=True)
    pdf_path = OUT_DIR / f"{name}.pdf"
    with fitz.open(pdf_path) as document:
        if len(document) != 1:
            raise ValueError("流程图应为单页。")
        page = document[0]
        width_mm = page.rect.width * 25.4 / 72
        height_mm = page.rect.height * 25.4 / 72
        if width_mm > MAX_WIDTH_MM or height_mm > MAX_HEIGHT_MM:
            raise ValueError("流程图超出 A4 纵向版心。")
        page.get_pixmap(dpi=300).save(OUT_DIR / f"{name}.png")
        (OUT_DIR / f"{name}.svg").write_text(
            page.get_svg_image(text_as_path=True), encoding="utf-8"
        )
    return width_mm, height_mm
