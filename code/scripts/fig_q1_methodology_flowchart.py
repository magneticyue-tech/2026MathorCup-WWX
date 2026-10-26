"""使用 XeLaTeX 编译 Q1 矩形流程图，并导出 SVG 与 300 dpi PNG。"""
from pathlib import Path
import shutil
import subprocess

import fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures'
SOURCE = ROOT / 'paper/figures/q1_methodology_flowchart.tex'


def main():
    compiler = shutil.which('xelatex')
    if compiler is None:
        raise RuntimeError('需要安装 XeLaTeX 并将其加入 PATH。')
    OUT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [compiler, '-interaction=nonstopmode', '-halt-on-error',
         f'-output-directory={OUT}', str(SOURCE)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(result.stdout.decode('utf-8', errors='replace'))
    pdf_path = OUT / 'q1_methodology_flowchart.pdf'
    with fitz.open(pdf_path) as document:
        if len(document) != 1:
            raise ValueError('流程图应为单页。')
        page = document[0]
        width_mm = page.rect.width * 25.4 / 72
        height_mm = page.rect.height * 25.4 / 72
        if width_mm > 159 or height_mm > 244:
            raise ValueError('流程图超出 A4 纵向版心。')
        page.get_pixmap(dpi=300).save(OUT / 'q1_methodology_flowchart.png')
        (OUT / 'q1_methodology_flowchart.svg').write_text(
            page.get_svg_image(text_as_path=True), encoding='utf-8')
    print(f'已编译 PDF，并导出 SVG、PNG：{width_mm:.1f} x {height_mm:.1f} mm')


if __name__ == '__main__':
    main()
