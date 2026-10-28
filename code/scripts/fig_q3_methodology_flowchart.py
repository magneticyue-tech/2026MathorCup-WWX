"""编译问题 3 方法流程图：paper/figures/q3_methodology_flowchart.tex -> figures/。"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.flowchart import compile_flowchart

NAME = "q3_methodology_flowchart"


def main():
    width_mm, height_mm = compile_flowchart(NAME)
    print(f"已编译 {NAME}：{width_mm:.1f} x {height_mm:.1f} mm")


if __name__ == "__main__":
    main()
