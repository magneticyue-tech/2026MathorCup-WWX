# code/src/q2/run_all_q2.py
"""一键运行 Q2 所有核心流程（图->Top3->显著性->Bootstrap->合理性->对比标识->画图）。"""
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from graph import build_coexpression_graph
from association import compute_top3
from inference import run_permutation_test
from stability import run_bootstrap
from rationality import run_rationality
from method_flags import add_flags
from pipeline_final import plot_heatmap

if __name__ == "__main__":
    print("🚀 开始一键执行 Q2 全流程...\n")
    
    print("--- [1/7] 构建共表达图 ---")
    build_coexpression_graph()
    
    print("\n--- [2/7] 计算感官维度 Top-3 ---")
    compute_top3()
    
    print("\n--- [3/7] 置换检验 (显著性) ---")
    run_permutation_test()
    
    print("\n--- [4/7] Bootstrap 稳定性检验 (耗时约1-2分钟) ---")
    run_bootstrap()
    
    print("\n--- [5/7] 合理性验证与 M3 并列合并 (耗时约1-2分钟) ---")
    run_rationality()
    
    print("\n--- [6/7] 追加对比方法标识列 ---")
    add_flags()
    
    print("\n--- [7/7] 绘制热力图 ---")
    plot_heatmap()
    
    print("\n🎉 全流程完美收官！请去 code/data/processed/q2/ 和 figures/q2/ 检查产出！")