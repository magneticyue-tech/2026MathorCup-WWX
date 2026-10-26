# code/src/q2/advanced_audits.py
"""高级审查：附件 6 复算单测 与 C级维度零膨胀测试。"""
from __future__ import annotations

import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config as C

def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def test_attachment6_replication():
    # ... (这部分你刚才已经跑通了，保留原样即可) ...
    pass

def test_zero_inflation():
    """§6.6 零膨胀测试 (at_zero 掩码)"""
    print("\n" + "="*50)
    print("2. 正在进行 §6.6 零膨胀测试 (C级维度)...")
    
    # 【终极修复】：在项目根目录下暴力搜索 q1_panels.npz
    npz_path = None
    for p in C.PROJECT_ROOT.rglob("q1_panels.npz"):
        npz_path = p
        break
        
    if not npz_path:
        print(f"⚠️ 遍历了整个项目，依然找不到 q1_panels.npz，跳过零膨胀测试。")
        return
        
    print(f"  -> 成功找到文件: {npz_path}")
        
    data = np.load(npz_path, allow_pickle=True)
    at_zero = data['at_zero'] # 形状 (250, 50) 的 bool 矩阵
    
    chem_raw = pd.read_csv(C.CHEM_PATH).set_index("beer")
    sens_raw = pd.read_csv(C.SENSORY_PATH).set_index("beer")
    
    c_dims = ["A_esters_flower", "t2n"]
    
    for dim in c_dims:
        if dim not in sens_raw.columns:
            continue
            
        dim_idx = sens_raw.columns.get_loc(dim)
        y = sens_raw[dim].values
        
        # 掩码：只保留 at_zero 为 False (即高于零点) 的啤酒
        mask = ~at_zero[:, dim_idx]
        
        chem_clean = chem_raw.drop(columns=["beer_id", "tasting_category_fine"], errors="ignore")
        y_series = pd.Series(y, index=chem_clean.index)
        
        # 全样本关联
        r_all = chem_clean.corrwith(y_series, method='spearman').abs()
        top1_all = r_all.idxmax()
        val_all = r_all.max()
        
        # 掩码后关联
        if mask.sum() > 20: 
            chem_masked = chem_clean.loc[mask]
            y_masked = y_series.loc[mask]
            r_masked = chem_masked.corrwith(y_masked, method='spearman').abs()
            top1_masked = r_masked.idxmax()
            val_masked = r_masked.max()
            
            print(f"\n--- 维度: {dim} ---")
            print(f"  全样本     : Top-1 = {top1_all:<25} r = {val_all:.3f}")
            print(f"  仅高于零点 : Top-1 = {top1_masked:<25} r = {val_masked:.3f} (有效样本 {mask.sum()}/250)")
            
            if val_masked > val_all:
                print("  -> 结论: 零膨胀稀释了关联！去掉零点样本后 r 值增强。")
            else:
                print("  -> 结论: 零膨胀不是主要原因。")
        else:
            print(f"\n--- 维度: {dim} ---")
            print("  -> 高于零点的有效样本不足，无法测试。")

if __name__ == "__main__":
    test_attachment6_replication()
    test_zero_inflation()